#include "sore/cuda/scan.cuh"
#include <algorithm>
#include <stdexcept>

namespace sore {
namespace cuda {

/**
 * @brief Kernel 1: Warp-Coalesced Lockstep Scan.
 * 
 * Cada Warp de 32 threads processa 32 canais adjacentes de 'D' simultaneamente.
 * Em cada timestep 't', as 32 threads realizam leituras e escritas perfeitamente
 * coalescidas de 128 bytes em memória global, mantendo o estado 'h' em registradores.
 */
__global__ void linear_rnn_coalesced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ X,
    const float* __restrict__ h0,
    float* __restrict__ H,
    size_t B,
    size_t T,
    size_t D
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || d >= D) return;

    // Estado inicial h_{-1} mantido em registrador de altíssima velocidade
    float h = (h0 != nullptr) ? h0[b * D + d] : 0.0f;

    // Loop temporal com acessos contíguos em D (coalescimento de 128 bytes por warp)
    for (size_t t = 0; t < T; ++t) {
        size_t idx = (b * T + t) * D + d;
        float a_t = A[idx];
        float x_t = X[idx];

        // Recorrência Linear: h_t = a_t * h_{t-1} + x_t
        h = a_t * h + x_t;
        H[idx] = h;
    }
}

/**
 * @brief Kernel 2: Parallel Associative Scan em Shared Memory (Kogge-Stone).
 * 
 * Paraleliza diretamente a dimensão temporal T em O(log T) passos paralelos
 * utilizando a álgebra associativa de tuplas (a_t, x_t):
 * (a_i, x_i) \bullet (a_j, x_j) = (a_i * a_j,  a_j * x_i + x_j)
 */
__global__ void associative_scan_kogge_stone_kernel(
    const float* __restrict__ A,
    const float* __restrict__ X,
    const float* __restrict__ h0,
    float* __restrict__ H,
    size_t B,
    size_t T,
    size_t D
) {
    // Cada bloco processa uma sequência inteira no tempo para um dado par (b, d)
    size_t d = blockIdx.x;
    size_t b = blockIdx.y;

    if (b >= B || d >= D) return;

    extern __shared__ float smem[];
    float* s_a = smem;               // Tamanho: blockDim.x
    float* s_x = smem + blockDim.x;   // Tamanho: blockDim.x

    size_t t = threadIdx.x;

    // Carrega dados da memória global para a Shared Memory
    if (t < T) {
        size_t idx = (b * T + t) * D + d;
        s_a[t] = A[idx];
        s_x[t] = X[idx];
    } else {
        // Elemento neutro da operação associativa: (1.0, 0.0)
        s_a[t] = 1.0f;
        s_x[t] = 0.0f;
    }
    __syncthreads();

    // Kogge-Stone Parallel Associative Scan na Shared Memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float a_prev = 1.0f;
        float x_prev = 0.0f;
        if (t >= static_cast<size_t>(stride)) {
            a_prev = s_a[t - stride];
            x_prev = s_x[t - stride];
        }
        __syncthreads();

        if (t >= static_cast<size_t>(stride)) {
            // Operação associativa: p_{prev} \bullet p_{curr}
            s_x[t] = s_a[t] * x_prev + s_x[t];
            s_a[t] = s_a[t] * a_prev;
        }
        __syncthreads();
    }

    // Aplicação do estado inicial h0 se fornecido
    if (t < T) {
        size_t idx = (b * T + t) * D + d;
        float final_h = s_x[t];
        if (h0 != nullptr) {
            final_h += s_a[t] * h0[b * D + d];
        }
        H[idx] = final_h;
    }
}

Tensor linear_rnn_forward_cuda(const Tensor& A, const Tensor& X, const Tensor* h0) {
    if (A.device() != Device::CUDA || X.device() != Device::CUDA) {
        throw std::runtime_error("linear_rnn_forward_cuda requer tensores na GPU.");
    }
    if (A.shape() != X.shape() || A.rank() != 3) {
        throw std::runtime_error("A e X devem ser tensores 3D [B, T, D] com mesmo formato.");
    }

    size_t B = A.dim(0);
    size_t T = A.dim(1);
    size_t D = A.dim(2);

    Tensor H(A.shape(), DType::Float32, Device::CUDA);

    // Configuração de grid/block: Warp coalescing ao longo de D (32 threads) e batch (8)
    dim3 block(32, 8);
    dim3 grid(
        static_cast<unsigned int>((D + block.x - 1) / block.x),
        static_cast<unsigned int>((B + block.y - 1) / block.y)
    );

    const float* h0_ptr = (h0 && h0->is_defined()) ? h0->data<float>() : nullptr;

    linear_rnn_coalesced_kernel<<<grid, block>>>(
        A.data<float>(),
        X.data<float>(),
        h0_ptr,
        H.data<float>(),
        B, T, D
    );
    CUDA_SYNC_CHECK();

    return H;
}

Tensor associative_scan_forward_cuda(const Tensor& A, const Tensor& X, const Tensor* h0) {
    if (A.device() != Device::CUDA || X.device() != Device::CUDA) {
        throw std::runtime_error("associative_scan_forward_cuda requer tensores na GPU.");
    }
    size_t B = A.dim(0);
    size_t T = A.dim(1);
    size_t D = A.dim(2);

    Tensor H(A.shape(), DType::Float32, Device::CUDA);

    // Arredonda T para a próxima potência de 2 para o Kogge-Stone scan
    size_t threads = 1;
    while (threads < T) threads *= 2;
    if (threads > 1024) {
        throw std::runtime_error("associative_scan_forward_cuda para T > 1024 requer múltiplos blocos.");
    }

    dim3 block(static_cast<unsigned int>(threads));
    dim3 grid(static_cast<unsigned int>(D), static_cast<unsigned int>(B));

    size_t smem_bytes = 2 * threads * sizeof(float);
    const float* h0_ptr = (h0 && h0->is_defined()) ? h0->data<float>() : nullptr;

    associative_scan_kogge_stone_kernel<<<grid, block, smem_bytes>>>(
        A.data<float>(),
        X.data<float>(),
        h0_ptr,
        H.data<float>(),
        B, T, D
    );
    CUDA_SYNC_CHECK();

    return H;
}

} // namespace cuda
} // namespace sore
