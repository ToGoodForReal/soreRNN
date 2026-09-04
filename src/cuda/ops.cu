#include "sore/cuda/ops.cuh"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace sore {
namespace cuda {

// Kernel de broadcast e adição de viés com grid-stride loop
__global__ void add_bias_kernel(float* __restrict__ Y, const float* __restrict__ bias, size_t M, size_t D_out) {
    size_t total = M * D_out;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        size_t j = i % D_out;
        Y[i] += bias[j];
    }
}

// Kernel vetorial de ativação Sigmóide utilizando a SFU (Special Function Unit da GPU)
__global__ void sigmoid_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float val = in[i];
        out[i] = 1.0f / (1.0f + __expf(-val));
    }
}

void add_bias_cuda_(Tensor& Y, const Tensor& bias) {
    if (Y.device() != Device::CUDA || bias.device() != Device::CUDA) {
        throw std::runtime_error("add_bias_cuda_ requer tensores na GPU.");
    }
    size_t D_out = bias.numel();
    size_t total = Y.numel();
    size_t M = total / D_out;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((total + threads - 1) / threads, size_t(1024)));

    add_bias_kernel<<<blocks, threads>>>(Y.data<float>(), bias.data<float>(), M, D_out);
    CUDA_SYNC_CHECK();
}

Tensor sigmoid_cuda(const Tensor& x) {
    if (x.device() != Device::CUDA) {
        throw std::runtime_error("sigmoid_cuda requer tensor na GPU.");
    }
    Tensor out(x.shape(), x.dtype(), Device::CUDA);
    size_t n = x.numel();
    if (n == 0) return out;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    sigmoid_kernel<<<blocks, threads>>>(x.data<float>(), out.data<float>(), n);
    CUDA_SYNC_CHECK();
    return out;
}

Tensor matmul2d_cuda(const Tensor& A, const Tensor& B) {
    if (A.device() != Device::CUDA || B.device() != Device::CUDA) {
        throw std::runtime_error("matmul2d_cuda requer tensores na GPU.");
    }
    if (A.rank() != 2 || B.rank() != 2) {
        throw std::runtime_error("matmul2d_cuda requer tensores 2D.");
    }
    size_t M = A.dim(0);
    size_t K = A.dim(1);
    size_t K2 = B.dim(0);
    size_t N = B.dim(1);

    if (K != K2) {
        std::ostringstream oss;
        oss << "Dimensões incompatíveis para matmul: [" << M << "," << K << "] e [" << K2 << "," << N << "]";
        throw std::runtime_error(oss.str());
    }

    Tensor C({M, N}, DType::Float32, Device::CUDA);
    cublasHandle_t handle = get_cublas_handle();

    // Mapeamento Row-Major para Column-Major cuBLAS:
    // C_row = A_row * B_row  <=>  C_col^T = B_col^T * A_col^T
    // Passamos B como primeira matriz e A como segunda, dimensões N, M, K
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
        &alpha,
        B.data<float>(), static_cast<int>(N),
        A.data<float>(), static_cast<int>(K),
        &beta,
        C.data<float>(), static_cast<int>(N)
    ));

    return C;
}

Tensor linear_cuda(const Tensor& X, const Tensor& W, const Tensor* bias) {
    if (X.device() != Device::CUDA || W.device() != Device::CUDA) {
        throw std::runtime_error("linear_cuda requer tensores na GPU.");
    }
    if (W.rank() != 2) {
        throw std::runtime_error("W deve ser 2D [D_out, D_in].");
    }

    size_t D_out = W.dim(0);
    size_t D_in = W.dim(1);

    if (X.shape().back() != D_in) {
        throw std::runtime_error("Última dimensão de X deve ser igual a D_in de W.");
    }

    size_t M = X.numel() / D_in;
    std::vector<size_t> out_shape = X.shape();
    out_shape.back() = D_out;

    Tensor Y({M, D_out}, DType::Float32, Device::CUDA);
    cublasHandle_t handle = get_cublas_handle();

    // Y = X @ W^T
    // Em Column-major: Y^T = W @ X^T
    // W é [D_out, D_in] (Row-major) -> W_col é [D_in, D_out] com ld=D_in
    // Aplicando CUBLAS_OP_T em W_col, obtemos [D_out, D_in]
    // X é [M, D_in] (Row-major) -> X_col é [D_in, M] com ld=D_in (CUBLAS_OP_N)
    // Resultado: [D_out, M] com ld=D_out, correspondendo a [M, D_out] em Row-major!
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(D_out), static_cast<int>(M), static_cast<int>(D_in),
        &alpha,
        W.data<float>(), static_cast<int>(D_in),
        X.data<float>(), static_cast<int>(D_in),
        &beta,
        Y.data<float>(), static_cast<int>(D_out)
    ));

    // Adiciona viés se fornecido
    if (bias && bias->is_defined()) {
        add_bias_cuda_(Y, *bias);
    }

    return Y.view(out_shape);
}

} // namespace cuda
} // namespace sore
