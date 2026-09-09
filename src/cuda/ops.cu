#include "sore/cuda/ops.cuh"
#include "sore/cuda/caching_allocator.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace sore {
namespace cuda {

// ===================== Suporte a precisao mista (BF16) =====================
#include <cuda_bf16.h>
#include <unordered_map>

__global__ void fp32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n){
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for(; i < n; i += stride){ dst[i] = __float2bfloat16(src[i]); }
}
static inline void launch_fp32_to_bf16(const float* src, void* dst, size_t n){
    if(n==0) return;
    constexpr int threads=256;
    int blocks=(int)std::min((n + threads - 1)/(size_t)threads, (size_t)8192);
    fp32_to_bf16_kernel<<<blocks,threads>>>(src, (__nv_bfloat16*)dst, n);
}

// Espelhos BF16 persistentes das matrizes de peso FP32 (atualizados 1x por passo do otimizador).
struct Bf16Mirror { void* ptr=nullptr; size_t numel=0; };
static std::unordered_map<const void*, Bf16Mirror>& bf16_mirrors(){
    static std::unordered_map<const void*, Bf16Mirror> m; return m;
}
static void* bf16_mirror_of(const void* key){
    auto& m = bf16_mirrors(); auto it=m.find(key); return it==m.end()? nullptr : it->second.ptr;
}
void refresh_bf16_weights(const std::vector<Tensor*>& params){
    auto& m = bf16_mirrors();
    for(auto* t : params){
        if(!t || !t->is_defined() || t->device()!=Device::CUDA) continue;
        if(t->rank()!=2) continue;
        if(t->dtype()!=DType::Float32) continue;
        const float* wp = t->data<float>();
        size_t n = t->numel();
        Bf16Mirror& mir = m[wp];
        if(mir.numel!=n){
            if(mir.ptr) CUDACachingAllocator::instance().deallocate(mir.ptr);
            mir.ptr = CUDACachingAllocator::instance().allocate(n*sizeof(__nv_bfloat16));
            mir.numel = n;
        }
        launch_fp32_to_bf16(wp, mir.ptr, n);
    }
}
void clear_bf16_weights(){
    auto& m = bf16_mirrors();
    for(auto& kv : m){ if(kv.second.ptr) CUDACachingAllocator::instance().deallocate(kv.second.ptr); }
    m.clear();
}
static void* bf16_cast_tmp(const float* src, size_t n){
    void* p = CUDACachingAllocator::instance().allocate(n*sizeof(__nv_bfloat16));
    launch_fp32_to_bf16(src, p, n);
    return p;
}

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

    void* Wbf = bf16_mirror_of((const void*)W.data<float>());
    if (Wbf) {
        void* Xbf = bf16_cast_tmp(X.data<float>(), M * D_in);
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(D_out), static_cast<int>(M), static_cast<int>(D_in),
            &alpha,
            Wbf, CUDA_R_16BF, static_cast<int>(D_in),
            Xbf, CUDA_R_16BF, static_cast<int>(D_in),
            &beta,
            Y.data<float>(), CUDA_R_32F, static_cast<int>(D_out),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUDACachingAllocator::instance().deallocate(Xbf);
    } else {
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
    }

    // Adiciona viés se fornecido
    if (bias && bias->is_defined()) {
        add_bias_cuda_(Y, *bias);
    }

    return Y.view(out_shape);
}

__global__ void sum_rows_bias_kernel(const float* __restrict__ dY, float* __restrict__ dbias, size_t M, size_t D_out) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < D_out) {
        float s = 0.0f;
        for (size_t row = 0; row < M; ++row) {
            s += dY[row * D_out + col];
        }
        dbias[col] += s;
    }
}

void linear_backward_cuda(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& W,
    Tensor& dX,
    Tensor& dW,
    Tensor* dbias
) {
    if (dY.device() != Device::CUDA || X.device() != Device::CUDA || W.device() != Device::CUDA) {
        throw std::runtime_error("linear_backward_cuda requer tensores na GPU.");
    }
    size_t D_out = W.dim(0);
    size_t D_in = W.dim(1);
    size_t M = X.numel() / D_in;

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta_dx = 0.0f;
    const float beta_dw = 1.0f; // acumula em dW

    // ---- Precisao mista: usa espelho BF16 de W se existir ----
    void* Wbf = bf16_mirror_of((const void*)W.data<float>());
    void* dyb = Wbf ? bf16_cast_tmp(dY.data<float>(), M * D_out) : nullptr;

    // dX = dY @ W
    if (Wbf) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(D_in), static_cast<int>(M), static_cast<int>(D_out),
            &alpha,
            Wbf, CUDA_R_16BF, static_cast<int>(D_in),
            dyb, CUDA_R_16BF, static_cast<int>(D_out),
            &beta_dx,
            dX.data<float>(), CUDA_R_32F, static_cast<int>(D_in),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        CUBLAS_CHECK(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(D_in), static_cast<int>(M), static_cast<int>(D_out),
            &alpha,
            W.data<float>(), static_cast<int>(D_in),
            dY.data<float>(), static_cast<int>(D_out),
            &beta_dx,
            dX.data<float>(), static_cast<int>(D_in)
        ));
    }

    // dW += dY^T @ X
    if (Wbf) {
        void* xbf = bf16_cast_tmp(X.data<float>(), M * D_in);
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            static_cast<int>(D_in), static_cast<int>(D_out), static_cast<int>(M),
            &alpha,
            xbf, CUDA_R_16BF, static_cast<int>(D_in),
            dyb, CUDA_R_16BF, static_cast<int>(D_out),
            &beta_dw,
            dW.data<float>(), CUDA_R_32F, static_cast<int>(D_in),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUDACachingAllocator::instance().deallocate(xbf);
    } else {
        CUBLAS_CHECK(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            static_cast<int>(D_in), static_cast<int>(D_out), static_cast<int>(M),
            &alpha,
            X.data<float>(), static_cast<int>(D_in),
            dY.data<float>(), static_cast<int>(D_out),
            &beta_dw,
            dW.data<float>(), static_cast<int>(D_in)
        ));
    }
    if (dyb) CUDACachingAllocator::instance().deallocate(dyb);

    if (dbias && dbias->is_defined()) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((D_out + threads - 1) / threads);
        sum_rows_bias_kernel<<<blocks, threads>>>(dY.data<float>(), dbias->data<float>(), M, D_out);
        CUDA_SYNC_CHECK();
    }
}

__global__ void embedding_forward_kernel(
    const uint16_t* __restrict__ tokens,
    const float* __restrict__ weight,
    float* __restrict__ out,
    size_t total_elements,
    size_t emb_dim,
    size_t vocab_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        size_t token_idx = idx / emb_dim;
        size_t dim_idx = idx % emb_dim;
        uint16_t tok = tokens[token_idx];
        if (tok >= vocab_size) tok = 0;
        out[idx] = weight[tok * emb_dim + dim_idx];
    }
}

Tensor embedding_forward_cuda(
    const std::vector<uint16_t>& tokens,
    const Tensor& weight,
    size_t B,
    size_t T
) {
    size_t N = B * T;
    size_t D = weight.dim(1);
    size_t V = weight.dim(0);
    size_t total = N * D;

    Tensor out({B, T, D}, DType::Float32, Device::CUDA);

    // Aloca buffer temporário na GPU via Caching Allocator (zero overhead de driver)
    uint16_t* d_tokens = static_cast<uint16_t*>(CUDACachingAllocator::instance().allocate(N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    embedding_forward_kernel<<<blocks, threads>>>(
        d_tokens, weight.data<float>(), out.data<float>(), total, D, V
    );
    CUDA_SYNC_CHECK();
    CUDACachingAllocator::instance().deallocate(d_tokens);

    return out;
}

__global__ void embedding_backward_kernel(
    const uint16_t* __restrict__ tokens,
    const float* __restrict__ d_out,
    float* __restrict__ d_weight,
    size_t total_elements,
    size_t emb_dim,
    size_t vocab_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        size_t token_idx = idx / emb_dim;
        size_t dim_idx = idx % emb_dim;
        uint16_t tok = tokens[token_idx];
        if (tok < vocab_size) {
            atomicAdd(&d_weight[tok * emb_dim + dim_idx], d_out[idx]);
        }
    }
}

void embedding_backward_cuda(
    const Tensor& d_out,
    const std::vector<uint16_t>& tokens,
    Tensor& d_weight
) {
    size_t N = tokens.size();
    size_t D = d_weight.dim(1);
    size_t V = d_weight.dim(0);
    size_t total = N * D;

    uint16_t* d_tokens = static_cast<uint16_t*>(CUDACachingAllocator::instance().allocate(N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    embedding_backward_kernel<<<blocks, threads>>>(
        d_tokens, d_out.data<float>(), d_weight.data<float>(), total, D, V
    );
    CUDA_SYNC_CHECK();
    CUDACachingAllocator::instance().deallocate(d_tokens);
}

__global__ void cross_entropy_cuda_kernel(
    const float* __restrict__ logits,
    const uint16_t* __restrict__ targets,
    float* __restrict__ d_logits,
    float* __restrict__ losses,
    size_t N,
    size_t V,
    float inv_active_n
) {
    size_t row = blockIdx.x;
    if (row >= N) return;

    size_t tid = threadIdx.x;
    size_t bdim = blockDim.x;
    float* dlog_row = &d_logits[row * V];
    uint16_t target = targets[row];

    // Se o token for mascarado (ex: IGNORE_INDEX = 65535 ou qualquer valor >= V):
    // Loss e gradientes correspondentes são EXATAMENTE zero!
    if (target >= V) {
        if (tid == 0) {
            losses[row] = 0.0f;
        }
        for (size_t col = tid; col < V; col += bdim) {
            dlog_row[col] = 0.0f;
        }
        return;
    }

    const float* log_row = &logits[row * V];

    // 1. Encontrar o valor máximo
    float thread_max = -1e30f;
    for (size_t col = tid; col < V; col += bdim) {
        float val = log_row[col];
        if (val > thread_max) thread_max = val;
    }

    extern __shared__ float sdata[];
    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // 2. Soma de exponenciais
    float thread_sum = 0.0f;
    for (size_t col = tid; col < V; col += bdim) {
        float e = __expf(log_row[col] - max_val);
        dlog_row[col] = e;
        thread_sum += e;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / (sum_exp + 1e-12f);

    // 3. Loss = -log(P_target)
    if (tid == 0) {
        float p_target = dlog_row[target] * inv_sum;
        losses[row] = -__logf(fmaxf(p_target, 1e-12f));
    }

    // 4. dLogits = (Softmax - OneHot) * inv_active_n
    for (size_t col = tid; col < V; col += bdim) {
        float p = dlog_row[col] * inv_sum;
        float grad = p * inv_active_n;
        if (col == target) {
            grad -= inv_active_n;
        }
        dlog_row[col] = grad;
    }
}

float cross_entropy_loss_and_grad_cuda(
    const Tensor& logits,
    const std::vector<uint16_t>& targets,
    Tensor& d_logits
) {
    size_t N = targets.size();
    size_t V = logits.shape().back();

    size_t active_count = 0;
    for (size_t i = 0; i < N; ++i) {
        if (targets[i] < V) {
            active_count++;
        }
    }
    float inv_active_n = active_count > 0 ? (1.0f / static_cast<float>(active_count)) : 0.0f;

    uint16_t* d_targets = static_cast<uint16_t*>(CUDACachingAllocator::instance().allocate(N * sizeof(uint16_t)));
    float* d_losses = static_cast<float*>(CUDACachingAllocator::instance().allocate(N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    size_t smem = threads * sizeof(float);

    cross_entropy_cuda_kernel<<<static_cast<unsigned int>(N), threads, smem>>>(
        logits.data<float>(), d_targets, d_logits.data<float>(), d_losses, N, V, inv_active_n
    );
    CUDA_SYNC_CHECK();

    // Redução da loss na CPU (apenas sobre tokens ativos)
    std::vector<float> h_losses(N);
    CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDACachingAllocator::instance().deallocate(d_targets);
    CUDACachingAllocator::instance().deallocate(d_losses);

    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        if (targets[i] < V) {
            sum += h_losses[i];
        }
    }

    return active_count > 0 ? static_cast<float>(sum / static_cast<double>(active_count)) : 0.0f;
}

__global__ void add_residual_kernel(float* __restrict__ x, const float* __restrict__ res, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        x[idx] += res[idx];
    }
}

void add_residual_cuda(Tensor& x, const Tensor& residual) {
    size_t total = x.numel();
    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    add_residual_kernel<<<blocks, threads>>>(x.data<float>(), residual.data<float>(), total);
    CUDA_SYNC_CHECK();
}

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ rstd,
    size_t M,
    size_t D,
    float eps
) {
    size_t m = blockIdx.x;
    if (m >= M) return;

    size_t tid = threadIdx.x;
    const float* x_row = &x[m * D];
    float* y_row = &y[m * D];

    float sum_sq = 0.0f;
    for (size_t d = tid; d < D; d += blockDim.x) {
        float val = x_row[d];
        sum_sq += val * val;
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / static_cast<float>(D) + eps);
    if (tid == 0) {
        rstd[m] = inv_rms;
        sdata[0] = inv_rms;
    }
    __syncthreads();

    inv_rms = sdata[0];

    for (size_t d = tid; d < D; d += blockDim.x) {
        y_row[d] = x_row[d] * inv_rms * gamma[d];
    }
}

Tensor rmsnorm_forward_cuda(const Tensor& x, const Tensor& gamma, Tensor& rstd, float eps) {
    size_t D = gamma.numel();
    size_t M = x.numel() / D;

    Tensor y(x.shape(), DType::Float32, Device::CUDA);
    rstd = Tensor({M}, DType::Float32, Device::CUDA);

    constexpr int threads = 256;
    size_t smem = threads * sizeof(float);
    rmsnorm_forward_kernel<<<static_cast<unsigned int>(M), threads, smem>>>(
        x.data<float>(), gamma.data<float>(), y.data<float>(), rstd.data<float>(), M, D, eps
    );
    CUDA_SYNC_CHECK();
    return y;
}

__global__ void rmsnorm_backward_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ rstd,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    size_t M,
    size_t D
) {
    size_t m = blockIdx.x;
    if (m >= M) return;

    size_t tid = threadIdx.x;
    const float* dout_row = &dout[m * D];
    const float* x_row = &x[m * D];
    float* dx_row = &dx[m * D];
    float inv_rms = rstd[m];

    // Soma sum(dout * gamma * x)
    float sum_val = 0.0f;
    for (size_t d = tid; d < D; d += blockDim.x) {
        sum_val += dout_row[d] * gamma[d] * x_row[d];
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_proj = sdata[0];
    __syncthreads();

    float factor = sum_proj * (inv_rms * inv_rms) / static_cast<float>(D);

    for (size_t d = tid; d < D; d += blockDim.x) {
        float dy = dout_row[d];
        float x_val = x_row[d];
        dx_row[d] = inv_rms * (dy * gamma[d] - x_val * factor);
        atomicAdd(&dgamma[d], dy * x_val * inv_rms);
    }
}

void rmsnorm_backward_cuda(
    const Tensor& dout,
    const Tensor& x,
    const Tensor& gamma,
    const Tensor& rstd,
    Tensor& dx,
    Tensor& dgamma
) {
    size_t D = gamma.numel();
    size_t M = x.numel() / D;

    constexpr int threads = 256;
    size_t smem = threads * sizeof(float);
    rmsnorm_backward_kernel<<<static_cast<unsigned int>(M), threads, smem>>>(
        dout.data<float>(), x.data<float>(), gamma.data<float>(), rstd.data<float>(),
        dx.data<float>(), dgamma.data<float>(), M, D
    );
    CUDA_SYNC_CHECK();
}

__global__ void sum_sq_kernel(const float* __restrict__ data, double* __restrict__ result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        double v = static_cast<double>(data[i]);
        thread_sum += v * v;
    }
    extern __shared__ double sdata_d[];
    sdata_d[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_d[threadIdx.x] += sdata_d[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, sdata_d[0]);
    }
}

__global__ void scale_tensor_kernel(float* __restrict__ data, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        data[i] *= scale;
    }
}

void scale_tensor_cuda(Tensor& t, float scale) {
    if (t.device() != Device::CUDA) {
        throw std::runtime_error("scale_tensor_cuda requer tensor na GPU.");
    }
    size_t n = t.numel();
    if (n == 0) return;
    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));
    scale_tensor_kernel<<<blocks, threads>>>(t.data<float>(), scale, n);
    CUDA_SYNC_CHECK();
}

float clip_grad_norm_cuda(std::vector<Tensor*>& grads, float max_norm) {
    double* d_sum_sq = static_cast<double*>(CUDACachingAllocator::instance().allocate(sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_sq, 0, sizeof(double)));

    constexpr int threads = 256;
    for (auto* g : grads) {
        if (!g || !g->is_defined() || g->numel() == 0) continue;
        size_t n = g->numel();
        int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));
        size_t smem = threads * sizeof(double);
        sum_sq_kernel<<<blocks, threads, smem>>>(g->data<float>(), d_sum_sq, n);
    }
    CUDA_SYNC_CHECK();

    double h_sum_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost));
    CUDACachingAllocator::instance().deallocate(d_sum_sq);

    float total_norm = static_cast<float>(std::sqrt(h_sum_sq));
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto* g : grads) {
            if (!g || !g->is_defined() || g->numel() == 0) continue;
            size_t n = g->numel();
            int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));
            scale_tensor_kernel<<<blocks, threads>>>(g->data<float>(), scale, n);
        }
        CUDA_SYNC_CHECK();
    }
    return total_norm;
}

// ==============================================================================
//  Ativação GELU (Gaussian Error Linear Unit)
// ==============================================================================

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + erff(x * INV_SQRT2));
    }
}

__global__ void gelu_backward_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ in,
    float* __restrict__ din,
    size_t n
) {
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    constexpr float INV_SQRT_2PI = 0.3989422804014327f;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = in[i];
        float cdf = 0.5f * (1.0f + erff(x * INV_SQRT2));
        float pdf = INV_SQRT_2PI * __expf(-0.5f * x * x);
        din[i] = dout[i] * (cdf + x * pdf);
    }
}

Tensor gelu_cuda(const Tensor& x) {
    if (x.device() != Device::CUDA) {
        throw std::runtime_error("gelu_cuda requer tensor na GPU.");
    }
    Tensor out(x.shape(), x.dtype(), Device::CUDA);
    size_t n = x.numel();
    if (n == 0) return out;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    gelu_kernel<<<blocks, threads>>>(x.data<float>(), out.data<float>(), n);
    CUDA_SYNC_CHECK();
    return out;
}

Tensor gelu_backward_cuda(const Tensor& dout, const Tensor& x) {
    if (dout.device() != Device::CUDA || x.device() != Device::CUDA) {
        throw std::runtime_error("gelu_backward_cuda requer tensores na GPU.");
    }
    Tensor dx(x.shape(), x.dtype(), Device::CUDA);
    size_t n = x.numel();
    if (n == 0) return dx;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    gelu_backward_kernel<<<blocks, threads>>>(dout.data<float>(), x.data<float>(), dx.data<float>(), n);
    CUDA_SYNC_CHECK();
    return dx;
}

// ==============================================================================
//  Causal Depthwise Conv1D (Kernel Size K=4)
// ==============================================================================

__global__ void conv1d_causal_depthwise_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    size_t B,
    size_t T,
    size_t D,
    size_t total
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        size_t d = i % D;
        size_t t = (i / D) % T;
        size_t b = i / (T * D);

        float val = (bias != nullptr) ? bias[d] : 0.0f;
        const float* w_d = W + d * 4;

        if (t >= 3) {
            size_t base = (b * T + (t - 3)) * D + d;
            val += X[base] * w_d[0];
            val += X[base + D] * w_d[1];
            val += X[base + 2 * D] * w_d[2];
            val += X[base + 3 * D] * w_d[3];
        } else {
            if (t >= 2) val += X[(b * T + t - 2) * D + d] * w_d[1];
            if (t >= 1) val += X[(b * T + t - 1) * D + d] * w_d[2];
            val += X[(b * T + t) * D + d] * w_d[3];
        }
        Y[i] = val;
    }
}

__global__ void conv1d_causal_depthwise_backward_dx_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ W,
    float* __restrict__ dX,
    size_t B,
    size_t T,
    size_t D,
    size_t total
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < total; i += blockDim.x * gridDim.x) {
        size_t d = i % D;
        size_t t = (i / D) % T;
        size_t b = i / (T * D);

        const float* w_d = W + d * 4;
        float val = 0.0f;

        val += dY[(b * T + t) * D + d] * w_d[3];
        if (t + 1 < T) val += dY[(b * T + t + 1) * D + d] * w_d[2];
        if (t + 2 < T) val += dY[(b * T + t + 2) * D + d] * w_d[1];
        if (t + 3 < T) val += dY[(b * T + t + 3) * D + d] * w_d[0];

        dX[i] = val;
    }
}

__global__ void conv1d_causal_depthwise_backward_weights_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dW,
    float* __restrict__ dbias,
    size_t B,
    size_t T,
    size_t D
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    float dw0 = 0.0f;
    float dw1 = 0.0f;
    float dw2 = 0.0f;
    float dw3 = 0.0f;
    float db  = 0.0f;

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            float dy = dY[(b * T + t) * D + d];
            db += dy;

            dw3 += dy * X[(b * T + t) * D + d];
            if (t >= 1) dw2 += dy * X[(b * T + t - 1) * D + d];
            if (t >= 2) dw1 += dy * X[(b * T + t - 2) * D + d];
            if (t >= 3) dw0 += dy * X[(b * T + t - 3) * D + d];
        }
    }

    atomicAdd(&dW[d * 4 + 0], dw0);
    atomicAdd(&dW[d * 4 + 1], dw1);
    atomicAdd(&dW[d * 4 + 2], dw2);
    atomicAdd(&dW[d * 4 + 3], dw3);
    if (dbias != nullptr) {
        atomicAdd(&dbias[d], db);
    }
}

Tensor conv1d_causal_depthwise_forward_cuda(
    const Tensor& X,
    const Tensor& W,
    const Tensor* bias,
    size_t B,
    size_t T,
    size_t D
) {
    if (X.device() != Device::CUDA || W.device() != Device::CUDA) {
        throw std::runtime_error("conv1d_causal_depthwise_forward_cuda requer tensores na GPU.");
    }
    Tensor Y(X.shape(), DType::Float32, Device::CUDA);
    size_t total = B * T * D;
    if (total == 0) return Y;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((total + threads - 1) / threads, size_t(1024)));
    const float* bias_ptr = (bias && bias->is_defined()) ? bias->data<float>() : nullptr;

    conv1d_causal_depthwise_kernel<<<blocks, threads>>>(
        X.data<float>(), W.data<float>(), bias_ptr, Y.data<float>(), B, T, D, total
    );
    CUDA_SYNC_CHECK();
    return Y;
}

void conv1d_causal_depthwise_backward_cuda(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& W,
    Tensor& dX,
    Tensor& dW,
    Tensor* dbias,
    size_t B,
    size_t T,
    size_t D
) {
    if (dY.device() != Device::CUDA || X.device() != Device::CUDA || W.device() != Device::CUDA) {
        throw std::runtime_error("conv1d_causal_depthwise_backward_cuda requer tensores na GPU.");
    }
    size_t total = B * T * D;
    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((total + threads - 1) / threads, size_t(1024)));

    conv1d_causal_depthwise_backward_dx_kernel<<<blocks, threads>>>(
        dY.data<float>(), W.data<float>(), dX.data<float>(), B, T, D, total
    );
    CUDA_SYNC_CHECK();

    int weight_blocks = static_cast<int>((D + threads - 1) / threads);
    float* dbias_ptr = (dbias && dbias->is_defined()) ? dbias->data<float>() : nullptr;

    conv1d_causal_depthwise_backward_weights_kernel<<<weight_blocks, threads>>>(
        dY.data<float>(), X.data<float>(), dW.data<float>(), dbias_ptr, B, T, D
    );
    CUDA_SYNC_CHECK();
}

// ==============================================================================
//  Ativação SwiGLU (SiLU(gate) * x)
// ==============================================================================

__global__ void swiglu_forward_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ x,
    float* __restrict__ out,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float sig_g = 1.0f / (1.0f + __expf(-g));
        float silu_g = g * sig_g;
        out[i] = silu_g * x[i];
    }
}

__global__ void swiglu_backward_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ gate,
    const float* __restrict__ x,
    float* __restrict__ d_gate,
    float* __restrict__ d_x,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float dy = dY[i];
        float g = gate[i];
        float val_x = x[i];

        float sig_g = 1.0f / (1.0f + __expf(-g));
        float silu_g = g * sig_g;
        float d_silu = sig_g * (1.0f + g * (1.0f - sig_g));

        d_gate[i] = dy * val_x * d_silu;
        d_x[i] = dy * silu_g;
    }
}

Tensor swiglu_cuda(const Tensor& gate, const Tensor& x) {
    if (gate.device() != Device::CUDA || x.device() != Device::CUDA) {
        throw std::runtime_error("swiglu_cuda requer tensores na GPU.");
    }
    Tensor out(gate.shape(), DType::Float32, Device::CUDA);
    size_t n = gate.numel();
    if (n == 0) return out;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    swiglu_forward_kernel<<<blocks, threads>>>(gate.data<float>(), x.data<float>(), out.data<float>(), n);
    CUDA_SYNC_CHECK();
    return out;
}

void swiglu_backward_cuda(
    const Tensor& dY,
    const Tensor& gate,
    const Tensor& x,
    Tensor& d_gate,
    Tensor& d_x
) {
    if (dY.device() != Device::CUDA || gate.device() != Device::CUDA || x.device() != Device::CUDA) {
        throw std::runtime_error("swiglu_backward_cuda requer tensores na GPU.");
    }
    size_t n = dY.numel();
    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    swiglu_backward_kernel<<<blocks, threads>>>(
        dY.data<float>(), gate.data<float>(), x.data<float>(),
        d_gate.data<float>(), d_x.data<float>(), n
    );
    CUDA_SYNC_CHECK();
}

// ==============================================================================
//  [EXPERIMENTAL / WIP] Gated DeltaNet (Canal Escalar Experimental)
//  Nota: Módulo de pesquisa desacoplado do StackedLinearRNNLM (o modelo oficial
//  utiliza RG-LRU + Causal Depthwise Conv1D + GELU MLP).
// ==============================================================================

__global__ void gated_deltanet_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ gate,
    float* __restrict__ S_state,
    float* __restrict__ Out,
    size_t B,
    size_t T,
    size_t D
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y;
    if (b >= B || d >= D) return;

    float s = 0.0f;

    for (size_t t = 0; t < T; ++t) {
        size_t idx = (b * T + t) * D + d;

        float q = Q[idx];
        float k = K[idx];
        float v = V[idx];
        float a = 1.0f / (1.0f + __expf(-alpha[idx])); // retention gate in (0, 1)
        float b_w = 1.0f / (1.0f + __expf(-beta[idx])); // delta rate in (0, 1)
        float g = 1.0f / (1.0f + __expf(-gate[idx])); // output gate in (0, 1)

        // Delta rule memory update: S_t = a * S_{t-1} + b_w * (v - S_{t-1} * k) * k
        float retrieved = s * k;
        float delta_err = v - retrieved;
        s = a * s + b_w * delta_err * k;
        S_state[idx] = s;

        // Output projection with output gate
        Out[idx] = g * (s * q);
    }
}

__global__ void gated_deltanet_backward_kernel(
    const float* __restrict__ dOut,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ gate,
    const float* __restrict__ S_state,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    float* __restrict__ d_alpha,
    float* __restrict__ d_beta,
    float* __restrict__ d_gate,
    size_t B,
    size_t T,
    size_t D
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y;
    if (b >= B || d >= D) return;

    float ds_next = 0.0f;

    for (int64_t t = static_cast<int64_t>(T) - 1; t >= 0; --t) {
        size_t idx = (b * T + t) * D + d;

        float q = Q[idx];
        float k = K[idx];
        float v = V[idx];
        float a = 1.0f / (1.0f + __expf(-alpha[idx]));
        float b_w = 1.0f / (1.0f + __expf(-beta[idx]));
        float g = 1.0f / (1.0f + __expf(-gate[idx]));
        float s_curr = S_state[idx];
        float s_prev = (t > 0) ? S_state[(b * T + (t - 1)) * D + d] : 0.0f;

        float dout = dOut[idx];

        // Gradient for output gate and Q
        d_gate[idx] = dout * (s_curr * q) * g * (1.0f - g);
        dQ[idx] = dout * g * s_curr;

        // Gradient of state s_curr
        float ds = dout * g * q + ds_next;

        // Memory update derivatives: s_curr = a * s_prev + b_w * (v - s_prev * k) * k
        float err = v - s_prev * k;

        dV[idx] = ds * b_w * k;
        dK[idx] = ds * b_w * (v - 2.0f * s_prev * k);
        d_alpha[idx] = ds * s_prev * a * (1.0f - a);
        d_beta[idx] = ds * (err * k) * b_w * (1.0f - b_w);

        // Gradient propagating to s_prev: ds_prev = ds * (a - b_w * k * k)
        ds_next = ds * (a - b_w * k * k);
    }
}

Tensor gated_deltanet_forward_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& gate,
    Tensor& S_state
) {
    size_t B = Q.dim(0);
    size_t T = Q.dim(1);
    size_t D = Q.dim(2);

    Tensor Out(Q.shape(), DType::Float32, Device::CUDA);
    S_state = Tensor(Q.shape(), DType::Float32, Device::CUDA);

    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>((D + threads - 1) / threads), static_cast<unsigned int>(B));

    gated_deltanet_forward_kernel<<<grid, block>>>(
        Q.data<float>(), K.data<float>(), V.data<float>(),
        alpha.data<float>(), beta.data<float>(), gate.data<float>(),
        S_state.data<float>(), Out.data<float>(), B, T, D
    );
    CUDA_SYNC_CHECK();
    return Out;
}

void gated_deltanet_backward_cuda(
    const Tensor& dOut,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& gate,
    const Tensor& S_state,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    Tensor& d_alpha,
    Tensor& d_beta,
    Tensor& d_gate
) {
    size_t B = Q.dim(0);
    size_t T = Q.dim(1);
    size_t D = Q.dim(2);

    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>((D + threads - 1) / threads), static_cast<unsigned int>(B));

    gated_deltanet_backward_kernel<<<grid, block>>>(
        dOut.data<float>(), Q.data<float>(), K.data<float>(), V.data<float>(),
        alpha.data<float>(), beta.data<float>(), gate.data<float>(), S_state.data<float>(),
        dQ.data<float>(), dK.data<float>(), dV.data<float>(),
        d_alpha.data<float>(), d_beta.data<float>(), d_gate.data<float>(),
        B, T, D
    );
    CUDA_SYNC_CHECK();
}

// ==============================================================================
//  [EXPERIMENTAL / WIP] Qwen Sparse Attention (Sliding Window Local Attention)
//  Nota: Módulo experimental de canal local desacoplado do modelo principal.
// ==============================================================================

__global__ void sparse_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    size_t B,
    size_t T,
    size_t D,
    size_t W
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y;
    if (b >= B || d >= D) return;

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

    for (size_t t = 0; t < T; ++t) {
        size_t start_tau = (t >= W) ? (t - W + 1) : 0;
        size_t query_idx = (b * T + t) * D + d;
        float q = Q[query_idx];

        float sum_exp = 0.0f;
        float max_score = -1e9f;

        // 1. Encontra max score para estabilidade numérica
        for (size_t tau = start_tau; tau <= t; ++tau) {
            float k = K[(b * T + tau) * D + d];
            float score = (q * k) * inv_sqrt_d;
            if (score > max_score) max_score = score;
        }

        // 2. Calcula Softmax e acumula V
        float accum_v = 0.0f;
        for (size_t tau = start_tau; tau <= t; ++tau) {
            float k = K[(b * T + tau) * D + d];
            float v = V[(b * T + tau) * D + d];
            float score = (q * k) * inv_sqrt_d;
            float weight = __expf(score - max_score);
            sum_exp += weight;
            accum_v += weight * v;
        }

        Out[query_idx] = accum_v / (sum_exp + 1e-12f);
    }
}

__global__ void sparse_attention_backward_kernel(
    const float* __restrict__ dOut,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    size_t B,
    size_t T,
    size_t D,
    size_t W
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y;
    if (b >= B || d >= D) return;

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

    for (size_t t = 0; t < T; ++t) {
        size_t start_tau = (t >= W) ? (t - W + 1) : 0;
        size_t query_idx = (b * T + t) * D + d;
        float q = Q[query_idx];
        float dout = dOut[query_idx];

        float max_score = -1e9f;
        for (size_t tau = start_tau; tau <= t; ++tau) {
            float k = K[(b * T + tau) * D + d];
            float score = (q * k) * inv_sqrt_d;
            if (score > max_score) max_score = score;
        }

        float sum_exp = 0.0f;
        for (size_t tau = start_tau; tau <= t; ++tau) {
            float k = K[(b * T + tau) * D + d];
            float score = (q * k) * inv_sqrt_d;
            sum_exp += __expf(score - max_score);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);

        float dq_accum = 0.0f;
        for (size_t tau = start_tau; tau <= t; ++tau) {
            size_t key_idx = (b * T + tau) * D + d;
            float k = K[key_idx];
            float v = V[key_idx];
            float score = (q * k) * inv_sqrt_d;
            float p = __expf(score - max_score) * inv_sum;

            dV[key_idx] += dout * p;
            float dp = dout * v;
            float dscore = p * dp; // simplificação de softmax local

            dq_accum += dscore * k * inv_sqrt_d;
            dK[key_idx] += dscore * q * inv_sqrt_d;
        }
        dQ[query_idx] += dq_accum;
    }
}

Tensor sparse_attention_forward_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    size_t B,
    size_t T,
    size_t D,
    size_t window_size
) {
    Tensor Out(Q.shape(), DType::Float32, Device::CUDA);
    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>((D + threads - 1) / threads), static_cast<unsigned int>(B));

    sparse_attention_forward_kernel<<<grid, block>>>(
        Q.data<float>(), K.data<float>(), V.data<float>(),
        Out.data<float>(), B, T, D, window_size
    );
    CUDA_SYNC_CHECK();
    return Out;
}

void sparse_attention_backward_cuda(
    const Tensor& dOut,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    size_t B,
    size_t T,
    size_t D,
    size_t window_size
) {
    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>((D + threads - 1) / threads), static_cast<unsigned int>(B));

    sparse_attention_backward_kernel<<<grid, block>>>(
        dOut.data<float>(), Q.data<float>(), K.data<float>(), V.data<float>(),
        dQ.data<float>(), dK.data<float>(), dV.data<float>(),
        B, T, D, window_size
    );
    CUDA_SYNC_CHECK();
}

GpuDeviceInfo get_gpu_device_info(int device_id) {
    GpuDeviceInfo info;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        info.name = prop.name;
        info.major = prop.major;
        info.minor = prop.minor;
        info.sm_count = prop.multiProcessorCount;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        info.free_memory_mb = free_bytes / (1024 * 1024);
        info.total_memory_mb = total_bytes / (1024 * 1024);
    } else if (prop.totalGlobalMem > 0) {
        info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
    }
    return info;
}

void print_gpu_info(int device_id) {
    GpuDeviceInfo info = get_gpu_device_info(device_id);
    std::cout << "=================================================================\n"
              << "              DIAGNÓSTICO DE HARDWARE DA GPU ACELERADORA          \n"
              << "=================================================================\n"
              << " -> Dispositivo: " << info.name << "\n"
              << " -> Arquitetura / Compute Capability: sm_" << info.major << info.minor
              << " (" << info.major << "." << info.minor << ")\n"
              << " -> Streaming Multiprocessors (SMs): " << info.sm_count << "\n"
              << " -> VRAM Total: " << info.total_memory_mb << " MB ("
              << std::fixed << std::setprecision(1) << (static_cast<double>(info.total_memory_mb) / 1024.0) << " GB)\n"
              << " -> VRAM Livre Estimada: " << info.free_memory_mb << " MB\n"
              << "=================================================================" << std::endl;
}

} // namespace cuda
} // namespace sore
