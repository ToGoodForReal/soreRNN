#pragma once

#include "sore/core/tensor.hpp"
#include "sore/cuda/cublas_handle.hpp"
#include <string>

namespace sore {
namespace cuda {

/**
 * @brief Multiplicação de matrizes 2D na GPU via cuBLAS: C = A @ B
 * @param A Tensor [M, K] em VRAM
 * @param B Tensor [K, N] em VRAM
 * @return Tensor C [M, N] em VRAM
 */
Tensor matmul2d_cuda(const Tensor& A, const Tensor& B);

/**
 * @brief Projeção linear na GPU com cuBLAS e kernel de viés: Y = X @ W^T + bias
 * @param X Tensor [B, T, D_in] ou [M, D_in] em VRAM
 * @param W Tensor [D_out, D_in] em VRAM
 * @param bias Tensor [D_out] opcional em VRAM
 * @return Tensor Y [B, T, D_out] ou [M, D_out] em VRAM
 */
Tensor linear_cuda(const Tensor& X, const Tensor& W, const Tensor* bias = nullptr);

/**
 * @brief Sigmóide elemento a elemento acelerado por hardware GPU (SFU __expf)
 */
Tensor sigmoid_cuda(const Tensor& x);

/**
 * @brief Adição de viés in-place com broadcast: Y[m, j] += bias[j]
 */
void add_bias_cuda_(Tensor& Y, const Tensor& bias);

/**
 * @brief Retropropagação de projeção linear na GPU:
 *        dX = dY @ W,  dW += dY^T @ X,  dbias += sum(dY)
 */
void linear_backward_cuda(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& W,
    Tensor& dX,
    Tensor& dW,
    Tensor* dbias = nullptr
);

/**
 * @brief Embedding forward acelerado por GPU (Look-up Table direto em VRAM)
 */
Tensor embedding_forward_cuda(
    const std::vector<uint16_t>& tokens,
    const Tensor& weight,
    size_t B,
    size_t T
);

/**
 * @brief Embedding backward acelerado por GPU com atomicAdd
 */
void embedding_backward_cuda(
    const Tensor& d_out,
    const std::vector<uint16_t>& tokens,
    Tensor& d_weight
);

/**
 * @brief Cross-Entropy Loss e Gradientes analíticos diretamente na GPU
 */
float cross_entropy_loss_and_grad_cuda(
    const Tensor& logits,
    const std::vector<uint16_t>& targets,
    Tensor& d_logits
);

/**
 * @brief Conexão residual in-place na GPU: x += residual
 */
void add_residual_cuda(Tensor& x, const Tensor& residual);

/**
 * @brief RMSNorm forward na GPU: y = (x / rms(x)) * gamma
 */
Tensor rmsnorm_forward_cuda(const Tensor& x, const Tensor& gamma, Tensor& rstd, float eps = 1e-5f);

/**
 * @brief RMSNorm backward na GPU
 */
void rmsnorm_backward_cuda(
    const Tensor& dout,
    const Tensor& x,
    const Tensor& gamma,
    const Tensor& rstd,
    Tensor& dx,
    Tensor& dgamma
);

/**
 * @brief Gradient Clipping por norma L2 global na GPU
 */
float clip_grad_norm_cuda(std::vector<Tensor*>& grads, float max_norm = 1.0f);

/**
 * @brief Ativação GELU elemento a elemento na GPU:
 *        y = 0.5 * x * (1.0 + erf(x / sqrt(2)))
 */
Tensor gelu_cuda(const Tensor& x);

/**
 * @brief Gradiente da ativação GELU na GPU
 */
Tensor gelu_backward_cuda(const Tensor& dout, const Tensor& x);

/**
 * @brief Causal Depthwise Conv1D forward na GPU (Kernel size K=4)
 * @param X Tensor [B, T, D] em VRAM
 * @param W Tensor [D, K] em VRAM (K=4)
 * @param bias Tensor [D] opcional em VRAM
 * @return Tensor Y [B, T, D] em VRAM
 */
Tensor conv1d_causal_depthwise_forward_cuda(
    const Tensor& X,
    const Tensor& W,
    const Tensor* bias,
    size_t B,
    size_t T,
    size_t D
);

/**
 * @brief Causal Depthwise Conv1D backward na GPU:
 *        Calcula dX, dW e dbias
 */
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
);

/**
 * @brief Ativação SwiGLU: Y = SiLU(gate) * x
 */
Tensor swiglu_cuda(const Tensor& gate, const Tensor& x);

/**
 * @brief Backward da ativação SwiGLU: calcula d_gate e d_x
 */
void swiglu_backward_cuda(
    const Tensor& dY,
    const Tensor& gate,
    const Tensor& x,
    Tensor& d_gate,
    Tensor& d_x
);

/**
 * @brief Gated DeltaNet Forward (Delta Rule com Gating por canal)
 *        S_t = alpha_t * S_{t-1} + beta_t * (V_t - S_{t-1} * K_t) * K_t
 *        Out_t = sigmoid(G_t) * (S_t * Q_t)
 */
Tensor gated_deltanet_forward_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& alpha,
    const Tensor& beta,
    const Tensor& gate,
    Tensor& S_state
);

/**
 * @brief Gated DeltaNet Backward
 */
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
);

/**
 * @brief Qwen Sparse Attention Forward (Sliding Window Causal Attention, W=256)
 */
Tensor sparse_attention_forward_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    size_t B,
    size_t T,
    size_t D,
    size_t window_size = 256
);

/**
 * @brief Qwen Sparse Attention Backward
 */
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
    size_t window_size = 256
);

/**
 * @brief Estrutura com metadados do hardware da GPU detectada.
 */
struct GpuDeviceInfo {
    std::string name = "Unknown GPU";
    int major = 0;
    int minor = 0;
    size_t total_memory_mb = 0;
    size_t free_memory_mb = 0;
    int sm_count = 0;
};

/**
 * @brief Obtém informações detalhadas da GPU ativa via CUDA Runtime.
 */
GpuDeviceInfo get_gpu_device_info(int device_id = 0);

/**
 * @brief Imprime diagnóstico completo de hardware da GPU no console.
 */
void print_gpu_info(int device_id = 0);

/**
 * @brief Escala in-place um tensor na GPU por um escalar float: t = t * scale
 */
void scale_tensor_cuda(Tensor& t, float scale);

} // namespace cuda
} // namespace sore
