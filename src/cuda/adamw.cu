#include "sore/optim/adamw.hpp"
#include <algorithm>
#include <cmath>

namespace sore {
namespace optim {

__global__ void fused_adamw_kernel(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float grad = g[i];
        float param = p[i];
        float m_val = m[i];
        float v_val = v[i];

        // Decaimento de pesos desacoplado (Decoupled Weight Decay do AdamW)
        if (weight_decay != 0.0f) {
            param -= lr * weight_decay * param;
        }

        // Atualização de momentos com Fused Multiply-Add
        m_val = beta1 * m_val + (1.0f - beta1) * grad;
        v_val = beta2 * v_val + (1.0f - beta2) * (grad * grad);

        // Correção de viés
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        // Atualização final do parâmetro
        param -= lr * m_hat / (sqrtf(v_hat) + eps);

        // Escrita coalescida de volta na VRAM
        p[i] = param;
        m[i] = m_val;
        v[i] = v_val;
    }
}

void fused_adamw_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    size_t step
) {
    size_t n = param.numel();
    if (n == 0) return;

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    fused_adamw_kernel<<<blocks, threads>>>(
        param.data<float>(),
        grad.data<float>(),
        exp_avg.data<float>(),
        exp_avg_sq.data<float>(),
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n
    );
    CUDA_SYNC_CHECK();
}

} // namespace optim
} // namespace sore
