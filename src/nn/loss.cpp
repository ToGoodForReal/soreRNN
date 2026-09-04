#include "sore/nn/loss.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace sore {
namespace nn {

float cross_entropy_loss_and_grad(
    const Tensor& logits,
    const std::vector<uint16_t>& targets,
    Tensor& d_logits
) {
    size_t N = targets.size();
    size_t V = logits.shape().back();

    if (logits.numel() != N * V) {
        throw std::runtime_error("Dimensões incompatíveis entre logits e targets.");
    }

    Tensor logits_cpu = (logits.device() == Device::CUDA) ? logits.cpu() : logits;
    Tensor d_logits_cpu({N, V}, DType::Float32, Device::CPU);

    const float* log_ptr = logits_cpu.data<float>();
    float* grad_ptr = d_logits_cpu.data<float>();

    double total_loss = 0.0;
    float inv_n = 1.0f / static_cast<float>(N);

    for (size_t i = 0; i < N; ++i) {
        const float* row = &log_ptr[i * V];
        float* d_row = &grad_ptr[i * V];

        // 1. Estabilidade numérica: subtrai o máximo antes do exp
        float max_val = *std::max_element(row, row + V);

        // 2. Soma das exponenciais (Denominador do Softmax)
        float sum_exp = 0.0f;
        for (size_t j = 0; j < V; ++j) {
            float e = std::exp(row[j] - max_val);
            d_row[j] = e; // armazenamos temporariamente para reusar no gradiente
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        uint16_t y = targets[i];
        if (y >= V) y = 0;

        // Probabilidade da classe correta: P(y) = e / sum_exp
        float p_target = d_row[y] * inv_sum;
        total_loss += -std::log(std::max(p_target, 1e-12f));

        // 3. Gradiente analítico: dLogits = (Softmax(Z) - OneHot(y)) / N
        for (size_t j = 0; j < V; ++j) {
            float p = d_row[j] * inv_sum;
            d_row[j] = p * inv_n;
        }
        d_row[y] -= inv_n;
    }

    if (d_logits.device() == Device::CUDA) {
        d_logits = d_logits_cpu.cuda();
    } else {
        d_logits = std::move(d_logits_cpu);
    }

    return static_cast<float>(total_loss * inv_n);
}

} // namespace nn
} // namespace sore
