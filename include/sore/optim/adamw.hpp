#pragma once

#include "sore/core/tensor.hpp"
#include <vector>

namespace sore {
namespace optim {

struct AdamWConfig {
    float lr{1e-3f};
    float beta1{0.9f};
    float beta2{0.999f};
    float eps{1e-8f};
    float weight_decay{1e-2f};
};

/**
 * @brief Otimizador Fused AdamW implementado em CUDA nativo.
 * 
 * Executa todas as atualizações de primeiro/segundo momentos, correção de viés
 * e decaimento de peso desacoplado (weight decay) em uma única passada de kernel
 * por parâmetro (Fused Kernel), eliminando múltiplos acessos redundantes à VRAM.
 */
class AdamW {
public:
    AdamW(std::vector<Tensor*> params, AdamWConfig config = {});

    void step();
    void zero_grad();

    [[nodiscard]] size_t current_step() const noexcept { return step_count_; }
    [[nodiscard]] const AdamWConfig& config() const noexcept { return config_; }

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor> exp_avg_;    // Primeiro momento m
    std::vector<Tensor> exp_avg_sq_; // Segundo momento v
    AdamWConfig config_;
    size_t step_count_{0};
};

/**
 * @brief Kernel lançador Fused AdamW na GPU
 */
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
);

void fused_adamw_cpu(
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
);

} // namespace optim
} // namespace sore
