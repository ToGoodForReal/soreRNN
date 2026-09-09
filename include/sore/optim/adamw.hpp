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
    AdamW(std::vector<Tensor*> params, std::vector<Tensor*> grads, AdamWConfig config = {});

    void step();
    void step(const std::vector<Tensor*>& grads);
    void zero_grad();
    void zero_grad(std::vector<Tensor*>& grads);

    [[nodiscard]] size_t current_step() const noexcept { return step_count_; }
    void set_step(size_t step) noexcept { step_count_ = step; }
    void set_lr(float lr) noexcept { config_.lr = lr; }
    [[nodiscard]] const AdamWConfig& config() const noexcept { return config_; }
    [[nodiscard]] AdamWConfig& config() noexcept { return config_; }

    [[nodiscard]] const std::vector<Tensor>& exp_avg() const noexcept { return exp_avg_; }
    [[nodiscard]] std::vector<Tensor>& exp_avg() noexcept { return exp_avg_; }
    [[nodiscard]] const std::vector<Tensor>& exp_avg_sq() const noexcept { return exp_avg_sq_; }
    [[nodiscard]] std::vector<Tensor>& exp_avg_sq() noexcept { return exp_avg_sq_; }

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
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
