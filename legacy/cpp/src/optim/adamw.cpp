#include "sore/optim/adamw.hpp"
#include <cmath>
#include <stdexcept>

namespace sore {
namespace optim {

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
) {
    size_t n = param.numel();
    float* p = param.data<float>();
    const float* g = grad.data<float>();
    float* m = exp_avg.data<float>();
    float* v = exp_avg_sq.data<float>();

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        float grad_val = g[i];
        float param_val = p[i];

        if (weight_decay != 0.0f) {
            param_val -= lr * weight_decay * param_val;
        }

        float m_val = beta1 * m[i] + (1.0f - beta1) * grad_val;
        float v_val = beta2 * v[i] + (1.0f - beta2) * (grad_val * grad_val);

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        param_val -= lr * m_hat / (std::sqrt(v_hat) + eps);

        p[i] = param_val;
        m[i] = m_val;
        v[i] = v_val;
    }
}

AdamW::AdamW(std::vector<Tensor*> params, AdamWConfig config)
    : params_(std::move(params)), config_(config) {
    for (auto* p : params_) {
        if (!p) continue;
        exp_avg_.push_back(Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq_.push_back(Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }
}

AdamW::AdamW(std::vector<Tensor*> params, std::vector<Tensor*> grads, AdamWConfig config)
    : params_(std::move(params)), grads_(std::move(grads)), config_(config) {
    for (auto* p : params_) {
        if (!p) continue;
        exp_avg_.push_back(Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq_.push_back(Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }
}

void AdamW::step(const std::vector<Tensor*>& grads) {
    if (grads.size() != params_.size()) {
        throw std::runtime_error("[AdamW::step] Contagem de gradientes (" +
                                 std::to_string(grads.size()) + ") difere dos parâmetros (" +
                                 std::to_string(params_.size()) + ").");
    }

    step_count_++;

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor* p = params_[i];
        const Tensor* g = grads[i];
        if (!p || !g || !p->is_defined() || !g->is_defined()) continue;

        if (p->device() == Device::CUDA) {
            fused_adamw_cuda(
                *p, *g, exp_avg_[i], exp_avg_sq_[i],
                config_.lr, config_.beta1, config_.beta2, config_.eps,
                config_.weight_decay, step_count_
            );
        } else {
            fused_adamw_cpu(
                *p, *g, exp_avg_[i], exp_avg_sq_[i],
                config_.lr, config_.beta1, config_.beta2, config_.eps,
                config_.weight_decay, step_count_
            );
        }
    }
}

void AdamW::step() {
    if (grads_.empty()) {
        throw std::runtime_error("[AdamW::step] Chamado sem lista de gradientes vinculada.");
    }
    step(grads_);
}

void AdamW::zero_grad(std::vector<Tensor*>& grads) {
    for (auto* g : grads) {
        if (g && g->is_defined()) {
            g->zero_();
        }
    }
}

void AdamW::zero_grad() {
    if (!grads_.empty()) {
        zero_grad(grads_);
    }
}

} // namespace optim
} // namespace sore
