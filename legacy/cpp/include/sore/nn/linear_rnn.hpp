#pragma once

#include "sore/core/tensor.hpp"
#include "sore/nn/functional.hpp"
#include <vector>
#include <random>

namespace sore {
namespace nn {

/**
 * @brief Camada de Recorrência Linear Real-Gated (RG-LRU / estilo Griffin).
 * 
 * Equações de Forward Pass:
 * 1. Portão de Esquecimento / Decaimento:
 *    a_t = \sigma(W_gate \cdot x_t + b_gate)   \in (0, 1)
 * 
 * 2. Projeção de Entrada:
 *    u_t = W_in \cdot x_t + b_in
 * 
 * 3. Recorrência Linear Associativa:
 *    h_t = a_t \odot h_{t-1} + u_t
 * 
 * 4. Projeção de Saída:
 *    y_t = W_out \cdot h_t + b_out
 */
class LinearRNN {
public:
    LinearRNN(size_t in_dim, size_t hidden_dim, size_t out_dim, Device device = Device::CPU);

    [[nodiscard]] Tensor forward(const Tensor& x);

    // Getters dos parâmetros para inspeção e futuros gradientes
    Tensor& w_gate() noexcept { return w_gate_; }
    Tensor& b_gate() noexcept { return b_gate_; }
    Tensor& w_in() noexcept { return w_in_; }
    Tensor& b_in() noexcept { return b_in_; }
    Tensor& w_out() noexcept { return w_out_; }
    Tensor& b_out() noexcept { return b_out_; }

    const Tensor& w_gate() const noexcept { return w_gate_; }
    const Tensor& b_gate() const noexcept { return b_gate_; }
    const Tensor& w_in() const noexcept { return w_in_; }
    const Tensor& b_in() const noexcept { return b_in_; }
    const Tensor& w_out() const noexcept { return w_out_; }
    const Tensor& b_out() const noexcept { return b_out_; }

    [[nodiscard]] size_t in_dim() const noexcept { return in_dim_; }
    [[nodiscard]] size_t hidden_dim() const noexcept { return hidden_dim_; }
    [[nodiscard]] size_t out_dim() const noexcept { return out_dim_; }
    [[nodiscard]] Device device() const noexcept { return device_; }

    // Inicialização uniforme / Xavier para testes numéricos
    void init_weights(uint64_t seed = 42);

private:
    size_t in_dim_;
    size_t hidden_dim_;
    size_t out_dim_;
    Device device_;

    Tensor w_gate_;
    Tensor b_gate_;
    Tensor w_in_;
    Tensor b_in_;
    Tensor w_out_;
    Tensor b_out_;
};

} // namespace nn
} // namespace sore
