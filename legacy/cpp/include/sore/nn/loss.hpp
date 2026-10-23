#pragma once

#include "sore/core/tensor.hpp"
#include <vector>
#include <cstdint>

namespace sore {
namespace nn {

/**
 * @brief Computa a Cross-Entropy Loss com Softmax numericamente estável e o gradiente analítico:
 *        dLogits = (Softmax(Z) - OneHot(y)) / N
 * 
 * @param logits Tensor [N, VocabSize] (onde N = B * T)
 * @param targets Vetor de índices de tamanho N
 * @param d_logits Tensor de saída [N, VocabSize] com os gradientes
 * @return float Valor da Loss média
 */
float cross_entropy_loss_and_grad(
    const Tensor& logits,
    const std::vector<uint16_t>& targets,
    Tensor& d_logits
);

} // namespace nn
} // namespace sore
