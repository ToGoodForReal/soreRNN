#pragma once

#include "sore/core/tensor.hpp"

namespace sore {
namespace cuda {

/**
 * @brief Forward pass da RNN Linear na GPU com Parallel Associative Scan.
 * 
 * Resolve a recorrência linear h_t = a_t \odot h_{t-1} + x_t em paralelo.
 * 
 * @param A Tensor de decay gates [B, T, D] em VRAM, valores em (0, 1)
 * @param X Tensor de entrada projetada [B, T, D] em VRAM
 * @param h0 Tensor de estado inicial [B, D] (opcional)
 * @return Tensor H [B, T, D] com todos os estados ocultos no tempo
 */
Tensor linear_rnn_forward_cuda(const Tensor& A, const Tensor& X, const Tensor* h0 = nullptr);

/**
 * @brief Variante especializada com Associative Scan em Shared Memory (Kogge-Stone)
 *        para validação da formulação associativa (p_i \bullet p_j).
 */
Tensor associative_scan_forward_cuda(const Tensor& A, const Tensor& X, const Tensor* h0 = nullptr);

} // namespace cuda
} // namespace sore
