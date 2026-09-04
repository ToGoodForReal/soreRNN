#pragma once

#include "sore/core/tensor.hpp"

namespace sore {
namespace functional {

/**
 * @brief Multiplicação de matrizes 2D na CPU: C = A @ B
 * @param A Tensor [M, K]
 * @param B Tensor [K, N]
 * @return Tensor C [M, N]
 */
Tensor matmul2d_cpu(const Tensor& A, const Tensor& B);

/**
 * @brief Projeção linear com batch/sequência: Y = X @ W^T + bias
 * @param X Tensor [B, T, D_in] ou [N, D_in]
 * @param W Tensor [D_out, D_in] (pesos no padrão PyTorch/cuBLAS)
 * @param bias Tensor [D_out] opcional
 * @return Tensor Y [B, T, D_out] ou [N, D_out]
 */
Tensor linear_cpu(const Tensor& X, const Tensor& W, const Tensor* bias = nullptr);

/**
 * @brief Sigmóide elemento a elemento: sigma(x) = 1 / (1 + exp(-x))
 */
Tensor sigmoid_cpu(const Tensor& x);

/**
 * @brief Soma elemento a elemento com broadcasting de viés 1D se necessário
 */
Tensor add_cpu(const Tensor& a, const Tensor& b);

/**
 * @brief Multiplicação elemento a elemento (Hadamard): C = A * B
 */
Tensor mul_cpu(const Tensor& a, const Tensor& b);

/**
 * @brief Executa o loop sequencial da RNN Linear (RG-LRU baseline CPU):
 *        h_t = a_t * h_{t-1} + x_t
 * @param A Tensor de decay gates [B, T, D], valores em (0, 1)
 * @param X Tensor de inputs projetados [B, T, D]
 * @param h0 Tensor de estado inicial [B, D] (opcional, defaults to 0)
 * @return Tensor de estados ocultos H [B, T, D]
 */
Tensor linear_rnn_forward_cpu(const Tensor& A, const Tensor& X, const Tensor* h0 = nullptr);

} // namespace functional
} // namespace sore
