#pragma once

#include "sore/core/tensor.hpp"
#include "sore/cuda/cublas_handle.hpp"

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

} // namespace cuda
} // namespace sore
