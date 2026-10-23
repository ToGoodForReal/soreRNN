#pragma once

#include "sore/core/tensor.hpp"

namespace sore {
namespace autograd {

/**
 * @brief Backpropagation Through Time (BPTT) da recorrência linear na CPU:
 *        dh_t = dH_t + a_{t+1} * dh_{t+1}
 *        da_t = dh_t * h_{t-1}
 *        dx_t = dh_t
 * 
 * @param dH Gradiente vindo da camada superior [B, T, D]
 * @param A Tensor de decay gates do forward [B, T, D]
 * @param H Tensor de estados ocultos do forward [B, T, D]
 * @param h0 Estado inicial do forward [B, D] (opcional)
 * @param dA Gradiente de saída para os gates A [B, T, D]
 * @param dX Gradiente de saída para os inputs X [B, T, D]
 */
void linear_rnn_backward_cpu(
    const Tensor& dH,
    const Tensor& A,
    const Tensor& H,
    const Tensor* h0,
    Tensor& dA,
    Tensor& dX
);

/**
 * @brief Backpropagation Through Time (BPTT) da recorrência linear na GPU (CUDA)
 */
void linear_rnn_backward_cuda(
    const Tensor& dH,
    const Tensor& A,
    const Tensor& H,
    const Tensor* h0,
    Tensor& dA,
    Tensor& dX
);

/**
 * @brief Gradiente da função de ativação Sigmóide: dg = da * a * (1 - a)
 */
Tensor sigmoid_backward_cpu(const Tensor& dA, const Tensor& A);
Tensor sigmoid_backward_cuda(const Tensor& dA, const Tensor& A);

} // namespace autograd
} // namespace sore
