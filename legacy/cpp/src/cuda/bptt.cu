#include "sore/nn/autograd.hpp"
#include <algorithm>
#include <stdexcept>

namespace sore {
namespace autograd {

__global__ void linear_rnn_backward_coalesced_kernel(
    const float* __restrict__ dH,
    const float* __restrict__ A,
    const float* __restrict__ H,
    const float* __restrict__ h0,
    float* __restrict__ dA,
    float* __restrict__ dX,
    size_t B,
    size_t T,
    size_t D
) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || d >= D) return;

    float dh_next = 0.0f;

    // Varredura reversa no tempo t = T - 1 down to 0
    for (int64_t t = static_cast<int64_t>(T) - 1; t >= 0; --t) {
        size_t idx = (b * T + t) * D + d;
        float dh_t = dH[idx] + dh_next;
        dX[idx] = dh_t;

        float h_prev = 0.0f;
        if (t > 0) {
            size_t prev_idx = (b * T + (t - 1)) * D + d;
            h_prev = H[prev_idx];
        } else if (h0 != nullptr) {
            h_prev = h0[b * D + d];
        }

        dA[idx] = dh_t * h_prev;
        float a_t = A[idx];
        dh_next = a_t * dh_t;
    }
}

__global__ void sigmoid_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ A,
    float* __restrict__ dG,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float a = A[i];
        dG[i] = dA[i] * a * (1.0f - a);
    }
}

void linear_rnn_backward_cuda(
    const Tensor& dH,
    const Tensor& A,
    const Tensor& H,
    const Tensor* h0,
    Tensor& dA,
    Tensor& dX
) {
    if (dH.device() != Device::CUDA || A.device() != Device::CUDA || H.device() != Device::CUDA) {
        throw std::runtime_error("linear_rnn_backward_cuda requer tensores na GPU.");
    }
    size_t B = dH.dim(0);
    size_t T = dH.dim(1);
    size_t D = dH.dim(2);

    dim3 block(32, 8);
    dim3 grid(
        static_cast<unsigned int>((D + block.x - 1) / block.x),
        static_cast<unsigned int>((B + block.y - 1) / block.y)
    );

    const float* h0_ptr = (h0 && h0->is_defined()) ? h0->data<float>() : nullptr;

    linear_rnn_backward_coalesced_kernel<<<grid, block>>>(
        dH.data<float>(),
        A.data<float>(),
        H.data<float>(),
        h0_ptr,
        dA.data<float>(),
        dX.data<float>(),
        B, T, D
    );
    CUDA_SYNC_CHECK();
}

Tensor sigmoid_backward_cuda(const Tensor& dA, const Tensor& A) {
    if (dA.device() != Device::CUDA || A.device() != Device::CUDA) {
        throw std::runtime_error("sigmoid_backward_cuda requer tensores na GPU.");
    }
    Tensor dG(A.shape(), A.dtype(), Device::CUDA);
    size_t n = A.numel();
    if (n == 0) return dG;

    constexpr int threads = 256;
    int blocks = static_cast<int>(std::min((n + threads - 1) / threads, size_t(1024)));

    sigmoid_backward_kernel<<<blocks, threads>>>(
        dA.data<float>(),
        A.data<float>(),
        dG.data<float>(),
        n
    );
    CUDA_SYNC_CHECK();
    return dG;
}

} // namespace autograd
} // namespace sore
