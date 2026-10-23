#include "sore/nn/autograd.hpp"
#include <algorithm>
#include <stdexcept>

namespace sore {
namespace autograd {

void linear_rnn_backward_cpu(
    const Tensor& dH,
    const Tensor& A,
    const Tensor& H,
    const Tensor* h0,
    Tensor& dA,
    Tensor& dX
) {
    if (dH.device() != Device::CPU || A.device() != Device::CPU || H.device() != Device::CPU) {
        throw std::runtime_error("linear_rnn_backward_cpu requer tensores na CPU.");
    }
    size_t B = dH.dim(0);
    size_t T = dH.dim(1);
    size_t D = dH.dim(2);

    const float* dh_ptr = dH.data<float>();
    const float* a_ptr = A.data<float>();
    const float* h_ptr = H.data<float>();
    const float* h0_ptr = (h0 && h0->is_defined()) ? h0->data<float>() : nullptr;

    float* da_ptr = dA.data<float>();
    float* dx_ptr = dX.data<float>();

    std::vector<float> dh_next(D, 0.0f);

    for (size_t b = 0; b < B; ++b) {
        std::fill(dh_next.begin(), dh_next.end(), 0.0f);

        // Varredura reversa no tempo: t = T - 1 down to 0
        for (int64_t t = static_cast<int64_t>(T) - 1; t >= 0; --t) {
            size_t curr_offset = (b * T + t) * D;

            #pragma GCC ivdep
            for (size_t d = 0; d < D; ++d) {
                size_t idx = curr_offset + d;
                float dh_t = dh_ptr[idx] + dh_next[d];
                dx_ptr[idx] = dh_t;

                float h_prev = 0.0f;
                if (t > 0) {
                    size_t prev_offset = (b * T + (t - 1)) * D + d;
                    h_prev = h_ptr[prev_offset];
                } else if (h0_ptr) {
                    h_prev = h0_ptr[b * D + d];
                }

                da_ptr[idx] = dh_t * h_prev;

                // Propagação do gradiente para o passo anterior t-1: a_t * dh_t
                float a_t = a_ptr[idx];
                dh_next[d] = a_t * dh_t;
            }
        }
    }
}

Tensor sigmoid_backward_cpu(const Tensor& dA, const Tensor& A) {
    if (dA.numel() != A.numel()) {
        throw std::runtime_error("sigmoid_backward_cpu: dimensões incompatíveis.");
    }
    Tensor dG(A.shape(), A.dtype(), Device::CPU);
    const float* da = dA.data<float>();
    const float* a = A.data<float>();
    float* dg = dG.data<float>();
    size_t n = A.numel();

    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        float a_val = a[i];
        dg[i] = da[i] * a_val * (1.0f - a_val);
    }
    return dG;
}

} // namespace autograd
} // namespace sore
