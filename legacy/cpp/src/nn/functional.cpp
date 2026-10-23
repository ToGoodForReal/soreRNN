#include "sore/nn/functional.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace sore {
namespace functional {

Tensor matmul2d_cpu(const Tensor& A, const Tensor& B) {
    if (A.device() != Device::CPU || B.device() != Device::CPU) {
        throw std::runtime_error("matmul2d_cpu requer tensores no dispositivo CPU.");
    }
    if (A.rank() != 2 || B.rank() != 2) {
        throw std::runtime_error("matmul2d_cpu requer tensores de rank 2 (matrizes).");
    }
    size_t M = A.dim(0);
    size_t K = A.dim(1);
    size_t K2 = B.dim(0);
    size_t N = B.dim(1);

    if (K != K2) {
        std::ostringstream oss;
        oss << "Dimensões internas incompatíveis para matmul: A=[" 
            << M << ", " << K << "] e B=[" << K2 << ", " << N << "]";
        throw std::runtime_error(oss.str());
    }

    Tensor C = Tensor::zeros({M, N}, DType::Float32, Device::CPU);
    const float* a_ptr = A.data<float>();
    const float* b_ptr = B.data<float>();
    float* c_ptr = C.data<float>();

    // Otimização HPC clássica de CPU: Loop Order i-k-j para acesso contíguo à cache line
    // No loop interno (j), c_ptr[i * N + j] e b_ptr[k * N + j] avançam com stride 1,
    // permitindo vetorização automática AVX-512/AVX2 pelo compilador GCC (-O3 -march=native).
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a_ptr[i * K + k];
            const float* b_row = &b_ptr[k * N];
            float* c_row = &c_ptr[i * N];
            #pragma GCC ivdep
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }

    return C;
}

Tensor linear_cpu(const Tensor& X, const Tensor& W, const Tensor* bias) {
    if (X.device() != Device::CPU || W.device() != Device::CPU) {
        throw std::runtime_error("linear_cpu requer tensores na CPU.");
    }
    if (W.rank() != 2) {
        throw std::runtime_error("W deve ser 2D [D_out, D_in].");
    }

    size_t D_out = W.dim(0);
    size_t D_in = W.dim(1);

    if (X.shape().back() != D_in) {
        throw std::runtime_error("A última dimensão de X deve coincidir com D_in de W.");
    }

    // Calcula o número de vetores M = B * T
    size_t M = X.numel() / D_in;
    std::vector<size_t> out_shape = X.shape();
    out_shape.back() = D_out;

    Tensor Y({M, D_out}, DType::Float32, Device::CPU);
    const float* x_ptr = X.data<float>();
    const float* w_ptr = W.data<float>();
    const float* b_ptr = (bias && bias->is_defined()) ? bias->data<float>() : nullptr;
    float* y_ptr = Y.data<float>();

    // Y = X @ W^T + bias
    // X: [M, D_in], W: [D_out, D_in] -> W[j, :] é contíguo na memória!
    // y[m, j] = bias[j] + sum_k (x[m, k] * w[j, k])
    for (size_t m = 0; m < M; ++m) {
        const float* x_row = &x_ptr[m * D_in];
        float* y_row = &y_ptr[m * D_out];

        for (size_t j = 0; j < D_out; ++j) {
            float sum = b_ptr ? b_ptr[j] : 0.0f;
            const float* w_row = &w_ptr[j * D_in];
            #pragma GCC ivdep
            for (size_t k = 0; k < D_in; ++k) {
                sum += x_row[k] * w_row[k];
            }
            y_row[j] = sum;
        }
    }

    return Y.view(out_shape);
}

Tensor sigmoid_cpu(const Tensor& x) {
    if (x.device() != Device::CPU) {
        throw std::runtime_error("sigmoid_cpu requer tensor na CPU.");
    }
    Tensor out(x.shape(), x.dtype(), Device::CPU);
    const float* in_ptr = x.data<float>();
    float* out_ptr = out.data<float>();
    size_t n = x.numel();

    for (size_t i = 0; i < n; ++i) {
        float val = in_ptr[i];
        if (val >= 0.0f) {
            out_ptr[i] = 1.0f / (1.0f + std::exp(-val));
        } else {
            float exp_val = std::exp(val);
            out_ptr[i] = exp_val / (1.0f + exp_val);
        }
    }
    return out;
}

Tensor add_cpu(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("add_cpu requer mesmo número de elementos.");
    }
    Tensor out(a.shape(), a.dtype(), Device::CPU);
    const float* a_ptr = a.data<float>();
    const float* b_ptr = b.data<float>();
    float* out_ptr = out.data<float>();
    size_t n = a.numel();

    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    return out;
}

Tensor mul_cpu(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("mul_cpu requer mesmo número de elementos.");
    }
    Tensor out(a.shape(), a.dtype(), Device::CPU);
    const float* a_ptr = a.data<float>();
    const float* b_ptr = b.data<float>();
    float* out_ptr = out.data<float>();
    size_t n = a.numel();

    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return out;
}

Tensor linear_rnn_forward_cpu(const Tensor& A, const Tensor& X, const Tensor* h0) {
    if (A.device() != Device::CPU || X.device() != Device::CPU) {
        throw std::runtime_error("linear_rnn_forward_cpu requer tensores na CPU.");
    }
    if (A.rank() != 3 || X.rank() != 3) {
        throw std::runtime_error("A e X devem ter rank 3 [Batch, SeqLen, HiddenDim].");
    }
    if (A.shape() != X.shape()) {
        throw std::runtime_error("A e X devem ter exatamente as mesmas dimensões [B, T, D].");
    }

    size_t B = A.dim(0);
    size_t T = A.dim(1);
    size_t D = A.dim(2);

    Tensor H(A.shape(), DType::Float32, Device::CPU);
    const float* a_ptr = A.data<float>();
    const float* x_ptr = X.data<float>();
    float* h_out_ptr = H.data<float>();
    const float* h0_ptr = (h0 && h0->is_defined()) ? h0->data<float>() : nullptr;

    // Buffer de estado da batch atual [D]
    std::vector<float> current_h(D, 0.0f);

    for (size_t b = 0; b < B; ++b) {
        // Inicializa estado h_{-1}
        if (h0_ptr) {
            std::copy_n(&h0_ptr[b * D], D, current_h.data());
        } else {
            std::fill_n(current_h.data(), D, 0.0f);
        }

        // Recorrência no tempo t = 0 .. T-1
        for (size_t t = 0; t < T; ++t) {
            size_t step_offset = (b * T + t) * D;
            const float* a_t = &a_ptr[step_offset];
            const float* x_t = &x_ptr[step_offset];
            float* h_t = &h_out_ptr[step_offset];

            // h_t = a_t * h_{t-1} + x_t
            #pragma GCC ivdep
            for (size_t d = 0; d < D; ++d) {
                current_h[d] = a_t[d] * current_h[d] + x_t[d];
                h_t[d] = current_h[d];
            }
        }
    }

    return H;
}

} // namespace functional
} // namespace sore
