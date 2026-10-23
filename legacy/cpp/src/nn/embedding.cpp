#include "sore/nn/embedding.hpp"
#include <random>
#include <cmath>
#include <cstring>

namespace sore {
namespace nn {

Embedding::Embedding(size_t vocab_size, size_t emb_dim, Device device)
    : vocab_size_(vocab_size), emb_dim_(emb_dim), device_(device),
      weight_({vocab_size, emb_dim}, DType::Float32, Device::CPU) {
    // Inicialização normal padrão N(0, 1 / sqrt(emb_dim))
    std::mt19937 rng(1337);
    float stddev = 1.0f / std::sqrt(static_cast<float>(emb_dim));
    std::normal_distribution<float> dist(0.0f, stddev);

    float* w = weight_.data<float>();
    for (size_t i = 0; i < weight_.numel(); ++i) {
        w[i] = dist(rng);
    }

    if (device == Device::CUDA) {
        weight_ = weight_.cuda();
    }
}

Tensor Embedding::forward(const std::vector<uint16_t>& tokens, size_t B, size_t T) {
    size_t N = B * T;
    Tensor out_cpu({B, T, emb_dim_}, DType::Float32, Device::CPU);
    Tensor w_cpu = (device_ == Device::CUDA) ? weight_.cpu() : weight_;

    const float* w_ptr = w_cpu.data<float>();
    float* out_ptr = out_cpu.data<float>();

    for (size_t i = 0; i < N; ++i) {
        uint16_t tok = tokens[i];
        if (tok >= vocab_size_) {
            tok = 0; // clamp de segurança
        }
        std::memcpy(&out_ptr[i * emb_dim_], &w_ptr[tok * emb_dim_], emb_dim_ * sizeof(float));
    }

    if (device_ == Device::CUDA) {
        return out_cpu.cuda();
    }
    return out_cpu;
}

void Embedding::backward(const Tensor& d_out, const std::vector<uint16_t>& tokens, Tensor& d_weight) {
    Tensor d_out_cpu = (d_out.device() == Device::CUDA) ? d_out.cpu() : d_out;
    Tensor d_weight_cpu = (d_weight.device() == Device::CUDA) ? d_weight.cpu() : d_weight;

    const float* d_out_ptr = d_out_cpu.data<float>();
    float* dw_ptr = d_weight_cpu.data<float>();
    size_t N = tokens.size();

    for (size_t i = 0; i < N; ++i) {
        uint16_t tok = tokens[i];
        if (tok >= vocab_size_) tok = 0;

        const float* src = &d_out_ptr[i * emb_dim_];
        float* dst = &dw_ptr[tok * emb_dim_];

        #pragma GCC ivdep
        for (size_t d = 0; d < emb_dim_; ++d) {
            dst[d] += src[d];
        }
    }

    if (d_weight.device() == Device::CUDA) {
        d_weight = d_weight_cpu.cuda();
    }
}

} // namespace nn
} // namespace sore
