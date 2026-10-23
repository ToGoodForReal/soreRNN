#pragma once

#include "sore/core/tensor.hpp"
#include <vector>
#include <cstdint>

namespace sore {
namespace nn {

/**
 * @brief Camada de Embedding (Look-up Table discreto -> vetores contínuos).
 */
class Embedding {
public:
    Embedding(size_t vocab_size, size_t emb_dim, Device device = Device::CPU);

    [[nodiscard]] Tensor forward(const std::vector<uint16_t>& tokens, size_t B, size_t T);

    void backward(const Tensor& d_out, const std::vector<uint16_t>& tokens, Tensor& d_weight);

    Tensor& weight() noexcept { return weight_; }
    const Tensor& weight() const noexcept { return weight_; }

    [[nodiscard]] size_t vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] size_t emb_dim() const noexcept { return emb_dim_; }
    [[nodiscard]] Device device() const noexcept { return device_; }

private:
    size_t vocab_size_;
    size_t emb_dim_;
    Device device_;
    Tensor weight_;
};

} // namespace nn
} // namespace sore
