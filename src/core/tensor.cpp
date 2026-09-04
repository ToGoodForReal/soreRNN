#include "sore/core/tensor.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace sore {

size_t Tensor::compute_numel(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    size_t count = 1;
    for (size_t s : shape) {
        count *= s;
    }
    return count;
}

std::vector<size_t> Tensor::compute_contiguous_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

Tensor::Tensor(std::vector<size_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)),
      strides_(compute_contiguous_strides(shape_)),
      offset_(0),
      numel_(compute_numel(shape_)),
      dtype_(dtype) {
    size_t total_bytes = numel_ * dtype_size(dtype_);
    storage_ = std::make_shared<Storage>(total_bytes, device);
}

Tensor::Tensor(std::initializer_list<size_t> shape, DType dtype, Device device)
    : Tensor(std::vector<size_t>(shape), dtype, device) {}

Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset,
               StoragePtr storage, DType dtype)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      numel_(compute_numel(shape_)),
      dtype_(dtype),
      storage_(std::move(storage)) {}

size_t Tensor::dim(size_t index) const {
    if (index >= shape_.size()) {
        std::ostringstream oss;
        oss << "Dimension out of range (expected < " << shape_.size() << ", got " << index << ")";
        throw std::out_of_range(oss.str());
    }
    return shape_[index];
}

bool Tensor::is_contiguous() const noexcept {
    return strides_ == compute_contiguous_strides(shape_);
}

void* Tensor::raw_data() noexcept {
    return storage_ ? storage_->data() : nullptr;
}

const void* Tensor::raw_data() const noexcept {
    return storage_ ? storage_->data() : nullptr;
}

template <typename T>
T& Tensor::item(size_t index) {
    if (device() != Device::CPU) {
        throw std::runtime_error("item() só pode ser chamado em Tensores na CPU. Execute .cpu() antes.");
    }
    if (index >= numel_) {
        throw std::out_of_range("Index fora dos limites do Tensor.");
    }
    return data<T>()[index];
}

template <typename T>
const T& Tensor::item(size_t index) const {
    if (device() != Device::CPU) {
        throw std::runtime_error("item() só pode ser chamado em Tensores na CPU. Execute .cpu() antes.");
    }
    if (index >= numel_) {
        throw std::out_of_range("Index fora dos limites do Tensor.");
    }
    return data<T>()[index];
}

// Instanciações explícitas de template para item<float>
template float& Tensor::item<float>(size_t);
template const float& Tensor::item<float>(size_t) const;

Tensor Tensor::view(std::vector<size_t> new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("view() requer que o tensor de origem seja contíguo na memória.");
    }
    size_t new_numel = compute_numel(new_shape);
    if (new_numel != numel_) {
        std::ostringstream oss;
        oss << "Incompatibilidade no número de elementos para view: atual " << numel_ 
            << ", solicitado " << new_numel;
        throw std::runtime_error(oss.str());
    }
    auto new_strides = compute_contiguous_strides(new_shape);
    return Tensor(std::move(new_shape), std::move(new_strides), offset_, storage_, dtype_);
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
    if (is_contiguous()) {
        return view(std::move(new_shape));
    }
    return clone().view(std::move(new_shape));
}

Tensor Tensor::to(Device target_device) const {
    if (!storage_ || !storage_->data()) {
        return Tensor();
    }
    if (device() == target_device) {
        return *this;
    }

    Tensor dest(shape_, dtype_, target_device);
    size_t bytes = numel_ * dtype_size(dtype_);

    if (device() == Device::CPU && target_device == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(dest.raw_data(), data<void>(), bytes, cudaMemcpyHostToDevice));
    } else if (device() == Device::CUDA && target_device == Device::CPU) {
        CUDA_CHECK(cudaMemcpy(dest.data<void>(), raw_data(), bytes, cudaMemcpyDeviceToHost));
    } else if (device() == Device::CUDA && target_device == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(dest.raw_data(), raw_data(), bytes, cudaMemcpyDeviceToDevice));
    } else {
        std::memcpy(dest.raw_data(), raw_data(), bytes);
    }

    return dest;
}

void Tensor::fill_(float value) {
    if (numel_ == 0) return;
    if (device() == Device::CPU) {
        std::fill_n(data<float>(), numel_, value);
    } else if (device() == Device::CUDA) {
        if (value == 0.0f) {
            CUDA_CHECK(cudaMemset(raw_data(), 0, numel_ * sizeof(float)));
        } else {
            std::vector<float> host_buffer(numel_, value);
            CUDA_CHECK(cudaMemcpy(raw_data(), host_buffer.data(), 
                                  numel_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

void Tensor::zero_() {
    fill_(0.0f);
}

Tensor Tensor::clone() const {
    Tensor copy(shape_, dtype_, device());
    size_t bytes = numel_ * dtype_size(dtype_);
    if (device() == Device::CPU) {
        std::memcpy(copy.data<void>(), data<void>(), bytes);
    } else if (device() == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(copy.raw_data(), raw_data(), bytes, cudaMemcpyDeviceToDevice));
    }
    return copy;
}

Tensor Tensor::zeros(std::vector<size_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    t.zero_();
    return t;
}

Tensor Tensor::ones(std::vector<size_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    t.fill_(1.0f);
    return t;
}

void Tensor::print_info(std::ostream& os) const {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    os << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        os << strides_[i] << (i + 1 < strides_.size() ? ", " : "");
    }
    os << "], numel=" << numel_
       << ", dtype=" << to_string(dtype_)
       << ", device=" << to_string(device())
       << ", contiguous=" << (is_contiguous() ? "true" : "false")
       << ", storage_use_count=" << (storage_ ? storage_.use_count() : 0)
       << ")";
}

} // namespace sore
