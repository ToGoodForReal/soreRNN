#pragma once

#include "sore/core/common.hpp"
#include "sore/core/storage.hpp"
#include <vector>
#include <memory>
#include <initializer_list>
#include <iostream>

namespace sore {

/**
 * @brief Abstração multidimensional de Tensores.
 * 
 * Desacopla a semântica matemática (shape, strides, rank, offset) da posse de
 * memória física contígua (StoragePtr). Suporta views, slices e reshape com custo
 * zero de alocação (zero-copy).
 */
class Tensor {
public:
    Tensor() = default;

    // Construtores principais com alocação de novo buffer
    Tensor(std::vector<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);
    Tensor(std::initializer_list<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);

    // Construtor de View: compartilha o mesmo Storage físico com strides e offset customizados
    Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset,
           StoragePtr storage, DType dtype);

    // Metadados
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] size_t dim(size_t index) const;
    [[nodiscard]] size_t numel() const noexcept { return numel_; }
    [[nodiscard]] size_t offset() const noexcept { return offset_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] Device device() const noexcept { return storage_ ? storage_->device() : Device::CPU; }
    [[nodiscard]] const StoragePtr& storage() const noexcept { return storage_; }
    [[nodiscard]] bool is_contiguous() const noexcept;
    [[nodiscard]] bool is_defined() const noexcept { return storage_ != nullptr && storage_->data() != nullptr; }

    // Acesso aos dados brutos
    [[nodiscard]] void* raw_data() noexcept;
    [[nodiscard]] const void* raw_data() const noexcept;

    // Acesso tipado ao início lógico dos dados (respeita offset_)
    template <typename T>
    [[nodiscard]] T* data() noexcept {
        if (!storage_ || !storage_->data()) return nullptr;
        if constexpr (std::is_void_v<T>) {
            return static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype_);
        } else {
            return storage_->data_as<T>() + offset_;
        }
    }

    template <typename T>
    [[nodiscard]] const T* data() const noexcept {
        if (!storage_ || !storage_->data()) return nullptr;
        if constexpr (std::is_void_v<T>) {
            return static_cast<const char*>(storage_->data()) + offset_ * dtype_size(dtype_);
        } else {
            return storage_->data_as<const T>() + offset_;
        }
    }

    // Leitura/Escrita de elemento escalar (apenas para CPU ou validações)
    template <typename T = float>
    [[nodiscard]] T& item(size_t index);

    template <typename T = float>
    [[nodiscard]] const T& item(size_t index) const;

    // Operações de Layout e View (Zero-Copy)
    [[nodiscard]] Tensor view(std::vector<size_t> new_shape) const;
    [[nodiscard]] Tensor reshape(std::vector<size_t> new_shape) const;

    // Transferência explícita de memória (H2D, D2H, D2D)
    [[nodiscard]] Tensor to(Device target_device) const;
    [[nodiscard]] Tensor cpu() const { return to(Device::CPU); }
    [[nodiscard]] Tensor cuda() const { return to(Device::CUDA); }

    // Utilitários
    void fill_(float value);
    void zero_();
    [[nodiscard]] Tensor clone() const;

    // Métodos estáticos de fábrica (Factory Methods)
    static Tensor zeros(std::vector<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor ones(std::vector<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);

    void print_info(std::ostream& os = std::cout) const;

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_{0};
    size_t numel_{0};
    DType dtype_{DType::Float32};
    StoragePtr storage_{nullptr};

    static std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape);
    static size_t compute_numel(const std::vector<size_t>& shape);
};

} // namespace sore
