#pragma once

#include "sore/core/common.hpp"
#include <memory>
#include <cstdlib>

namespace sore {

/**
 * @brief Gerenciador de memória física de baixo nível (RAII).
 * 
 * Responsável estrito pela alocação e desalocação de buffers contíguos
 * de bytes em Host (CPU, com alinhamento de 64 bytes para AVX-512) ou
 * Device (GPU via cudaMalloc).
 */
class Storage {
public:
    // Alinhamento padrão para CPU (64 bytes = Cache Line x86_64 / AVX-512)
    static constexpr size_t CPU_ALIGNMENT = 64;

    Storage(size_t size_bytes, Device device);
    ~Storage();

    // Proibir cópia (garantia de ownership único por instância de Storage)
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Permitir movimentação (Move Semantics)
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    // Acessores de dados
    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }

    template <typename T>
    [[nodiscard]] T* data_as() noexcept {
        return static_cast<T*>(data_);
    }

    template <typename T>
    [[nodiscard]] const T* data_as() const noexcept {
        return static_cast<const T*>(data_);
    }

    [[nodiscard]] size_t size_bytes() const noexcept { return size_bytes_; }
    [[nodiscard]] Device device() const noexcept { return device_; }

private:
    void* data_{nullptr};
    size_t size_bytes_{0};
    Device device_{Device::CPU};

    void allocate();
    void deallocate() noexcept;
};

// Alias de ponteiro compartilhado para uso pelo Tensor
using StoragePtr = std::shared_ptr<Storage>;

} // namespace sore
