#pragma once

#include "sore/core/common.hpp"
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <map>

namespace sore {
namespace cuda {

struct AllocatorStats {
    size_t allocated_bytes{0};
    size_t cached_bytes{0};
    size_t reserved_bytes{0};
    size_t peak_allocated_bytes{0};
    size_t num_alloc_calls{0};
    size_t num_free_calls{0};
    size_t num_cuda_malloc_calls{0};
    size_t num_cuda_free_calls{0};
};

/**
 * @brief Alocador de memória GPU com cache (Memory Pool) de alta performance.
 * 
 * Elimina o overhead síncrono de chamadas recorrentes a cudaMalloc e cudaFree,
 * agrupando blocos de VRAM por classes de tamanho e reutilizando-os entre
 * passos de forward/backward.
 */
class CUDACachingAllocator {
public:
    static CUDACachingAllocator& instance();

    // Aloca buffer de VRAM com tamanho mínimo especificado
    void* allocate(size_t size_bytes);

    // Devolve buffer ao cache do pool
    void deallocate(void* ptr) noexcept;

    // Libera todos os blocos do cache para o driver CUDA (cudaFree)
    void empty_cache() noexcept;

    // Estatísticas de memória
    AllocatorStats stats() const noexcept;
    void reset_peak_stats() noexcept;

    // Alinhamento base de blocos (256 bytes para memória coalescida CUDA)
    static constexpr size_t ALIGNMENT = 256;

private:
    CUDACachingAllocator() = default;
    ~CUDACachingAllocator();

    CUDACachingAllocator(const CUDACachingAllocator&) = delete;
    CUDACachingAllocator& operator=(const CUDACachingAllocator&) = delete;

    static size_t round_size(size_t size);
    void empty_cache_internal() noexcept;

    mutable std::mutex mutex_;
    // Blocos atualmente alocados e em uso: ptr -> tamanho alocado no pool
    std::unordered_map<void*, size_t> allocated_blocks_;
    // Blocos livres em cache disponíveis para reuso: tamanho -> ptr
    std::multimap<size_t, void*> free_blocks_;

    size_t allocated_bytes_{0};
    size_t cached_bytes_{0};
    size_t peak_allocated_bytes_{0};
    size_t num_alloc_calls_{0};
    size_t num_free_calls_{0};
    size_t num_cuda_malloc_calls_{0};
    size_t num_cuda_free_calls_{0};
};

} // namespace cuda
} // namespace sore
