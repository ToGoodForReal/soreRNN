#include "sore/cuda/caching_allocator.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace sore {
namespace cuda {

CUDACachingAllocator& CUDACachingAllocator::instance() {
    static CUDACachingAllocator allocator;
    return allocator;
}

CUDACachingAllocator::~CUDACachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    empty_cache_internal();
    for (auto& pair : allocated_blocks_) {
        cudaFree(pair.first);
    }
    allocated_blocks_.clear();
}

size_t CUDACachingAllocator::round_size(size_t size) {
    if (size == 0) return ALIGNMENT;
    // Alinhamento mínimo de 256 bytes
    size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    if (size <= 1024 * 1024) {
        // Até 1 MB: blocos em múltiplos de 4 KB
        constexpr size_t BIN = 4096;
        size = (size + BIN - 1) & ~(BIN - 1);
    } else if (size <= 32 * 1024 * 1024) {
        // De 1 MB a 32 MB: blocos em múltiplos de 512 KB
        constexpr size_t BIN = 512 * 1024;
        size = (size + BIN - 1) & ~(BIN - 1);
    } else {
        // Acima de 32 MB (ex: matrizes gigantes de logits): blocos em múltiplos de 2 MB
        constexpr size_t BIN = 2 * 1024 * 1024;
        size = (size + BIN - 1) & ~(BIN - 1);
    }
    return size;
}

void* CUDACachingAllocator::allocate(size_t size_bytes) {
    if (size_bytes == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    num_alloc_calls_++;

    size_t needed = round_size(size_bytes);

    // Busca melhor bloco livre em cache com capacidade >= needed
    auto it = free_blocks_.lower_bound(needed);
    if (it != free_blocks_.end()) {
        size_t block_size = it->first;
        void* ptr = it->second;
        free_blocks_.erase(it);

        cached_bytes_ -= block_size;
        allocated_bytes_ += block_size;
        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);
        allocated_blocks_[ptr] = block_size;
        return ptr;
    }

    // Se nenhum bloco adequado existe no cache, solicita nova alocação ao driver CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, needed);

    if (err != cudaSuccess) {
        // Tentativa de recuperação OOM: esvazia blocos livres em cache e tenta novamente
        empty_cache_internal();
        cudaGetLastError(); // Limpa estado de erro interno da runtime
        err = cudaMalloc(&ptr, needed);
    }

    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[CUDACachingAllocator] Falha de memória GPU (OOM) ao tentar alocar "
            << needed << " bytes (" << (needed / (1024.0 * 1024.0)) << " MB): "
            << cudaGetErrorString(err)
            << " | Alocado ativo: " << (allocated_bytes_ / (1024.0 * 1024.0)) << " MB"
            << " | Cache livre: " << (cached_bytes_ / (1024.0 * 1024.0)) << " MB";
        throw std::runtime_error(oss.str());
    }

    num_cuda_malloc_calls_++;
    allocated_bytes_ += needed;
    peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);
    allocated_blocks_[ptr] = needed;
    return ptr;
}

void CUDACachingAllocator::deallocate(void* ptr) noexcept {
    if (ptr == nullptr) return;

    std::lock_guard<std::mutex> lock(mutex_);
    num_free_calls_++;

    auto it = allocated_blocks_.find(ptr);
    if (it != allocated_blocks_.end()) {
        size_t size = it->second;
        allocated_blocks_.erase(it);

        allocated_bytes_ -= size;
        cached_bytes_ += size;
        free_blocks_.insert({size, ptr});
    } else {
        // Caso ponteiro não seja do pool gerenciado, libera diretamente
        cudaFree(ptr);
    }
}

void CUDACachingAllocator::empty_cache() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    empty_cache_internal();
}

void CUDACachingAllocator::empty_cache_internal() noexcept {
    for (auto& pair : free_blocks_) {
        cudaError_t err = cudaFree(pair.second);
        if (err == cudaSuccess) {
            num_cuda_free_calls_++;
        }
    }
    free_blocks_.clear();
    cached_bytes_ = 0;
}

AllocatorStats CUDACachingAllocator::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    AllocatorStats s;
    s.allocated_bytes = allocated_bytes_;
    s.cached_bytes = cached_bytes_;
    s.reserved_bytes = allocated_bytes_ + cached_bytes_;
    s.peak_allocated_bytes = peak_allocated_bytes_;
    s.num_alloc_calls = num_alloc_calls_;
    s.num_free_calls = num_free_calls_;
    s.num_cuda_malloc_calls = num_cuda_malloc_calls_;
    s.num_cuda_free_calls = num_cuda_free_calls_;
    return s;
}

void CUDACachingAllocator::reset_peak_stats() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_allocated_bytes_ = allocated_bytes_;
}

} // namespace cuda
} // namespace sore
