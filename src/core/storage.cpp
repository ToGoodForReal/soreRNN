#include "sore/core/storage.hpp"
#include "sore/cuda/caching_allocator.hpp"
#include <new>

namespace sore {

Storage::Storage(size_t size_bytes, Device device)
    : size_bytes_(size_bytes), device_(device) {
    if (size_bytes_ > 0) {
        allocate();
    }
}

Storage::~Storage() {
    deallocate();
}

Storage::Storage(Storage&& other) noexcept
    : data_(other.data_), size_bytes_(other.size_bytes_), device_(other.device_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        device_ = other.device_;
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

void Storage::allocate() {
    if (device_ == Device::CPU) {
        // Alocação alinhada em 64 bytes para vetorização AVX-512 e alinhamento de cache
        void* ptr = nullptr;
        int res = posix_memalign(&ptr, CPU_ALIGNMENT, size_bytes_);
        if (res != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        data_ = ptr;
    } else if (device_ == Device::CUDA) {
        // Alocação de alta performance via pool de memória VRAM em cache
        data_ = cuda::CUDACachingAllocator::instance().allocate(size_bytes_);
    }
}

void Storage::deallocate() noexcept {
    if (data_ != nullptr) {
        if (device_ == Device::CPU) {
            std::free(data_);
        } else if (device_ == Device::CUDA) {
            // Devolução instantânea para o cache do pool (zero-overhead)
            cuda::CUDACachingAllocator::instance().deallocate(data_);
        }
        data_ = nullptr;
        size_bytes_ = 0;
    }
}

} // namespace sore
