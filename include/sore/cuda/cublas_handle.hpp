#pragma once

#include "sore/core/common.hpp"

namespace sore {
namespace cuda {

/**
 * @brief Encapsulamento RAII Singleton do handle cuBLAS.
 * 
 * Garante criação única e destruição limpa do contexto cuBLAS,
 * evitando custo de inicialização repetida no pipeline.
 */
class CublasHandle {
public:
    static CublasHandle& instance() {
        static CublasHandle inst;
        return inst;
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

private:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        // Configura modo de matemática determinística / padrão
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
    }

    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }

    cublasHandle_t handle_{nullptr};
};

inline cublasHandle_t get_cublas_handle() {
    return CublasHandle::instance().get();
}

} // namespace cuda
} // namespace sore
