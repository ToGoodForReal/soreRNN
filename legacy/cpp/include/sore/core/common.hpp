#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace sore {

// Dispositivo onde os dados residem fisicamente
enum class Device {
    CPU,
    CUDA
};

inline std::string to_string(Device dev) {
    switch (dev) {
        case Device::CPU:  return "CPU";
        case Device::CUDA: return "CUDA";
        default:           return "Unknown";
    }
}

// Tipos de dados suportados
enum class DType {
    Float32
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        default:             return sizeof(float);
    }
}

inline std::string to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "Float32";
        default:             return "Unknown";
    }
}

// Macro essencial de HPC para verificação síncrona de status do CUDA Runtime
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::ostringstream oss;                                            \
            oss << "[CUDA Error] " << cudaGetErrorString(err)                  \
                << " (" << #call << ") at " << __FILE__ << ":" << __LINE__;   \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

namespace cuda {
inline bool is_sync_debug_enabled() {
    static bool checked = false;
    static bool enabled = false;
    if (!checked) {
        const char* env = std::getenv("SORE_SYNC_DEBUG");
        enabled = (env != nullptr && std::string(env) != "0");
        checked = true;
    }
    return enabled;
}

inline void synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[CUDA Error] " << cudaGetErrorString(err)
            << " at device synchronize";
        throw std::runtime_error(oss.str());
    }
}
} // namespace cuda

// Verificação de lançamento de kernel CUDA (assíncrona: sem stall do host por padrão)
#define CUDA_POST_KERNEL_CHECK()                                               \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        if (::sore::cuda::is_sync_debug_enabled()) {                           \
            CUDA_CHECK(cudaDeviceSynchronize());                               \
        }                                                                      \
    } while (0)

// Macro de compatibilidade: por padrão não faz stall de GPU/CPU, pipeline segue assíncrono
#define CUDA_SYNC_CHECK() CUDA_POST_KERNEL_CHECK()

// Sincronização explícita sob demanda do dispositivo GPU
#define CUDA_DEVICE_SYNC()                                                     \
    do {                                                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// Macro de checagem para cuBLAS
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::ostringstream oss;                                            \
            oss << "[cuBLAS Error] Status code " << status                     \
                << " (" << #call << ") at " << __FILE__ << ":" << __LINE__;   \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

} // namespace sore

