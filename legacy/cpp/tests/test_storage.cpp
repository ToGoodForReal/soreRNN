#include "sore/core/storage.hpp"
#include <cassert>
#include <iostream>
#include <vector>

void test_cpu_storage() {
    std::cout << "[Test] CPU Storage Alinhado (64 bytes)..." << std::endl;
    constexpr size_t bytes = 1024 * sizeof(float); // 1024 floats (4 KB)
    sore::Storage storage(bytes, sore::Device::CPU);

    assert(storage.data() != nullptr);
    assert(storage.size_bytes() == bytes);
    assert(storage.device() == sore::Device::CPU);

    // Validação de alinhamento estrito de 64 bytes (Cache Line / AVX-512)
    auto addr = reinterpret_cast<uintptr_t>(storage.data());
    assert(addr % sore::Storage::CPU_ALIGNMENT == 0);

    // Escrita e leitura na CPU
    float* fptr = storage.data_as<float>();
    for (size_t i = 0; i < 1024; ++i) {
        fptr[i] = static_cast<float>(i * 2);
    }
    for (size_t i = 0; i < 1024; ++i) {
        assert(fptr[i] == static_cast<float>(i * 2));
    }

    std::cout << " -> CPU Storage OK! Ponteiro: 0x" << std::hex << addr 
              << std::dec << " (alinhado a 64 bytes)" << std::endl;
}

void test_cuda_storage() {
    std::cout << "[Test] CUDA Storage (VRAM na GPU)..." << std::endl;
    constexpr size_t bytes = 1024 * sizeof(float);
    sore::Storage gpu_storage(bytes, sore::Device::CUDA);

    assert(gpu_storage.data() != nullptr);
    assert(gpu_storage.size_bytes() == bytes);
    assert(gpu_storage.device() == sore::Device::CUDA);

    // Verificar via CUDA Runtime se o ponteiro é de fato memória de Device
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, gpu_storage.data()));
    assert(attr.type == cudaMemoryTypeDevice);

    std::cout << " -> CUDA Storage OK! Ponteiro VRAM: " << gpu_storage.data() 
              << " verificado pelo driver CUDA." << std::endl;
}

void test_move_semantics() {
    std::cout << "[Test] Move Semantics (RAII)..." << std::endl;
    constexpr size_t bytes = 512 * sizeof(float);
    sore::Storage s1(bytes, sore::Device::CPU);
    void* original_ptr = s1.data();

    // Move construct
    sore::Storage s2(std::move(s1));
    assert(s2.data() == original_ptr);
    assert(s2.size_bytes() == bytes);
    assert(s1.data() == nullptr);
    assert(s1.size_bytes() == 0);

    std::cout << " -> Move Semantics OK!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 1 (Storage & Memory Management) ===" << std::endl;
    try {
        test_cpu_storage();
        test_cuda_storage();
        test_move_semantics();
        std::cout << "Todos os testes de Storage passaram com sucesso!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha no teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
