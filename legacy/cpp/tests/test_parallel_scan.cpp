#include "sore/cuda/scan.cuh"
#include "sore/nn/functional.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

void test_scan_gpu_vs_cpu_baseline() {
    std::cout << "[Test] Validação Numérica Cruzada: GPU Warp-Coalesced Scan vs CPU Baseline..." << std::endl;
    size_t B = 8;
    size_t T = 128;
    size_t D = 256;

    sore::Tensor A_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor X_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(42);
    // Gates a_t devem estar em (0, 1) para simular o sigmoid de decaimento do Griffin
    std::uniform_real_distribution<float> dist_a(0.1f, 0.95f);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);

    for (size_t i = 0; i < A_cpu.numel(); ++i) A_cpu.data<float>()[i] = dist_a(rng);
    for (size_t i = 0; i < X_cpu.numel(); ++i) X_cpu.data<float>()[i] = dist_x(rng);

    // 1. Execução no CPU Baseline
    auto t0_cpu = std::chrono::high_resolution_clock::now();
    sore::Tensor H_cpu = sore::functional::linear_rnn_forward_cpu(A_cpu, X_cpu);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();

    // 2. Execução no Kernel GPU Coalescido
    sore::Tensor A_gpu = A_cpu.cuda();
    sore::Tensor X_gpu = X_cpu.cuda();

    // Warm-up
    sore::Tensor H_gpu = sore::cuda::linear_rnn_forward_cuda(A_gpu, X_gpu);
    CUDA_SYNC_CHECK();

    auto t0_gpu = std::chrono::high_resolution_clock::now();
    H_gpu = sore::cuda::linear_rnn_forward_cuda(A_gpu, X_gpu);
    CUDA_SYNC_CHECK();
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1_gpu - t0_gpu).count();

    sore::Tensor H_back = H_gpu.cpu();

    // Comparação numérica de precisão
    float max_diff = 0.0f;
    for (size_t i = 0; i < H_cpu.numel(); ++i) {
        float diff = std::fabs(H_cpu.item(i) - H_back.item(i));
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << " -> Tempo CPU: " << cpu_ms << " ms | Tempo GPU Coalesced: " << gpu_ms << " ms" << std::endl;
    std::cout << " -> Aceleração (Speedup): " << (cpu_ms / gpu_ms) << "x" << std::endl;
    std::cout << " -> Erro absoluto máximo GPU vs CPU: " << max_diff << std::endl;
    assert(max_diff < 1e-4f);
    std::cout << " -> Validação Warp-Coalesced OK!" << std::endl;
}

void test_associative_scan_shared_memory() {
    std::cout << "[Test] Validação do Parallel Associative Scan (Kogge-Stone em Shared Memory)..." << std::endl;
    size_t B = 4;
    size_t T = 64; // Potência de 2 dentro do range de blockDim
    size_t D = 32;

    sore::Tensor A_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor X_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist_a(0.2f, 0.9f);
    std::uniform_real_distribution<float> dist_x(-0.5f, 0.5f);

    for (size_t i = 0; i < A_cpu.numel(); ++i) A_cpu.data<float>()[i] = dist_a(rng);
    for (size_t i = 0; i < X_cpu.numel(); ++i) X_cpu.data<float>()[i] = dist_x(rng);

    // CPU Baseline
    sore::Tensor H_cpu = sore::functional::linear_rnn_forward_cpu(A_cpu, X_cpu);

    // GPU Kogge-Stone Shared Memory Scan
    sore::Tensor A_gpu = A_cpu.cuda();
    sore::Tensor X_gpu = X_cpu.cuda();
    sore::Tensor H_assoc = sore::cuda::associative_scan_forward_cuda(A_gpu, X_gpu);
    CUDA_SYNC_CHECK();

    sore::Tensor H_back = H_assoc.cpu();

    float max_diff = 0.0f;
    for (size_t i = 0; i < H_cpu.numel(); ++i) {
        float diff = std::fabs(H_cpu.item(i) - H_back.item(i));
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << " -> Erro absoluto máximo Kogge-Stone Scan vs CPU: " << max_diff << std::endl;
    assert(max_diff < 1e-4f);
    std::cout << " -> Kogge-Stone Parallel Associative Scan validado com sucesso!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 4 - Parallel Scan CUDA ===" << std::endl;
    try {
        test_scan_gpu_vs_cpu_baseline();
        test_associative_scan_shared_memory();
        std::cout << "TODOS OS TESTES DO MÓDULO 4 PASSARAM COM SUCESSO!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha no teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
