#include "sore/cuda/ops.cuh"
#include "sore/nn/functional.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>

void test_cublas_gemm_vs_cpu() {
    std::cout << "[Test] cuBLAS GEMM (GPU CUDA) vs Baseline CPU..." << std::endl;
    size_t M = 128;
    size_t K = 256;
    size_t N = 64;

    sore::Tensor A_cpu({M, K}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor B_cpu({K, N}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* a_ptr = A_cpu.data<float>();
    for (size_t i = 0; i < A_cpu.numel(); ++i) a_ptr[i] = dist(rng);

    float* b_ptr = B_cpu.data<float>();
    for (size_t i = 0; i < B_cpu.numel(); ++i) b_ptr[i] = dist(rng);

    // Execução CPU
    sore::Tensor C_cpu = sore::functional::matmul2d_cpu(A_cpu, B_cpu);

    // Execução GPU via cuBLAS
    sore::Tensor A_gpu = A_cpu.cuda();
    sore::Tensor B_gpu = B_cpu.cuda();
    sore::Tensor C_gpu = sore::cuda::matmul2d_cuda(A_gpu, B_gpu);

    sore::Tensor C_back = C_gpu.cpu();

    // Validação de erro relativo / absoluto
    float max_diff = 0.0f;
    for (size_t i = 0; i < C_cpu.numel(); ++i) {
        float diff = std::fabs(C_cpu.item(i) - C_back.item(i));
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << " -> Erro absoluto máximo cuBLAS vs CPU: " << max_diff << std::endl;
    // Com TF32 (TensorFloat-32 dos Tensor Cores), a mantissa tem 10 bits (~1e-2f de precisão sobre somas)
    assert(max_diff < 1e-2f);
    std::cout << " -> cuBLAS GEMM validado com precisão numérica TF32!" << std::endl;
}

void test_cublas_linear_vs_cpu() {
    std::cout << "[Test] cuBLAS Linear Projection com Viés (GPU vs CPU)..." << std::endl;
    size_t B = 4;
    size_t T = 32;
    size_t D_in = 64;
    size_t D_out = 128;

    sore::Tensor X_cpu({B, T, D_in}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor W_cpu({D_out, D_in}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor b_cpu({D_out}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < X_cpu.numel(); ++i) X_cpu.data<float>()[i] = dist(rng);
    for (size_t i = 0; i < W_cpu.numel(); ++i) W_cpu.data<float>()[i] = dist(rng);
    for (size_t i = 0; i < b_cpu.numel(); ++i) b_cpu.data<float>()[i] = dist(rng);

    // CPU
    sore::Tensor Y_cpu = sore::functional::linear_cpu(X_cpu, W_cpu, &b_cpu);

    // GPU
    sore::Tensor X_gpu = X_cpu.cuda();
    sore::Tensor W_gpu = W_cpu.cuda();
    sore::Tensor b_gpu = b_cpu.cuda();
    sore::Tensor Y_gpu = sore::cuda::linear_cuda(X_gpu, W_gpu, &b_gpu);

    sore::Tensor Y_back = Y_gpu.cpu();

    float max_diff = 0.0f;
    for (size_t i = 0; i < Y_cpu.numel(); ++i) {
        float diff = std::fabs(Y_cpu.item(i) - Y_back.item(i));
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << " -> Erro absoluto máximo Linear Proj GPU vs CPU: " << max_diff << std::endl;
    assert(max_diff < 1e-2f);
    std::cout << " -> cuBLAS Linear Projection validada com TF32!" << std::endl;
}

void test_cuda_sigmoid() {
    std::cout << "[Test] CUDA SFU Sigmoid Kernel..." << std::endl;
    size_t N = 10000;
    sore::Tensor X_cpu({N}, sore::DType::Float32, sore::Device::CPU);
    for (size_t i = 0; i < N; ++i) {
        X_cpu.data<float>()[i] = (static_cast<float>(i) - 5000.0f) * 0.01f;
    }

    sore::Tensor S_cpu = sore::functional::sigmoid_cpu(X_cpu);

    sore::Tensor X_gpu = X_cpu.cuda();
    sore::Tensor S_gpu = sore::cuda::sigmoid_cuda(X_gpu);
    sore::Tensor S_back = S_gpu.cpu();

    float max_diff = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float diff = std::fabs(S_cpu.item(i) - S_back.item(i));
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << " -> Erro máximo Sigmoid GPU vs CPU: " << max_diff << std::endl;
    assert(max_diff < 1e-5f);
    std::cout << " -> Sigmoid CUDA SFU validado!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 3 - cuBLAS e Kernels GPU ===" << std::endl;
    try {
        test_cublas_gemm_vs_cpu();
        test_cublas_linear_vs_cpu();
        test_cuda_sigmoid();
        std::cout << "TODOS OS TESTES DO MÓDULO 3 PASSARAM COM SUCESSO!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha no teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
