#include "sore/nn/autograd.hpp"
#include "sore/nn/functional.hpp"
#include "sore/cuda/scan.cuh"
#include "sore/optim/adamw.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>

void test_finite_difference_gradient_check() {
    std::cout << "[Test] Verificação Numérica de Gradientes por Diferenças Finitas (Finite Differences)..." << std::endl;
    size_t B = 2;
    size_t T = 4;
    size_t D = 2;

    sore::Tensor A({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor X({B, T, D}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_a(0.3f, 0.8f);
    std::uniform_real_distribution<float> dist_x(-0.5f, 0.5f);

    for (size_t i = 0; i < A.numel(); ++i) A.data<float>()[i] = dist_a(rng);
    for (size_t i = 0; i < X.numel(); ++i) X.data<float>()[i] = dist_x(rng);

    // Forward pass base
    sore::Tensor H = sore::functional::linear_rnn_forward_cpu(A, X);

    // Definimos Loss L = 0.5 * sum(H^2), logo dH = H
    sore::Tensor dH = H.clone();

    // Backward pass analítico
    sore::Tensor dA({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor dX({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::autograd::linear_rnn_backward_cpu(dH, A, H, nullptr, dA, dX);

    // Função auxiliar para calcular Loss = 0.5 * sum(H^2)
    auto compute_loss = [&](const sore::Tensor& cur_A, const sore::Tensor& cur_X) {
        sore::Tensor cur_H = sore::functional::linear_rnn_forward_cpu(cur_A, cur_X);
        float loss = 0.0f;
        for (size_t i = 0; i < cur_H.numel(); ++i) {
            float val = cur_H.item(i);
            loss += 0.5f * val * val;
        }
        return loss;
    };

    constexpr float eps = 1e-3f;

    // Checagem de gradiente numérico para X
    for (size_t i = 0; i < X.numel(); ++i) {
        float orig_x = X.item(i);

        X.data<float>()[i] = orig_x + eps;
        float loss_plus = compute_loss(A, X);

        X.data<float>()[i] = orig_x - eps;
        float loss_minus = compute_loss(A, X);

        X.data<float>()[i] = orig_x;

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float ana_grad = dX.item(i);

        float rel_err = std::fabs(num_grad - ana_grad) / (std::fabs(num_grad) + std::fabs(ana_grad) + 1e-7f);
        assert(rel_err < 1e-3f);
    }
    std::cout << " -> Gradientes dX validados com Diferenças Finitas!" << std::endl;

    // Checagem de gradiente numérico para A
    for (size_t i = 0; i < A.numel(); ++i) {
        float orig_a = A.item(i);

        A.data<float>()[i] = orig_a + eps;
        float loss_plus = compute_loss(A, X);

        A.data<float>()[i] = orig_a - eps;
        float loss_minus = compute_loss(A, X);

        A.data<float>()[i] = orig_a;

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float ana_grad = dA.item(i);

        float rel_err = std::fabs(num_grad - ana_grad) / (std::fabs(num_grad) + std::fabs(ana_grad) + 1e-5f);
        assert(rel_err < 5e-3f);
    }
    std::cout << " -> Gradientes dA validados com Diferenças Finitas!" << std::endl;
}

void test_bptt_cuda_vs_cpu() {
    std::cout << "[Test] Comparação de Precisão BPTT GPU vs CPU..." << std::endl;
    size_t B = 4;
    size_t T = 32;
    size_t D = 64;

    sore::Tensor A_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor X_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor dH_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    for (size_t i = 0; i < A_cpu.numel(); ++i) {
        A_cpu.data<float>()[i] = dist(rng);
        X_cpu.data<float>()[i] = dist(rng) - 0.5f;
        dH_cpu.data<float>()[i] = dist(rng) - 0.5f;
    }

    sore::Tensor H_cpu = sore::functional::linear_rnn_forward_cpu(A_cpu, X_cpu);

    sore::Tensor dA_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor dX_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::autograd::linear_rnn_backward_cpu(dH_cpu, A_cpu, H_cpu, nullptr, dA_cpu, dX_cpu);

    // GPU BPTT
    sore::Tensor A_gpu = A_cpu.cuda();
    sore::Tensor H_gpu = H_cpu.cuda();
    sore::Tensor dH_gpu = dH_cpu.cuda();
    sore::Tensor dA_gpu({B, T, D}, sore::DType::Float32, sore::Device::CUDA);
    sore::Tensor dX_gpu({B, T, D}, sore::DType::Float32, sore::Device::CUDA);

    sore::autograd::linear_rnn_backward_cuda(dH_gpu, A_gpu, H_gpu, nullptr, dA_gpu, dX_gpu);

    sore::Tensor dA_back = dA_gpu.cpu();
    sore::Tensor dX_back = dX_gpu.cpu();

    float max_diff_a = 0.0f;
    float max_diff_x = 0.0f;
    for (size_t i = 0; i < dA_cpu.numel(); ++i) {
        float diff_a = std::fabs(dA_cpu.item(i) - dA_back.item(i));
        float diff_x = std::fabs(dX_cpu.item(i) - dX_back.item(i));
        if (diff_a > max_diff_a) max_diff_a = diff_a;
        if (diff_x > max_diff_x) max_diff_x = diff_x;
    }

    std::cout << " -> Erro máximo dA GPU vs CPU: " << max_diff_a << std::endl;
    std::cout << " -> Erro máximo dX GPU vs CPU: " << max_diff_x << std::endl;
    assert(max_diff_a < 1e-5f);
    assert(max_diff_x < 1e-5f);
    std::cout << " -> BPTT GPU validado com precisão analítica!" << std::endl;
}

void test_fused_adamw_cuda_vs_cpu() {
    std::cout << "[Test] Fused AdamW GPU vs CPU..." << std::endl;
    size_t N = 1024;
    sore::Tensor p_cpu({N}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor g_cpu({N}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor m_cpu = sore::Tensor::zeros({N}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor v_cpu = sore::Tensor::zeros({N}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        p_cpu.data<float>()[i] = dist(rng);
        g_cpu.data<float>()[i] = dist(rng) * 0.1f;
    }

    // Clone para GPU
    sore::Tensor p_gpu = p_cpu.cuda();
    sore::Tensor g_gpu = g_cpu.cuda();
    sore::Tensor m_gpu = sore::Tensor::zeros({N}, sore::DType::Float32, sore::Device::CUDA);
    sore::Tensor v_gpu = sore::Tensor::zeros({N}, sore::DType::Float32, sore::Device::CUDA);

    // Rodar 5 passos do AdamW
    for (size_t step = 1; step <= 5; ++step) {
        sore::optim::fused_adamw_cpu(p_cpu, g_cpu, m_cpu, v_cpu, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-2f, step);
        sore::optim::fused_adamw_cuda(p_gpu, g_gpu, m_gpu, v_gpu, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-2f, step);
    }

    sore::Tensor p_back = p_gpu.cpu();

    float max_diff = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float diff = std::fabs(p_cpu.item(i) - p_back.item(i));
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << " -> Erro máximo AdamW GPU vs CPU: " << max_diff << std::endl;
    assert(max_diff < 1e-5f);
    std::cout << " -> Fused AdamW validado com sucesso!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 5 - Autograd & AdamW ===" << std::endl;
    try {
        test_finite_difference_gradient_check();
        test_bptt_cuda_vs_cpu();
        test_fused_adamw_cuda_vs_cpu();
        std::cout << "TODOS OS TESTES DO MÓDULO 5 PASSARAM COM SUCESSO!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha no teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
