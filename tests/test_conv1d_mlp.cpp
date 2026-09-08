#include "sore/cuda/ops.cuh"
#include "sore/nn/stacked_rnn.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>

void test_gelu_cuda() {
    std::cout << "[Test 1/4] GELU CUDA Forward e Backward vs Definição Analítica..." << std::endl;
    constexpr size_t N = 1024;
    sore::Tensor x_cpu({N}, sore::DType::Float32, sore::Device::CPU);
    float* ptr = x_cpu.data<float>();
    for (size_t i = 0; i < N; ++i) {
        ptr[i] = -3.0f + 6.0f * (static_cast<float>(i) / static_cast<float>(N));
    }

    sore::Tensor x_gpu = x_cpu.cuda();
    sore::Tensor y_gpu = sore::cuda::gelu_cuda(x_gpu);
    sore::Tensor y_back = y_gpu.cpu();

    float max_err = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float x = ptr[i];
        float expected = 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
        float err = std::fabs(y_back.item(i) - expected);
        if (err > max_err) max_err = err;
    }
    std::cout << " -> Erro máximo GELU forward: " << max_err << std::endl;
    assert(max_err < 1e-5f);

    // Teste de backward GELU
    sore::Tensor dout_gpu = sore::Tensor::ones({N}, sore::DType::Float32, sore::Device::CUDA);
    sore::Tensor dx_gpu = sore::cuda::gelu_backward_cuda(dout_gpu, x_gpu);
    sore::Tensor dx_back = dx_gpu.cpu();

    float max_bwd_err = 0.0f;
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    constexpr float INV_SQRT_2PI = 0.3989422804014327f;
    for (size_t i = 0; i < N; ++i) {
        float x = ptr[i];
        float cdf = 0.5f * (1.0f + std::erf(x * INV_SQRT2));
        float pdf = INV_SQRT_2PI * std::exp(-0.5f * x * x);
        float expected_dx = cdf + x * pdf;
        float err = std::fabs(dx_back.item(i) - expected_dx);
        if (err > max_bwd_err) max_bwd_err = err;
    }
    std::cout << " -> Erro máximo GELU backward: " << max_bwd_err << std::endl;
    assert(max_bwd_err < 1e-5f);
    std::cout << " -> [OK] GELU validado com sucesso!" << std::endl;
}

void test_conv1d_causal_cuda() {
    std::cout << "[Test 2/4] Causal Depthwise Conv1D (K=4) Forward vs CPU Loop..." << std::endl;
    size_t B = 2;
    size_t T = 8;
    size_t D = 16;
    size_t K = 4;

    sore::Tensor X_cpu({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor W_cpu({D, K}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor b_cpu({D}, sore::DType::Float32, sore::Device::CPU);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < X_cpu.numel(); ++i) X_cpu.data<float>()[i] = dist(rng);
    for (size_t i = 0; i < W_cpu.numel(); ++i) W_cpu.data<float>()[i] = dist(rng);
    for (size_t i = 0; i < b_cpu.numel(); ++i) b_cpu.data<float>()[i] = dist(rng);

    // CPU loop de referência
    sore::Tensor Y_ref({B, T, D}, sore::DType::Float32, sore::Device::CPU);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                float val = b_cpu.data<float>()[d];
                const float* w = W_cpu.data<float>() + d * 4;
                if (t >= 3) {
                    val += X_cpu.data<float>()[(b * T + t - 3) * D + d] * w[0];
                    val += X_cpu.data<float>()[(b * T + t - 2) * D + d] * w[1];
                    val += X_cpu.data<float>()[(b * T + t - 1) * D + d] * w[2];
                    val += X_cpu.data<float>()[(b * T + t) * D + d] * w[3];
                } else if (t == 2) {
                    val += X_cpu.data<float>()[(b * T + 0) * D + d] * w[1];
                    val += X_cpu.data<float>()[(b * T + 1) * D + d] * w[2];
                    val += X_cpu.data<float>()[(b * T + 2) * D + d] * w[3];
                } else if (t == 1) {
                    val += X_cpu.data<float>()[(b * T + 0) * D + d] * w[2];
                    val += X_cpu.data<float>()[(b * T + 1) * D + d] * w[3];
                } else if (t == 0) {
                    val += X_cpu.data<float>()[(b * T + 0) * D + d] * w[3];
                }
                Y_ref.data<float>()[(b * T + t) * D + d] = val;
            }
        }
    }

    // Execução GPU
    sore::Tensor X_gpu = X_cpu.cuda();
    sore::Tensor W_gpu = W_cpu.cuda();
    sore::Tensor b_gpu = b_cpu.cuda();
    sore::Tensor Y_gpu = sore::cuda::conv1d_causal_depthwise_forward_cuda(X_gpu, W_gpu, &b_gpu, B, T, D);
    sore::Tensor Y_back = Y_gpu.cpu();

    float max_err = 0.0f;
    for (size_t i = 0; i < Y_ref.numel(); ++i) {
        float err = std::fabs(Y_ref.item(i) - Y_back.item(i));
        if (err > max_err) max_err = err;
    }
    std::cout << " -> Erro máximo Conv1D Causal: " << max_err << std::endl;
    assert(max_err < 1e-5f);
    std::cout << " -> [OK] Conv1D Causal validado com sucesso!" << std::endl;
}

void test_stacked_rnn_150m_config() {
    std::cout << "[Test 3/4] Verificação de Parâmetros da soreRNN-LM v2..." << std::endl;
    sore::nn::StackedRNNConfig config;
    config.vocab_size = 50257;
    config.d_model = 1024;
    config.num_layers = 12;
    config.d_mlp = 2560;
    config.device = sore::Device::CUDA;

    sore::nn::StackedLinearRNNLM model(config);
    size_t total = model.total_parameters();
    double mb = static_cast<double>(total) * 4.0 / (1024.0 * 1024.0);

    std::cout << " -> Total de parâmetros alocados: " << total << " floats (" << mb << " MB)" << std::endl;
    // Orçamento esperado: ~152.3M parâmetros
    assert(total > 150000000 && total < 155000000);
    std::cout << " -> [OK] Contagem de parâmetros validada no alvo de ~152.3M!" << std::endl;
}

void test_forward_backward_pass() {
    std::cout << "[Test 4/4] Teste de Forward & Backward Completo na GPU L40S..." << std::endl;
    sore::nn::StackedRNNConfig config;
    config.vocab_size = 1000; // vocabulário reduzido para teste rápido
    config.d_model = 64;
    config.num_layers = 2;
    config.d_mlp = 160;
    config.device = sore::Device::CUDA;

    sore::nn::StackedLinearRNNLM model(config);

    size_t B = 2;
    size_t T = 8;
    std::vector<uint16_t> tokens = {10, 20, 30, 40, 50, 60, 70, 80,
                                    11, 21, 31, 41, 51, 61, 71, 81};
    std::vector<uint16_t> targets = {20, 30, 40, 50, 60, 70, 80, 90,
                                     21, 31, 41, 51, 61, 71, 81, 91};

    sore::Tensor logits = model.forward(tokens, B, T);
    assert(logits.shape() == std::vector<size_t>({B, T, config.vocab_size}));

    sore::Tensor d_logits = sore::Tensor::zeros(logits.shape(), sore::DType::Float32, sore::Device::CUDA);
    float loss = sore::cuda::cross_entropy_loss_and_grad_cuda(logits, targets, d_logits);
    std::cout << " -> Loss inicial: " << loss << std::endl;
    assert(!std::isnan(loss) && loss > 0.0f);

    model.backward(d_logits, tokens);

    auto grads = model.gradients();
    float grad_norm = sore::cuda::clip_grad_norm_cuda(grads, 1.0f);
    std::cout << " -> Grad norm: " << grad_norm << std::endl;
    assert(!std::isnan(grad_norm) && grad_norm > 0.0f);

    std::cout << " -> [OK] Forward e Backward executados sem NaNs e com gradientes válidos!" << std::endl;
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  EXECUÇÃO DA SUÍTE DE TESTES: Conv1D, GELU & soreRNN-v2" << std::endl;
    std::cout << "=================================================" << std::endl;
    test_gelu_cuda();
    test_conv1d_causal_cuda();
    test_stacked_rnn_150m_config();
    test_forward_backward_pass();
    std::cout << "\n>>> TODOS OS 4 TESTES PASSARAM COM 100% DE SUCESSO! <<<\n" << std::endl;
    return 0;
}
