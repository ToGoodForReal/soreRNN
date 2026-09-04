#include "sore/nn/linear_rnn.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void test_matmul_cpu() {
    std::cout << "[Test] GEMM CPU Otimizado (i-k-j)..." << std::endl;
    // A: 2x3, B: 3x2
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7, 8],
    //      [9, 1],
    //      [2, 3]]
    // C = A @ B =
    // [[1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3],   = [[31, 19],
    //  [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3]]   =  [85, 55]]
    sore::Tensor A({2, 3}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor B({3, 2}, sore::DType::Float32, sore::Device::CPU);

    float* a_ptr = A.data<float>();
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;

    float* b_ptr = B.data<float>();
    b_ptr[0] = 7.0f; b_ptr[1] = 8.0f;
    b_ptr[2] = 9.0f; b_ptr[3] = 1.0f;
    b_ptr[4] = 2.0f; b_ptr[5] = 3.0f;

    sore::Tensor C = sore::functional::matmul2d_cpu(A, B);
    assert(C.rank() == 2);
    assert(C.dim(0) == 2 && C.dim(1) == 2);

    assert(std::fabs(C.item(0) - 31.0f) < 1e-5f);
    assert(std::fabs(C.item(1) - 19.0f) < 1e-5f);
    assert(std::fabs(C.item(2) - 85.0f) < 1e-5f);
    assert(std::fabs(C.item(3) - 55.0f) < 1e-5f);

    std::cout << " -> GEMM CPU OK!" << std::endl;
}

void test_recurrence_analytical_step() {
    std::cout << "[Test] Validação Analítica da Recorrência Linear (h_t = a_t * h_{t-1} + x_t)..." << std::endl;
    // B=1, T=4, D=1
    // a_t constante = 0.5
    // x_t constante = 1.0
    // h_0 = 0.5 * 0    + 1.0 = 1.0
    // h_1 = 0.5 * 1.0  + 1.0 = 1.5
    // h_2 = 0.5 * 1.5  + 1.0 = 1.75
    // h_3 = 0.5 * 1.75 + 1.0 = 1.875
    sore::Tensor A({1, 4, 1}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor X({1, 4, 1}, sore::DType::Float32, sore::Device::CPU);
    A.fill_(0.5f);
    X.fill_(1.0f);

    sore::Tensor H = sore::functional::linear_rnn_forward_cpu(A, X, nullptr);
    assert(H.shape() == std::vector<size_t>({1, 4, 1}));

    assert(std::fabs(H.item(0) - 1.000f) < 1e-5f);
    assert(std::fabs(H.item(1) - 1.500f) < 1e-5f);
    assert(std::fabs(H.item(2) - 1.750f) < 1e-5f);
    assert(std::fabs(H.item(3) - 1.875f) < 1e-5f);

    std::cout << " -> Recorrência Analítica OK! Valores exatos: [1.0, 1.5, 1.75, 1.875]" << std::endl;
}

void test_full_linear_rnn_layer() {
    std::cout << "[Test] Camada Completa LinearRNN Forward Pass (Baseline CPU)..." << std::endl;
    size_t B = 2;
    size_t T = 16;
    size_t in_dim = 8;
    size_t hidden_dim = 16;
    size_t out_dim = 4;

    sore::nn::LinearRNN layer(in_dim, hidden_dim, out_dim, sore::Device::CPU);

    // Entrada randômica
    sore::Tensor x({B, T, in_dim}, sore::DType::Float32, sore::Device::CPU);
    x.fill_(0.25f);

    sore::Tensor y = layer.forward(x);

    assert(y.rank() == 3);
    assert(y.dim(0) == B);
    assert(y.dim(1) == T);
    assert(y.dim(2) == out_dim);

    // Checar por NaNs ou Infs
    const float* y_data = y.data<float>();
    for (size_t i = 0; i < y.numel(); ++i) {
        assert(!std::isnan(y_data[i]));
        assert(!std::isinf(y_data[i]));
    }

    std::cout << " -> Camada LinearRNN Forward CPU validada com sucesso! Saída: ["
              << y.dim(0) << ", " << y.dim(1) << ", " << y.dim(2) << "]" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 2 - Linear RNN CPU Baseline ===" << std::endl;
    try {
        test_matmul_cpu();
        test_recurrence_analytical_step();
        test_full_linear_rnn_layer();
        std::cout << "TODOS OS TESTES DO MÓDULO 2 PASSARAM COM SUCESSO!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha no teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
