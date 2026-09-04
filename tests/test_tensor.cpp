#include "sore/core/tensor.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void test_tensor_metadata_and_strides() {
    std::cout << "[Test] Tensor Shape, Strides e Metadados..." << std::endl;
    sore::Tensor t({2, 3, 4}, sore::DType::Float32, sore::Device::CPU);

    assert(t.rank() == 3);
    assert(t.dim(0) == 2);
    assert(t.dim(1) == 3);
    assert(t.dim(2) == 4);
    assert(t.numel() == 24);

    // Verificação matemática de Strides Row-Major (C-contiguous):
    // stride[2] = 1
    // stride[1] = 4 * 1 = 4
    // stride[0] = 3 * 4 = 12
    const auto& strides = t.strides();
    assert(strides.size() == 3);
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);
    assert(t.is_contiguous());

    t.print_info();
    std::cout << "\n -> Metadados e Strides OK!" << std::endl;
}

void test_tensor_zero_copy_view() {
    std::cout << "[Test] Zero-Copy View & Compartilhamento de Storage..." << std::endl;
    sore::Tensor original({2, 3, 4}, sore::DType::Float32, sore::Device::CPU);
    original.fill_(42.0f);

    assert(original.storage().use_count() == 1);

    // Reformatação para 2D [6, 4] com custo zero de alocação de dados
    sore::Tensor v = original.view({6, 4});

    assert(v.numel() == 24);
    assert(v.dim(0) == 6);
    assert(v.dim(1) == 4);
    assert(v.strides()[0] == 4);
    assert(v.strides()[1] == 1);

    // Devem compartilhar o MESMO bloco de memória física
    assert(v.storage() == original.storage());
    assert(v.data<float>() == original.data<float>());
    assert(original.storage().use_count() == 2);

    // Modificação através da view deve refletir imediatamente no tensor original
    v.data<float>()[0] = 999.0f;
    assert(original.data<float>()[0] == 999.0f);

    std::cout << " -> Zero-Copy View e ponteiro compartilhado OK!" << std::endl;
}

void test_tensor_h2d_d2h_transfer() {
    std::cout << "[Test] Transferência Bidirecional Host (CPU) <-> Device (GPU L40S)..." << std::endl;
    
    // 1. Criar e popular no Host
    constexpr size_t N = 1024;
    sore::Tensor cpu_tensor({N}, sore::DType::Float32, sore::Device::CPU);
    float* cpu_ptr = cpu_tensor.data<float>();
    for (size_t i = 0; i < N; ++i) {
        cpu_ptr[i] = static_cast<float>(i) * 0.5f - 100.0f;
    }

    // 2. Transferir para a GPU (H2D)
    sore::Tensor gpu_tensor = cpu_tensor.cuda();
    assert(gpu_tensor.device() == sore::Device::CUDA);
    assert(gpu_tensor.numel() == N);
    assert(gpu_tensor.storage() != cpu_tensor.storage()); // buffers físicos distintos

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, gpu_tensor.raw_data()));
    assert(attr.type == cudaMemoryTypeDevice);

    // 3. Transferir de volta para a CPU (D2H)
    sore::Tensor roundtrip = gpu_tensor.cpu();
    assert(roundtrip.device() == sore::Device::CPU);
    assert(roundtrip.numel() == N);

    // 4. Validar integridade exata dos valores
    for (size_t i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) * 0.5f - 100.0f;
        float actual = roundtrip.item(i);
        assert(std::fabs(actual - expected) < 1e-6f);
    }

    std::cout << " -> Transferência H2D e D2H com 1024 elementos validada com sucesso!" << std::endl;
}

void test_factory_methods() {
    std::cout << "[Test] Factory Methods (zeros, ones)..." << std::endl;

    auto z_cpu = sore::Tensor::zeros({4, 4}, sore::DType::Float32, sore::Device::CPU);
    for (size_t i = 0; i < 16; ++i) {
        assert(z_cpu.item(i) == 0.0f);
    }

    auto o_cpu = sore::Tensor::ones({4, 4}, sore::DType::Float32, sore::Device::CPU);
    for (size_t i = 0; i < 16; ++i) {
        assert(o_cpu.item(i) == 1.0f);
    }

    auto z_gpu = sore::Tensor::zeros({4, 4}, sore::DType::Float32, sore::Device::CUDA);
    auto z_back = z_gpu.cpu();
    for (size_t i = 0; i < 16; ++i) {
        assert(z_back.item(i) == 0.0f);
    }

    std::cout << " -> Factory methods OK!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 1 - Classe Tensor ===" << std::endl;
    try {
        test_tensor_metadata_and_strides();
        test_tensor_zero_copy_view();
        test_tensor_h2d_d2h_transfer();
        test_factory_methods();
        std::cout << "TODOS OS TESTES DA CLASSE TENSOR PASSARAM!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erro durante teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
