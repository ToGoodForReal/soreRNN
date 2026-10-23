#include "sore/data/dataloader.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

void test_mmap_dataset_and_batches() {
    std::cout << "[Test] Leitor de Dataset mmap (Zero-Copy) e DataLoader..." << std::endl;

    std::string test_file = "test_tokens.bin";
    constexpr size_t N_TOKENS = 1000;
    std::vector<uint16_t> sample_tokens(N_TOKENS);
    for (size_t i = 0; i < N_TOKENS; ++i) {
        sample_tokens[i] = static_cast<uint16_t>(i % 256);
    }

    // Cria arquivo binário para teste
    {
        std::ofstream ofs(test_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(sample_tokens.data()), N_TOKENS * sizeof(uint16_t));
    }

    // 1. Carrega via POSIX mmap
    auto dataset = std::make_shared<sore::data::MMapDataset>(test_file);
    assert(dataset->total_tokens() == N_TOKENS);

    // Valida que o ponteiro virtual de mmap contém os dados exatos
    const uint16_t* ptr = dataset->token_data();
    for (size_t i = 0; i < N_TOKENS; ++i) {
        assert(ptr[i] == sample_tokens[i]);
    }
    std::cout << " -> Mapeamento de memória mmap validado com sucesso!" << std::endl;

    // 2. Criação do DataLoader
    size_t B = 4;
    size_t T = 16;
    sore::data::DataLoader loader(dataset, B, T);
    assert(loader.has_next());

    size_t batches_read = 0;
    while (loader.has_next() && batches_read < 5) {
        auto batch = loader.next();
        assert(batch.inputs.size() == B * T);
        assert(batch.targets.size() == B * T);

        // Verifica que o alvo é exatamente o próximo token no tempo: target[t] == input[t] + 1 (em termos de sequência)
        for (size_t i = 0; i < B * T; ++i) {
            size_t token_idx = (batches_read * B * T) + i;
            assert(batch.inputs[i] == sample_tokens[token_idx]);
            assert(batch.targets[i] == sample_tokens[token_idx + 1]);
        }
        batches_read++;
    }

    std::cout << " -> DataLoader: " << batches_read << " batches (B=4, T=16) gerados com alinhamento temporal perfeito!" << std::endl;

    // Limpeza
    dataset.reset();
    std::filesystem::remove(test_file);
}

int main() {
    std::cout << "=== Bateria de Testes: Módulo 6 - Pipeline de Dados com mmap ===" << std::endl;
    try {
        test_mmap_dataset_and_batches();
        std::cout << "TESTES DO PIPELINE DE DADOS PASSARAM COM SUCESSO!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha no teste: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
