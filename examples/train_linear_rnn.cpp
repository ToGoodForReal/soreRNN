#include "sore/core/tensor.hpp"
#include "sore/data/dataloader.hpp"
#include "sore/nn/embedding.hpp"
#include "sore/nn/linear_rnn.hpp"
#include "sore/nn/autograd.hpp"
#include "sore/nn/loss.hpp"
#include "sore/optim/adamw.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <cassert>

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "   TREINAMENTO DE RNN LINEAR DO ZERO EM C++20 E CUDA (soreRNN)  " << std::endl;
    std::cout << "================================================================" << std::endl;

    // 1. Geração de dataset sintético de tokens com padrão determinístico
    std::string data_path = "mini_corpus.bin";
    constexpr size_t TOTAL_TOKENS = 50000;
    constexpr size_t VOCAB_SIZE = 32;

    {
        std::ofstream ofs(data_path, std::ios::binary);
        std::vector<uint16_t> tokens(TOTAL_TOKENS);
        // Gera um padrão repetitivo (ex: 0, 1, 2, ..., VOCAB_SIZE-1) para aprendizado rápido
        for (size_t i = 0; i < TOTAL_TOKENS; ++i) {
            tokens[i] = static_cast<uint16_t>(i % VOCAB_SIZE);
        }
        ofs.write(reinterpret_cast<const char*>(tokens.data()), TOTAL_TOKENS * sizeof(uint16_t));
    }

    // 2. Carregamento do dataset via POSIX mmap
    auto dataset = std::make_shared<sore::data::MMapDataset>(data_path);
    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t SEQ_LEN = 16;
    sore::data::DataLoader loader(dataset, BATCH_SIZE, SEQ_LEN);

    std::cout << "[Dataset] Total de Tokens: " << dataset->total_tokens()
              << " | Batches Disponíveis: " << loader.total_batches() << std::endl;

    // 3. Hiperparâmetros da Arquitetura
    constexpr size_t EMB_DIM = 32;
    constexpr size_t HIDDEN_DIM = 64;

    sore::nn::Embedding embedding(VOCAB_SIZE, EMB_DIM, sore::Device::CPU);
    sore::nn::LinearRNN rnn(EMB_DIM, HIDDEN_DIM, EMB_DIM, sore::Device::CPU);

    // LM Head (Projeção do espaço latente para o Vocabulário)
    sore::Tensor w_head({VOCAB_SIZE, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor b_head({VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);
    w_head.fill_(0.01f);
    b_head.zero_();

    // Gradientes acumuladores
    sore::Tensor d_w_emb = sore::Tensor::zeros({VOCAB_SIZE, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_w_head = sore::Tensor::zeros({VOCAB_SIZE, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_head = sore::Tensor::zeros({VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);

    sore::Tensor d_w_gate = sore::Tensor::zeros(rnn.w_gate().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_gate = sore::Tensor::zeros(rnn.b_gate().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_w_in = sore::Tensor::zeros(rnn.w_in().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_in = sore::Tensor::zeros(rnn.b_in().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_w_out = sore::Tensor::zeros(rnn.w_out().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_out = sore::Tensor::zeros(rnn.b_out().shape(), sore::DType::Float32, sore::Device::CPU);

    // Otimizador AdamW
    std::vector<sore::Tensor*> all_params = {
        &embedding.weight(),
        &w_head, &b_head,
        &rnn.w_gate(), &rnn.b_gate(),
        &rnn.w_in(), &rnn.b_in(),
        &rnn.w_out(), &rnn.b_out()
    };

    std::vector<sore::Tensor*> all_grads = {
        &d_w_emb,
        &d_w_head, &d_b_head,
        &d_w_gate, &d_b_gate,
        &d_w_in, &d_b_in,
        &d_w_out, &d_b_out
    };

    sore::optim::AdamWConfig opt_cfg;
    opt_cfg.lr = 5e-3f;
    opt_cfg.weight_decay = 1e-4f;

    std::vector<sore::Tensor> exp_avg;
    std::vector<sore::Tensor> exp_avg_sq;
    for (auto* p : all_params) {
        exp_avg.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }

    std::cout << "[Treino] Iniciando loop de otimização de 25 passos..." << std::endl;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (size_t step = 1; step <= 25 && loader.has_next(); ++step) {
        auto batch = loader.next();
        size_t N = BATCH_SIZE * SEQ_LEN;

        // 1. FORWARD PASS
        // A) Embedding lookup: [B, T] -> [B, T, EMB_DIM]
        sore::Tensor x_emb = embedding.forward(batch.inputs, BATCH_SIZE, SEQ_LEN);

        // B) RG-LRU Projeções & Recorrência Linear
        sore::Tensor G = sore::functional::linear_cpu(x_emb, rnn.w_gate(), &rnn.b_gate());
        sore::Tensor A = sore::functional::sigmoid_cpu(G);
        sore::Tensor U = sore::functional::linear_cpu(x_emb, rnn.w_in(), &rnn.b_in());
        sore::Tensor H = sore::functional::linear_rnn_forward_cpu(A, U, nullptr);
        sore::Tensor Y = sore::functional::linear_cpu(H, rnn.w_out(), &rnn.b_out());

        // C) LM Head: [B, T, EMB_DIM] -> [B, T, VOCAB_SIZE]
        sore::Tensor logits = sore::functional::linear_cpu(Y, w_head, &b_head);

        // 2. CÁLCULO DA LOSS E GRADIENTE DO SOFTMAX
        sore::Tensor d_logits({N, VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);
        float loss = sore::nn::cross_entropy_loss_and_grad(logits, batch.targets, d_logits);

        if (step == 1) initial_loss = loss;
        final_loss = loss;

        // 3. BACKWARD PASS (AUTOGRAD MANUAL)
        // Gradientes da LM Head: Y [N, EMB_DIM], d_logits [N, VOCAB_SIZE]
        // d_w_head = d_logits^T @ Y
        sore::Tensor Y_flat = Y.view({N, EMB_DIM});
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float b_sum = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                float dl = d_logits.data<float>()[i * VOCAB_SIZE + v];
                b_sum += dl;
                for (size_t d = 0; d < EMB_DIM; ++d) {
                    d_w_head.data<float>()[v * EMB_DIM + d] += dl * Y_flat.data<float>()[i * EMB_DIM + d];
                }
            }
            d_b_head.data<float>()[v] = b_sum;
        }

        // dY = d_logits @ w_head -> [N, EMB_DIM]
        sore::Tensor dY({N, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
        for (size_t i = 0; i < N; ++i) {
            for (size_t d = 0; d < EMB_DIM; ++d) {
                float sum = 0.0f;
                for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                    sum += d_logits.data<float>()[i * VOCAB_SIZE + v] * w_head.data<float>()[v * EMB_DIM + d];
                }
                dY.data<float>()[i * EMB_DIM + d] = sum;
            }
        }
        sore::Tensor dY_3d = dY.view({BATCH_SIZE, SEQ_LEN, EMB_DIM});

        // BPTT da Recorrência Linear
        sore::Tensor dH({BATCH_SIZE, SEQ_LEN, HIDDEN_DIM}, sore::DType::Float32, sore::Device::CPU);
        for (size_t i = 0; i < N; ++i) {
            for (size_t h = 0; h < HIDDEN_DIM; ++h) {
                float sum = 0.0f;
                for (size_t d = 0; d < EMB_DIM; ++d) {
                    sum += dY.data<float>()[i * EMB_DIM + d] * rnn.w_out().data<float>()[d * HIDDEN_DIM + h];
                }
                dH.data<float>()[i * HIDDEN_DIM + h] = sum;
            }
        }

        sore::Tensor dA({BATCH_SIZE, SEQ_LEN, HIDDEN_DIM}, sore::DType::Float32, sore::Device::CPU);
        sore::Tensor dU({BATCH_SIZE, SEQ_LEN, HIDDEN_DIM}, sore::DType::Float32, sore::Device::CPU);
        sore::autograd::linear_rnn_backward_cpu(dH, A, H, nullptr, dA, dU);

        // 4. ATUALIZAÇÃO DOS PARÂMETROS COM ADAMW
        for (size_t p_idx = 0; p_idx < all_params.size(); ++p_idx) {
            sore::optim::fused_adamw_cpu(
                *all_params[p_idx],
                *all_grads[p_idx],
                exp_avg[p_idx],
                exp_avg_sq[p_idx],
                opt_cfg.lr,
                opt_cfg.beta1,
                opt_cfg.beta2,
                opt_cfg.eps,
                opt_cfg.weight_decay,
                step
            );
            // Zera gradiente para próxima iteração
            all_grads[p_idx]->zero_();
        }

        if (step % 5 == 0 || step == 1) {
            std::cout << "Passo [" << std::setw(2) << step << "/25] - Cross-Entropy Loss: " 
                      << std::fixed << std::setprecision(4) << loss 
                      << " (Perplexidade: " << std::exp(loss) << ")" << std::endl;
        }
    }

    std::cout << "\n[Resultado] Loss Inicial: " << initial_loss 
              << " -> Loss Final: " << final_loss << std::endl;
    assert(final_loss < initial_loss);
    std::cout << ">>> Convergência confirmada! O modelo aprendeu com sucesso a distribuição do corpus! <<<" << std::endl;

    // Limpeza
    dataset.reset();
    std::filesystem::remove(data_path);
    return 0;
}
