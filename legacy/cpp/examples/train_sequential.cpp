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

void save_checkpoint(const std::string& filepath, const std::vector<sore::Tensor*>& params) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[Erro] Não foi possível salvar checkpoint em: " << filepath << std::endl;
        return;
    }
    for (const auto* p : params) {
        size_t bytes = p->numel() * sizeof(float);
        sore::Tensor cpu_tensor = (p->device() == sore::Device::CUDA) ? p->cpu() : *p;
        ofs.write(reinterpret_cast<const char*>(cpu_tensor.data<float>()), bytes);
    }
    std::cout << " -> Checkpoint salvo com sucesso: " << filepath << " (" 
              << std::filesystem::file_size(filepath) / (1024.0f * 1024.0f) << " MB)" << std::endl;
}

int main(int argc, char** argv) {
    std::string pretrain_path = "data/pretrain_75pt_25en.bin";
    std::string sft_path = "data/sft_chat_pt.bin";

    size_t pretrain_steps = 30;
    size_t sft_steps = 20;

    if (argc > 1) pretrain_steps = std::stoul(argv[1]);
    if (argc > 2) sft_steps = std::stoul(argv[2]);

    std::cout << "=================================================================" << std::endl;
    std::cout << "  soreRNN: TREINAMENTO SEQUENCIAL (PRÉ-TREINO 75/25 + SFT CHAT)  " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Configuração: " << pretrain_steps << " passos de Pré-Treino | " 
              << sft_steps << " passos de SFT Conversacional\n" << std::endl;

    std::filesystem::create_directories("checkpoints");

    // Vocabulário Tiktoken GPT-2
    constexpr size_t VOCAB_SIZE = 50257;
    constexpr size_t EMB_DIM = 32;
    constexpr size_t HIDDEN_DIM = 64;
    constexpr size_t BATCH_SIZE = 4;
    constexpr size_t SEQ_LEN = 32;

    // Inicialização do Modelo
    sore::nn::Embedding embedding(VOCAB_SIZE, EMB_DIM, sore::Device::CPU);
    sore::nn::LinearRNN rnn(EMB_DIM, HIDDEN_DIM, EMB_DIM, sore::Device::CPU);

    sore::Tensor w_head({VOCAB_SIZE, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor b_head({VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);
    w_head.fill_(0.01f);
    b_head.zero_();

    // Tensores de gradiente acumuladores
    sore::Tensor d_w_emb = sore::Tensor::zeros({VOCAB_SIZE, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_w_head = sore::Tensor::zeros({VOCAB_SIZE, EMB_DIM}, sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_head = sore::Tensor::zeros({VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);

    sore::Tensor d_w_gate = sore::Tensor::zeros(rnn.w_gate().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_gate = sore::Tensor::zeros(rnn.b_gate().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_w_in = sore::Tensor::zeros(rnn.w_in().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_in = sore::Tensor::zeros(rnn.b_in().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_w_out = sore::Tensor::zeros(rnn.w_out().shape(), sore::DType::Float32, sore::Device::CPU);
    sore::Tensor d_b_out = sore::Tensor::zeros(rnn.b_out().shape(), sore::DType::Float32, sore::Device::CPU);

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

    std::vector<sore::Tensor> exp_avg;
    std::vector<sore::Tensor> exp_avg_sq;
    for (auto* p : all_params) {
        exp_avg.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }

    // ---------------------------------------------------------
    // FASE 1: PRÉ-TREINAMENTO (75% PT-BR / 25% EN)
    // ---------------------------------------------------------
    std::cout << ">>> FASE 1: INICIANDO PRÉ-TREINAMENTO (75% PT / 25% EN) <<<" << std::endl;
    auto pretrain_ds = std::make_shared<sore::data::MMapDataset>(pretrain_path);
    sore::data::DataLoader pretrain_loader(pretrain_ds, BATCH_SIZE, SEQ_LEN);
    std::cout << "[Pretrain Dataset] " << pretrain_ds->total_tokens() << " tokens mapeados via mmap.\n";

    float lr_pretrain = 3e-3f;
    float weight_decay = 1e-4f;

    for (size_t step = 1; step <= pretrain_steps && pretrain_loader.has_next(); ++step) {
        auto batch = pretrain_loader.next();
        size_t N = BATCH_SIZE * SEQ_LEN;

        // Forward
        sore::Tensor x_emb = embedding.forward(batch.inputs, BATCH_SIZE, SEQ_LEN);
        sore::Tensor G = sore::functional::linear_cpu(x_emb, rnn.w_gate(), &rnn.b_gate());
        sore::Tensor A = sore::functional::sigmoid_cpu(G);
        sore::Tensor U = sore::functional::linear_cpu(x_emb, rnn.w_in(), &rnn.b_in());
        sore::Tensor H = sore::functional::linear_rnn_forward_cpu(A, U, nullptr);
        sore::Tensor Y = sore::functional::linear_cpu(H, rnn.w_out(), &rnn.b_out());
        sore::Tensor logits = sore::functional::linear_cpu(Y, w_head, &b_head);

        // Loss & Grad
        sore::Tensor d_logits({N, VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);
        float loss = sore::nn::cross_entropy_loss_and_grad(logits, batch.targets, d_logits);

        // Backward LM Head
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

        // BPTT Linear RNN
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

        // AdamW Update
        for (size_t p_idx = 0; p_idx < all_params.size(); ++p_idx) {
            sore::optim::fused_adamw_cpu(
                *all_params[p_idx], *all_grads[p_idx], exp_avg[p_idx], exp_avg_sq[p_idx],
                lr_pretrain, 0.9f, 0.999f, 1e-8f, weight_decay, step
            );
            all_grads[p_idx]->zero_();
        }

        if (step % 10 == 0 || step == 1 || step == pretrain_steps) {
            std::cout << "[Pré-treino Passo " << std::setw(3) << step << "/" << pretrain_steps 
                      << "] Loss: " << std::fixed << std::setprecision(4) << loss 
                      << " (PPL: " << std::exp(loss) << ")" << std::endl;
        }
    }

    save_checkpoint("checkpoints/checkpoint_pretrain.bin", all_params);

    // ---------------------------------------------------------
    // FASE 2: SFT / FINE-TUNING CONVERSACIONAL EM PT-BR
    // ---------------------------------------------------------
    std::cout << "\n>>> FASE 2: INICIANDO FINE-TUNING INSTRUCIONAL (SFT CHAT PT-BR) <<<" << std::endl;
    auto sft_ds = std::make_shared<sore::data::MMapDataset>(sft_path);
    sore::data::DataLoader sft_loader(sft_ds, BATCH_SIZE, SEQ_LEN);
    std::cout << "[SFT Dataset] " << sft_ds->total_tokens() << " tokens de diálogo mapeados via mmap.\n";

    // Taxa de aprendizado mais suave para SFT
    float lr_sft = 5e-4f;

    for (size_t step = 1; step <= sft_steps && sft_loader.has_next(); ++step) {
        auto batch = sft_loader.next();
        size_t N = BATCH_SIZE * SEQ_LEN;

        // Forward
        sore::Tensor x_emb = embedding.forward(batch.inputs, BATCH_SIZE, SEQ_LEN);
        sore::Tensor G = sore::functional::linear_cpu(x_emb, rnn.w_gate(), &rnn.b_gate());
        sore::Tensor A = sore::functional::sigmoid_cpu(G);
        sore::Tensor U = sore::functional::linear_cpu(x_emb, rnn.w_in(), &rnn.b_in());
        sore::Tensor H = sore::functional::linear_rnn_forward_cpu(A, U, nullptr);
        sore::Tensor Y = sore::functional::linear_cpu(H, rnn.w_out(), &rnn.b_out());
        sore::Tensor logits = sore::functional::linear_cpu(Y, w_head, &b_head);

        // Loss & Grad
        sore::Tensor d_logits({N, VOCAB_SIZE}, sore::DType::Float32, sore::Device::CPU);
        float loss = sore::nn::cross_entropy_loss_and_grad(logits, batch.targets, d_logits);

        // Backward LM Head
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

        // BPTT
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

        // AdamW com taxa de SFT
        for (size_t p_idx = 0; p_idx < all_params.size(); ++p_idx) {
            sore::optim::fused_adamw_cpu(
                *all_params[p_idx], *all_grads[p_idx], exp_avg[p_idx], exp_avg_sq[p_idx],
                lr_sft, 0.9f, 0.999f, 1e-8f, weight_decay, pretrain_steps + step
            );
            all_grads[p_idx]->zero_();
        }

        if (step % 5 == 0 || step == 1 || step == sft_steps) {
            std::cout << "[SFT Passo " << std::setw(3) << step << "/" << sft_steps 
                      << "] Loss de Diálogo: " << std::fixed << std::setprecision(4) << loss 
                      << " (PPL: " << std::exp(loss) << ")" << std::endl;
        }
    }

    save_checkpoint("checkpoints/checkpoint_chat_final.bin", all_params);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << ">>> SUCESSO: PIPELINE COMPLETO EXECUTADO E MODELO SALVO! <<<" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
