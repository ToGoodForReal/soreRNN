#include "sore/core/tensor.hpp"
#include "sore/data/dataloader.hpp"
#include "sore/nn/stacked_rnn.hpp"
#include "sore/cuda/ops.cuh"
#include "sore/optim/adamw.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>

// Agendador de Taxa de Aprendizado: Linear Warmup + Cosine Annealing
float compute_lr(size_t step, size_t warmup_steps, size_t total_steps, float max_lr, float min_lr) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step >= total_steps) return min_lr;
    float progress = static_cast<float>(step - warmup_steps) / static_cast<float>(total_steps - warmup_steps);
    constexpr float PI = 3.14159265358979323846f;
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(progress * PI));
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << std::unitbuf;
    std::cout << "=================================================================" << std::endl;
    std::cout << "      soreRNN-LM v2: TREINAMENTO DO MODELO DE LINGUAGEM 150M     " << std::endl;
    std::cout << "         (Conv1D + RG-LRU + GELU MLP + Weight Tying)            " << std::endl;
    std::cout << "             100% CUDA NATIVO (NVIDIA L40S 48GB)                " << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Configuração dos Hiperparâmetros
    std::string pretrain_path = "data/pretrain_75pt_25en.bin";
    std::string sft_path = "data/sft_chat_pt.bin";
    std::string resume_ckpt = "";

    size_t pretrain_steps = 10000;
    size_t sft_steps = 300;
    size_t batch_size = 32;
    size_t seq_len = 1024;
    float max_lr = 4e-4f;
    float min_lr = 3e-5f;
    float weight_decay = 0.01f;
    size_t save_every = 250;

    if (argc > 1) pretrain_steps = std::stoul(argv[1]);
    if (argc > 2) sft_steps = std::stoul(argv[2]);
    if (argc > 3) batch_size = std::stoul(argv[3]);
    if (argc > 4) seq_len = std::stoul(argv[4]);
    if (argc > 5) resume_ckpt = argv[5];

    std::filesystem::create_directories("checkpoints");
    std::filesystem::create_directories("logs");

    // 1. Inicialização da Arquitetura soreRNN-LM v2 (152.3M Parâmetros)
    sore::nn::StackedRNNConfig config;
    config.vocab_size = 50257;
    config.d_model = 1024;
    config.num_layers = 12;
    config.d_mlp = 2560;
    config.conv_kernel = 4;
    config.tie_weights = true;
    config.device = sore::Device::CUDA;

    std::cout << "\n[1/3] Instanciando soreRNN-LM v2 em VRAM..." << std::endl;
    sore::nn::StackedLinearRNNLM model(config);

    if (!resume_ckpt.empty() && std::filesystem::exists(resume_ckpt)) {
        std::cout << " -> Retomando treinamento a partir do checkpoint: " << resume_ckpt << std::endl;
        model.load_checkpoint(resume_ckpt);
    }

    size_t total_params = model.total_parameters();
    double vram_params_mb = static_cast<double>(total_params) * 4.0 / (1024.0 * 1024.0);
    std::cout << " -> Vocabulário: " << config.vocab_size << " tokens\n"
              << " -> Dimensão Oculta (d_model): " << config.d_model << "\n"
              << " -> Dimensão MLP (d_mlp): " << config.d_mlp << " (Expansão 2.5x com GELU)\n"
              << " -> Convolução Causal 1D: Kernel K=" << config.conv_kernel << "\n"
              << " -> Camadas Neurais: " << config.num_layers << "\n"
              << " -> Weight Tying Ativado: Sim (LM Head compartilha pesos do Embedding)\n"
              << " -> Total de Parâmetros: " << total_params << " floats (" 
              << std::fixed << std::setprecision(2) << vram_params_mb << " MB)\n"
              << " -> Dispositivo: GPU CUDA (NVIDIA Ada Lovelace L40S 48GB)" << std::endl;

    auto all_params = model.parameters();
    auto all_grads = model.gradients();

    // 2. Alocação dos Estados do Fused AdamW na GPU
    std::cout << "\n[2/3] Alocando tensores de momentos do Fused AdamW..." << std::endl;
    std::vector<sore::Tensor> exp_avg;
    std::vector<sore::Tensor> exp_avg_sq;
    for (auto* p : all_params) {
        exp_avg.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }
    std::cout << " -> Otimizador AdamW configurado para " << all_params.size() << " tensores de pesos." << std::endl;

    // 3. FASE 1: PRÉ-TREINAMENTO DE LINGUAGEM
    std::cout << "\n[3/3] >>> INICIANDO FASE 1: PRÉ-TREINAMENTO DE LINGUAGEM <<<" << std::endl;
    if (!std::filesystem::exists(pretrain_path)) {
        std::cerr << "[Erro] Arquivo de pré-treino não encontrado em: " << pretrain_path << std::endl;
        return 1;
    }

    auto pretrain_ds = std::make_shared<sore::data::MMapDataset>(pretrain_path);
    sore::data::DataLoader pretrain_loader(pretrain_ds, batch_size, seq_len);
    size_t tokens_per_batch = batch_size * seq_len;
    std::cout << " -> Dataset Pré-Treino: " << pretrain_ds->total_tokens() << " tokens carregados via mmap.\n"
              << " -> Configuração de Batch: B=" << batch_size << " | T=" << seq_len 
              << " (" << tokens_per_batch << " tokens/batch)\n"
              << " -> Alvo de Passos: " << pretrain_steps 
              << " (Total Alvo: " << (pretrain_steps * tokens_per_batch) << " tokens)" << std::endl;

    size_t warmup_steps = std::min(size_t(500), pretrain_steps / 10);
    size_t total_trained_tokens = 0;
    auto start_all = std::chrono::high_resolution_clock::now();

    for (size_t step = 1; step <= pretrain_steps; ++step) {
        if (!pretrain_loader.has_next()) {
            pretrain_loader.reset();
        }
        auto batch = pretrain_loader.next();
        total_trained_tokens += tokens_per_batch;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Forward Pass (100% GPU)
        sore::Tensor logits = model.forward(batch.inputs, batch_size, seq_len);

        // 2. Cross-Entropy Loss & Grad (100% GPU)
        sore::Tensor d_logits = sore::Tensor::zeros(logits.shape(), sore::DType::Float32, sore::Device::CUDA);
        float loss = sore::cuda::cross_entropy_loss_and_grad_cuda(logits, batch.targets, d_logits);

        // 3. Backward Pass BPTT (100% GPU)
        model.backward(d_logits, batch.inputs);

        // Gradient Clipping (Norma Máxima = 1.0f)
        float grad_norm = sore::cuda::clip_grad_norm_cuda(all_grads, 1.0f);

        // 4. Agendamento de LR & Fused AdamW Step (100% GPU)
        float cur_lr = compute_lr(step, warmup_steps, pretrain_steps, max_lr, min_lr);
        for (size_t i = 0; i < all_params.size(); ++i) {
            sore::optim::fused_adamw_cuda(
                *all_params[i], *all_grads[i], exp_avg[i], exp_avg_sq[i],
                cur_lr, 0.9f, 0.95f, 1e-8f, weight_decay, step
            );
        }
        model.zero_grad();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double tokens_per_sec = tokens_per_batch / (ms / 1000.0);

        if (step % 10 == 0 || step == 1 || step == pretrain_steps) {
            float ppl = std::exp(std::min(loss, 20.0f));
            double elapsed_sec = std::chrono::duration<double>(t1 - start_all).count();
            double remaining_sec = (elapsed_sec / step) * (pretrain_steps - step);
            int rem_hrs = static_cast<int>(remaining_sec) / 3600;
            int rem_min = (static_cast<int>(remaining_sec) % 3600) / 60;

            std::cout << "[Passo " << std::setw(5) << step << "/" << pretrain_steps 
                      << "] Loss: " << std::fixed << std::setprecision(4) << loss 
                      << " | PPL: " << std::setw(6) << std::setprecision(2) << ppl
                      << " | Norm: " << std::fixed << std::setprecision(2) << grad_norm
                      << " | LR: " << std::scientific << std::setprecision(2) << cur_lr
                      << " | " << std::fixed << std::setprecision(0) << tokens_per_sec << " tok/s"
                      << " | ETA: " << rem_hrs << "h " << rem_min << "m" << std::endl;
        }

        if (step % save_every == 0 || step == pretrain_steps) {
            std::string ckpt = "checkpoints/sore_lm_150m_pretrain.bin";
            model.save_checkpoint(ckpt);
            std::cout << " -> Checkpoint periódico salvo: " << ckpt << std::endl;
        }
    }

    // 4. FASE 2: FINE-TUNING CONVERSACIONAL (SFT CHAT PT-BR)
    std::cout << "\n>>> INICIANDO FASE 2: SFT CONVERSACIONAL (DIÁLOGOS) <<<" << std::endl;
    if (std::filesystem::exists(sft_path)) {
        auto sft_ds = std::make_shared<sore::data::MMapDataset>(sft_path);
        sore::data::DataLoader sft_loader(sft_ds, batch_size, seq_len);
        std::cout << " -> Dataset SFT: " << sft_ds->total_tokens() << " tokens de conversa.\n"
                  << " -> Alvo de Passos SFT: " << sft_steps << "\n" << std::endl;

        float sft_max_lr = 6e-5f;
        float sft_min_lr = 1e-5f;

        for (size_t step = 1; step <= sft_steps; ++step) {
            if (!sft_loader.has_next()) {
                sft_loader.reset();
            }
            auto batch = sft_loader.next();
            total_trained_tokens += tokens_per_batch;

            auto t0 = std::chrono::high_resolution_clock::now();

            sore::Tensor logits = model.forward(batch.inputs, batch_size, seq_len);
            sore::Tensor d_logits = sore::Tensor::zeros(logits.shape(), sore::DType::Float32, sore::Device::CUDA);
            float loss = sore::cuda::cross_entropy_loss_and_grad_cuda(logits, batch.targets, d_logits);

            model.backward(d_logits, batch.inputs);

            // Gradient Clipping
            float grad_norm = sore::cuda::clip_grad_norm_cuda(all_grads, 1.0f);

            float cur_lr = compute_lr(step, 10, sft_steps, sft_max_lr, sft_min_lr);
            for (size_t i = 0; i < all_params.size(); ++i) {
                sore::optim::fused_adamw_cuda(
                    *all_params[i], *all_grads[i], exp_avg[i], exp_avg_sq[i],
                    cur_lr, 0.9f, 0.95f, 1e-8f, weight_decay, pretrain_steps + step
                );
            }
            model.zero_grad();

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (step % 5 == 0 || step == 1 || step == sft_steps) {
                float ppl = std::exp(std::min(loss, 20.0f));
                std::cout << "[SFT Chat Passo " << std::setw(4) << step << "/" << sft_steps 
                          << "] Loss Diálogo: " << std::fixed << std::setprecision(4) << loss 
                          << " | PPL: " << std::setw(6) << std::setprecision(2) << ppl
                          << " | Norm: " << std::fixed << std::setprecision(2) << grad_norm
                          << " | Tempo: " << std::fixed << std::setprecision(1) << ms << " ms" << std::endl;
            }

            // Early stopping para prevenir colapso de perplexidade (overfitting catastrófico)
            if (loss < 0.6f) {
                std::cout << " -> Loss de SFT ideal atingida (" << loss << "). Parando para evitar memorização excessiva." << std::endl;
                break;
            }
        }
    }

    std::string final_ckpt = "checkpoints/sore_lm_150m_final.bin";
    model.save_checkpoint(final_ckpt);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << ">>> SUCESSO: MODELO soreRNN-LM v2 (152.3M) TREINADO E SALVO! <<<" << std::endl;
    std::cout << " -> Arquivo final: " << final_ckpt << std::endl;
    std::cout << " -> Total de tokens processados no treino: " << total_trained_tokens << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
