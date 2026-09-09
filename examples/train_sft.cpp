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
#include <fstream>
#include <string>

static float compute_lr(size_t step, size_t warmup_steps, size_t total_steps, float max_lr, float min_lr) {
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

    std::string in_ckpt = "checkpoints/sore_lm_150m_pretrain.bin";
    std::string sft_dataset = "data/sft_chat_pt.bin";
    std::string out_ckpt = "checkpoints/sore_lm_150m_sft_multitask.bin";
    size_t sft_steps = 200;
    size_t batch_size = 2;
    size_t seq_len = 256;
    std::string val_dataset = "data/sft_val.bin";

    if (argc > 1) in_ckpt = argv[1];
    if (argc > 2) sft_dataset = argv[2];
    if (argc > 3) out_ckpt = argv[3];
    if (argc > 4) sft_steps = std::stoul(argv[4]);
    if (argc > 5) batch_size = std::stoul(argv[5]);
    if (argc > 6) seq_len = std::stoul(argv[6]);
    if (argc > 7) val_dataset = argv[7];
    else if (!std::filesystem::exists(val_dataset) && std::filesystem::exists("data/sft_multitask_val.bin")) {
        val_dataset = "data/sft_multitask_val.bin";
    }

    std::cout << "=================================================================" << std::endl;
    std::cout << "   soreRNN-LM v2: TESTE DE SFT MULTI-TAREFA (CONVERSA + MATH CoT)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << " -> Checkpoint de Entrada: " << in_ckpt << std::endl;
    std::cout << " -> Dataset SFT Multi-Tarefa: " << sft_dataset << std::endl;
    std::cout << " -> Checkpoint de Saída:   " << out_ckpt << std::endl;
    std::cout << " -> Passos de Treino:      " << sft_steps << std::endl;
    std::cout << " -> Configuração Batch:    B=" << batch_size << " | T=" << seq_len 
              << " (" << batch_size * seq_len << " tokens/passo)" << std::endl;
    std::cout << " -> VRAM Estimada:         ~2.2 GB (Seguro para rodar em paralelo)" << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    if (!std::filesystem::exists(in_ckpt)) {
        std::cerr << "[Erro] Checkpoint não encontrado: " << in_ckpt << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(sft_dataset)) {
        std::cerr << "[Erro] Dataset SFT não encontrado: " << sft_dataset << std::endl;
        return 1;
    }

    // 1. Configurar Arquitetura soreRNN-LM (Auto-detecção 300M vs 150M)
    sore::nn::StackedRNNConfig config;
    config.vocab_size = 50257;
    config.conv_kernel = 4;
    config.tie_weights = true;
    config.device = sore::Device::CUDA;

    std::ifstream ifs(in_ckpt, std::ios::binary);
    char magic[4]{0};
    ifs.read(magic, 4);
    if (magic[0] == 'S' && magic[1] == 'O' && magic[2] == 'R' && magic[3] == 'E') {
        ifs.seekg(0, std::ios::beg);
        sore::nn::CheckpointHeader header;
        ifs.read(reinterpret_cast<char*>(&header), sizeof(sore::nn::CheckpointHeader));
        config.d_model = header.d_model;
        config.num_layers = header.num_layers;
        config.d_mlp = header.d_mlp;
        config.conv_kernel = header.conv_kernel;
        config.vocab_size = header.vocab_size;
        std::cout << "[1/4] Header SORE v" << header.version << " detectado no checkpoint: L=" 
                  << config.num_layers << ", D=" << config.d_model << ", MLP=" << config.d_mlp << std::endl;
    } else {
        uintmax_t ckpt_bytes = std::filesystem::file_size(in_ckpt);
        if (ckpt_bytes > 1000000000ULL) {
            config.d_model = 1280;
            config.num_layers = 18;
            config.d_mlp = 3200;
            std::cout << "[1/4] Detectada arquitetura soreRNN 300M (legado: 18 camadas, D=1280, d_mlp=3200)..." << std::endl;
        } else {
            config.d_model = 1024;
            config.num_layers = 12;
            config.d_mlp = 2560;
            std::cout << "[1/4] Detectada arquitetura soreRNN 150M (legado: 12 camadas, D=1024, d_mlp=2560)..." << std::endl;
        }
    }

    sore::nn::StackedLinearRNNLM model(config);
    model.load_checkpoint(in_ckpt);

    auto all_params = model.parameters();
    auto all_grads = model.gradients();

    // 2. Alocar tensores de AdamW
    std::cout << "[2/4] Alocando otimizador Fused AdamW..." << std::endl;
    std::vector<sore::Tensor> exp_avg;
    std::vector<sore::Tensor> exp_avg_sq;
    for (auto* p : all_params) {
        exp_avg.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }

    // 3. Carregar Dataset via MMap (Modo Pares com Máscara de Loss: inputs e targets com 65535 para prompt)
    std::cout << "[3/4] Mapeando dataset SFT multi-tarefa (com Máscara de Loss)..." << std::endl;
    auto sft_ds = std::make_shared<sore::data::MMapDataset>(sft_dataset);
    sore::data::DataLoader sft_loader(sft_ds, batch_size, seq_len, true);
    std::cout << " -> Total de tokens no dataset SFT: " << sft_ds->total_tokens() / 2 << " pares (Tokens + Máscaras)" << std::endl;

    std::shared_ptr<sore::data::DataLoader> val_loader = nullptr;
    if (std::filesystem::exists(val_dataset)) {
        auto val_ds = std::make_shared<sore::data::MMapDataset>(val_dataset);
        val_loader = std::make_shared<sore::data::DataLoader>(val_ds, batch_size, seq_len, true);
        std::cout << " -> Validação SFT Held-out ativa: " << val_ds->total_tokens() / 2 
                  << " pares (" << val_dataset << ")" << std::endl;
    }

    // 4. Loop de SFT
    std::cout << "\n[4/4] Executando Fine-Tuning Multi-Tarefa..." << std::endl;
    float max_lr = 5e-5f;
    float min_lr = 1e-5f;
    float weight_decay = 0.01f;
    size_t tokens_per_batch = batch_size * seq_len;
    size_t total_tokens_trained = 0;
    float best_val_loss = 1e9f;
    size_t patience_counter = 0;
    constexpr size_t max_patience = 4;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t step = 1; step <= sft_steps; ++step) {
        if (!sft_loader.has_next()) {
            sft_loader.reset();
        }
        auto batch = sft_loader.next();
        total_tokens_trained += tokens_per_batch;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Forward
        sore::Tensor logits = model.forward(batch.inputs, batch_size, seq_len);
        sore::Tensor d_logits = sore::Tensor::zeros(logits.shape(), sore::DType::Float32, sore::Device::CUDA);
        float loss = sore::cuda::cross_entropy_loss_and_grad_cuda(logits, batch.targets, d_logits);

        // Backward
        model.backward(d_logits, batch.inputs);

        // Clip Grads
        float grad_norm = sore::cuda::clip_grad_norm_cuda(all_grads, 1.0f);

        // AdamW
        float cur_lr = compute_lr(step, 10, sft_steps, max_lr, min_lr);
        for (size_t i = 0; i < all_params.size(); ++i) {
            sore::optim::fused_adamw_cuda(
                *all_params[i], *all_grads[i], exp_avg[i], exp_avg_sq[i],
                cur_lr, 0.9f, 0.95f, 1e-8f, weight_decay, step
            );
        }
        model.zero_grad();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (step % 10 == 0 || step == 1 || step == sft_steps) {
            float ppl = std::exp(std::min(loss, 20.0f));
            double speed = tokens_per_batch / (ms / 1000.0);
            std::cout << "[SFT Passo " << std::setw(4) << step << "/" << sft_steps 
                      << "] Loss: " << std::fixed << std::setprecision(4) << loss 
                      << " | PPL: " << std::setw(6) << std::setprecision(2) << ppl
                      << " | Norm: " << std::fixed << std::setprecision(2) << grad_norm
                      << " | LR: " << std::scientific << std::setprecision(2) << cur_lr
                      << " | " << std::fixed << std::setprecision(0) << speed << " tok/s" << std::endl;
        }

        // Validação Periódica no conjunto Held-out
        if (val_loader && (step % 20 == 0 || step == sft_steps)) {
            float val_loss = model.evaluate(*val_loader, 10);
            float val_ppl = std::exp(std::min(val_loss, 20.0f));
            std::cout << " >>> [SFT VALIDAÇÃO] Val Loss: " << std::fixed << std::setprecision(4) << val_loss
                      << " | Val PPL: " << std::setprecision(2) << val_ppl << std::endl;

            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                std::string best_out = out_ckpt + ".best.bin";
                model.save_checkpoint(best_out);
                std::cout << " -> Melhor modelo SFT salvo em: " << best_out << " (val_loss=" << val_loss << ")" << std::endl;
            } else if (step >= 50) {
                patience_counter++;
                if (patience_counter >= max_patience) {
                    std::cout << " -> Early stopping: Validação estagnou por " << max_patience 
                              << " avaliações consecutivas. Encerrando para prevenir sobreajuste." << std::endl;
                    break;
                }
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Salvar Checkpoint
    std::filesystem::create_directories("checkpoints");
    model.save_checkpoint(out_ckpt);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << ">>> SUCESSO: TESTE SFT MULTI-TAREFA CONCLUÍDO! <<<" << std::endl;
    std::cout << " -> Modelo alinhado salvo em: " << out_ckpt << std::endl;
    std::cout << " -> Tokens treinados: " << total_tokens_trained << " em " 
              << std::fixed << std::setprecision(1) << total_sec << "s" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
