#include "sore/core/tensor.hpp"
#include "sore/data/dataloader.hpp"
#include "sore/nn/stacked_rnn.hpp"
#include "sore/cuda/ops.cuh"
#include "sore/cuda/caching_allocator.hpp"
#include "sore/optim/adamw.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>

// Agendador de Taxa de Aprendizado: Linear Warmup + Cosine Annealing
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
    std::cout << std::unitbuf;

    std::cout << "=================================================================" << std::endl;
    std::cout << "      soreRNN-LM v2: TREINAMENTO DO MODELO DE LINGUAGEM 300M     " << std::endl;
    std::cout << "        (18 Camadas | D=1280 | MLP=3200 | RG-LRU + TF32)         " << std::endl;
    std::cout << "            GRADIENT ACCUMULATION (OTIMIZADO PARA 8GB VRAM)      " << std::endl;
    std::cout << "=================================================================" << std::endl;

    // 0. Diagnóstico de Hardware da GPU
    sore::cuda::print_gpu_info();
    auto gpu_info = sore::cuda::get_gpu_device_info();

    // Hiperparâmetros configuráveis
    std::string pretrain_path = "data/pretrain_interleaved.bin";
    if (!std::filesystem::exists(pretrain_path)) {
        pretrain_path = "data/pretrain_75pt_25en.bin";
    }
    std::string val_path = "data/val.bin";
    std::string resume_ckpt = "";

    size_t pretrain_steps = 15000;
    size_t micro_batch_size = 2;
    size_t grad_accum_steps = 16; // Lote efetivo = 2 * 16 = 32 (16,384 tokens/passo otimizador)
    size_t seq_len = 512;        // 512 tokens (rápido e estável em 8GB)
    float max_lr = 3e-4f;
    float min_lr = 3e-5f;
    float weight_decay = 0.01f;
    size_t save_every = 500;
    size_t eval_every = 100;

    if (argc > 1) pretrain_steps = std::stoul(argv[1]);
    if (argc > 2) micro_batch_size = std::stoul(argv[2]);
    if (argc > 3) grad_accum_steps = std::stoul(argv[3]);
    if (argc > 4) seq_len = std::stoul(argv[4]);
    if (argc > 5) resume_ckpt = argv[5];

    size_t effective_batch_size = micro_batch_size * grad_accum_steps;
    size_t tokens_per_step = effective_batch_size * seq_len;

    if (gpu_info.total_memory_mb > 0 && gpu_info.total_memory_mb <= 9000 && micro_batch_size > 4) {
        std::cout << "\n[Aviso VRAM] GPU com " << gpu_info.total_memory_mb << " MB VRAM detectada. "
                  << "O micro_batch=" << micro_batch_size << " para um modelo de 300M pode atingir o limite da VRAM. "
                  << "Recomendado: micro_batch_size=2 com grad_accum_steps=4 ou 8.\n" << std::endl;
    }

    std::filesystem::create_directories("checkpoints");
    std::filesystem::create_directories("logs");

    // 1. Instanciar Arquitetura soreRNN-LM 300M (Width-Optimized)
    sore::nn::StackedRNNConfig config;
    config.vocab_size = 50257;
    config.d_model = 1280;
    config.num_layers = 18;
    config.d_mlp = 3200; // Expansão 2.5x
    config.conv_kernel = 4;
    config.tie_weights = true;
    config.device = sore::Device::CUDA;

    std::cout << "\n[1/3] Instanciando soreRNN-LM 300M na VRAM..." << std::endl;
    sore::nn::StackedLinearRNNLM model(config);

    sore::nn::TrainingState resume_state;
    size_t start_step = 1;
    size_t total_trained_tokens = 0;

    if (!resume_ckpt.empty() && std::filesystem::exists(resume_ckpt)) {
        std::cout << " -> Retomando treinamento do checkpoint: " << resume_ckpt << std::endl;
        bool has_header = model.load_checkpoint(resume_ckpt, &resume_state);
        if (has_header && resume_state.step > 0) {
            start_step = resume_state.step + 1;
            total_trained_tokens = resume_state.total_trained_tokens;
            std::cout << " -> Passo inicial: " << start_step 
                      << " | Tokens acumulados: " << total_trained_tokens << std::endl;
        }
    }

    size_t total_params = model.total_parameters();
    double vram_params_mb = static_cast<double>(total_params) * 4.0 / (1024.0 * 1024.0);
    std::cout << " -> Vocabulário: " << config.vocab_size << " tokens\n"
              << " -> Dimensão Oculta (d_model): " << config.d_model << "\n"
              << " -> Dimensão MLP (d_mlp): " << config.d_mlp << "\n"
              << " -> Camadas Neurais: " << config.num_layers << "\n"
              << " -> Weight Tying: Sim (Economia de 51.5M parâmetros na projeção final)\n"
              << " -> Total de Parâmetros: " << total_params << " floats (" 
              << std::fixed << std::setprecision(2) << vram_params_mb << " MB em VRAM)\n"
              << " -> Dispositivo: GPU CUDA (" << gpu_info.name << " | sm_" << gpu_info.major << gpu_info.minor << ")" << std::endl;

    auto all_params = model.parameters();
    auto all_grads = model.gradients();

    // 2. Alocar tensores de momentos do Fused AdamW na GPU
    std::cout << "\n[2/3] Alocando momentos do Fused AdamW (" << all_params.size() << " tensores)..." << std::endl;
    std::vector<sore::Tensor> exp_avg;
    std::vector<sore::Tensor> exp_avg_sq;

    if (resume_state.has_optimizer && resume_state.exp_avg.size() == all_params.size()) {
        exp_avg = std::move(resume_state.exp_avg);
        exp_avg_sq = std::move(resume_state.exp_avg_sq);
        std::cout << " -> Momentos do AdamW restaurados com sucesso do checkpoint!" << std::endl;
    } else {
        for (auto* p : all_params) {
            exp_avg.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
            exp_avg_sq.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
        }
    }
    double vram_optim_mb = vram_params_mb * 2.0; // exp_avg + exp_avg_sq
    std::cout << " -> Memória de Parâmetros + Otimizador: ~" 
              << std::fixed << std::setprecision(1) << (vram_params_mb + vram_optim_mb) << " MB VRAM." << std::endl;

    // 3. Pré-Treinamento de Linguagem com Gradient Accumulation
    std::cout << "\n[3/3] >>> INICIANDO TREINAMENTO DO MODELO 300M <<<" << std::endl;
    if (!std::filesystem::exists(pretrain_path)) {
        std::cerr << "[Aviso] Dataset de pré-treino não encontrado em: " << pretrain_path << std::endl;
        std::cerr << "Execute 'python3 scripts/prepare_4b_pretrain.py --out data/pretrain_interleaved.bin --val_out data/val.bin' primeiro." << std::endl;
        return 1;
    }

    auto pretrain_ds = std::make_shared<sore::data::MMapDataset>(pretrain_path);
    bool pretrain_paired = false;
    if (pretrain_ds->total_tokens() >= 200) {
        const uint16_t* ptr = pretrain_ds->token_data();
        for (size_t i = 1; i < std::min(size_t(2000), pretrain_ds->total_tokens()); i += 2) {
            if (ptr[i] == 65535) {
                pretrain_paired = true;
                break;
            }
        }
    }
    sore::data::DataLoader pretrain_loader(pretrain_ds, micro_batch_size, seq_len, pretrain_paired);
    if (resume_state.dataloader_cursor > 0) {
        pretrain_loader.set_cursor(resume_state.dataloader_cursor);
        std::cout << " -> Cursor do DataLoader restaurado: token offset " << resume_state.dataloader_cursor << std::endl;
    }

    // Dataset de validação held-out opcional
    std::shared_ptr<sore::data::DataLoader> val_loader = nullptr;
    if (std::filesystem::exists(val_path)) {
        auto val_ds = std::make_shared<sore::data::MMapDataset>(val_path);
        bool val_paired = false;
        if (val_ds->total_tokens() >= 200) {
            const uint16_t* vptr = val_ds->token_data();
            for (size_t i = 1; i < std::min(size_t(2000), val_ds->total_tokens()); i += 2) {
                if (vptr[i] == 65535) {
                    val_paired = true;
                    break;
                }
            }
        }
        val_loader = std::make_shared<sore::data::DataLoader>(val_ds, micro_batch_size, seq_len, val_paired);
        std::cout << " -> Validação Held-out ativa: " 
                  << (val_paired ? val_ds->total_tokens() / 2 : val_ds->total_tokens()) 
                  << " tokens (" << val_path << ", paired=" << (val_paired ? "Sim" : "Não") << ")" << std::endl;
    }

    std::cout << " -> Dataset: " << pretrain_path << " (" 
              << (pretrain_paired ? pretrain_ds->total_tokens() / 2 : pretrain_ds->total_tokens()) 
              << " tokens úteis, paired=" << (pretrain_paired ? "Sim" : "Não") << ")\n"
              << " -> Configuração de Execução:\n"
              << "      - Micro-Batch (GPU): " << micro_batch_size << " sequências\n"
              << "      - Passos de Acumulação: " << grad_accum_steps << "\n"
              << "      - Batch Efetivo Total: " << effective_batch_size << " sequências (" 
              << tokens_per_step << " tokens/passo otimizador)\n"
              << "      - Janela de Contexto: " << seq_len << " tokens\n"
              << "      - Passos Alvo: " << pretrain_steps 
              << " (Total: " << (pretrain_steps * tokens_per_step / 1000000.0) << "M tokens)\n"
              << "      - LR Máx / Mín: " << max_lr << " / " << min_lr << "\n" << std::endl;

    size_t warmup_steps = std::min(size_t(500), pretrain_steps / 10);
    float best_val_loss = 1e9f;
    auto start_all = std::chrono::high_resolution_clock::now();

    sore::cuda::refresh_bf16_weights(all_params); // inicializa espelhos BF16 dos pesos
    for (size_t step = start_step; step <= pretrain_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        float accum_loss = 0.0f;

        // Sub-loop de Acumulação de Gradientes
        for (size_t micro = 0; micro < grad_accum_steps; ++micro) {
            if (!pretrain_loader.has_next()) {
                pretrain_loader.reset();
            }
            auto batch = pretrain_loader.next();
            total_trained_tokens += (micro_batch_size * seq_len);

            // 1. Forward Pass (Assíncrono, gerenciado pelo Caching Allocator)
            sore::Tensor logits = model.forward(batch.inputs, micro_batch_size, seq_len);

            // 2. Cross-Entropy Loss & d_logits
            sore::Tensor d_logits = sore::Tensor::zeros(logits.shape(), sore::DType::Float32, sore::Device::CUDA);
            float loss = sore::cuda::cross_entropy_loss_and_grad_cuda(logits, batch.targets, d_logits);
            accum_loss += loss;

            // Escalonar gradiente de logits pelo fator de acumulação (1.0 / grad_accum_steps)
            if (grad_accum_steps > 1) {
                sore::cuda::scale_tensor_cuda(d_logits, 1.0f / static_cast<float>(grad_accum_steps));
            }

            // 3. Backward Pass BPTT (Acumula gradientes em layer_grads_)
            model.backward(d_logits, batch.inputs);
        }

        accum_loss /= static_cast<float>(grad_accum_steps);

        // 4. Gradient Clipping (Norma Máxima com acumulador double)
        float grad_norm = sore::cuda::clip_grad_norm_cuda(all_grads, 1.0f);

        // 5. Agendamento de LR & Fused AdamW Step (100% GPU)
        float cur_lr = compute_lr(step, warmup_steps, pretrain_steps, max_lr, min_lr);
        for (size_t i = 0; i < all_params.size(); ++i) {
            sore::optim::fused_adamw_cuda(
                *all_params[i], *all_grads[i], exp_avg[i], exp_avg_sq[i],
                cur_lr, 0.9f, 0.95f, 1e-8f, weight_decay, step
            );
        }
        model.zero_grad();
        sore::cuda::refresh_bf16_weights(all_params); // re-cast espelhos BF16 apos o update

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double tokens_per_sec = tokens_per_step / (ms / 1000.0);

        if (step % 10 == 0 || step == 1 || step == pretrain_steps) {
            float ppl = std::exp(std::min(accum_loss, 20.0f));
            double elapsed_sec = std::chrono::duration<double>(t1 - start_all).count();
            double remaining_sec = (elapsed_sec / (step - start_step + 1)) * (pretrain_steps - step);
            int rem_hrs = static_cast<int>(remaining_sec) / 3600;
            int rem_min = (static_cast<int>(remaining_sec) % 3600) / 60;

            auto mem_stats = sore::cuda::CUDACachingAllocator::instance().stats();
            double peak_mb = mem_stats.peak_allocated_bytes / (1024.0 * 1024.0);

            std::cout << "[300M Passo " << std::setw(5) << step << "/" << pretrain_steps 
                      << "] Loss: " << std::fixed << std::setprecision(4) << accum_loss 
                      << " | PPL: " << std::setw(6) << std::setprecision(2) << ppl
                      << " | Norm: " << std::fixed << std::setprecision(2) << grad_norm
                      << " | LR: " << std::scientific << std::setprecision(2) << cur_lr
                      << " | " << std::fixed << std::setprecision(0) << tokens_per_sec << " tok/s"
                      << " | VRAM: " << std::fixed << std::setprecision(0) << peak_mb << " MB"
                      << " | Total: " << (total_trained_tokens / 1000) << "k tok"
                      << " | ETA: " << rem_hrs << "h " << rem_min << "m" << std::endl;
        }

        // Validação Periódica (Held-Out)
        if (val_loader && (step % eval_every == 0 || step == pretrain_steps)) {
            float val_loss = model.evaluate(*val_loader, 20);
            float val_ppl = std::exp(std::min(val_loss, 20.0f));
            std::cout << " >>> [VALIDAÇÃO 300M] Val Loss: " << std::fixed << std::setprecision(4) << val_loss
                      << " | Val PPL: " << std::setprecision(2) << val_ppl << std::endl;

            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                std::string best_ckpt = "checkpoints/sore_lm_300m_best.bin";
                sore::nn::TrainingState state;
                state.step = step;
                state.total_trained_tokens = total_trained_tokens;
                state.dataloader_cursor = pretrain_loader.cursor();
                state.current_lr = cur_lr;
                state.has_optimizer = true;
                state.exp_avg = exp_avg;
                state.exp_avg_sq = exp_avg_sq;
                model.save_checkpoint(best_ckpt, &state);
                std::cout << " -> Melhor modelo 300M salvo em: " << best_ckpt << std::endl;
            }
        }

        if (step % save_every == 0 || step == pretrain_steps) {
            std::string ckpt = "checkpoints/sore_lm_300m_pretrain.bin";
            sore::nn::TrainingState state;
            state.step = step;
            state.total_trained_tokens = total_trained_tokens;
            state.dataloader_cursor = pretrain_loader.cursor();
            state.current_lr = cur_lr;
            state.has_optimizer = true;
            state.exp_avg = exp_avg;
            state.exp_avg_sq = exp_avg_sq;
            model.save_checkpoint(ckpt, &state);
            std::cout << " -> Checkpoint estruturado (v2) salvo em: " << ckpt << std::endl;
        }
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << " Treinamento soreRNN-LM 300M concluído com sucesso!" << std::endl;
    std::string final_ckpt = "checkpoints/sore_lm_300m_final.bin";
    sore::nn::TrainingState final_state;
    final_state.step = pretrain_steps;
    final_state.total_trained_tokens = total_trained_tokens;
    final_state.has_optimizer = true;
    final_state.exp_avg = exp_avg;
    final_state.exp_avg_sq = exp_avg_sq;
    model.save_checkpoint(final_ckpt, &final_state);
    std::cout << " Modelo final salvo em: " << final_ckpt << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
