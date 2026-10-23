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
#include <vector>

#pragma pack(push, 1)
struct DPOHeader {
    char magic[4]{'D', 'P', 'O', '1'};
    uint32_t seq_len{256};
    uint32_t num_pairs{0};
    uint32_t vocab_size{50257};
    uint8_t reserved[16]{0};
};
#pragma pack(pop)

struct DPOPair {
    std::vector<uint16_t> chosen_inputs;
    std::vector<uint16_t> chosen_targets;
    std::vector<uint16_t> rejected_inputs;
    std::vector<uint16_t> rejected_targets;
    float ref_logp_w{0.0f};
    float ref_logp_l{0.0f};
};

class DPODatasetReader {
public:
    explicit DPODatasetReader(const std::string& path) {
        ifs_.open(path, std::ios::binary);
        if (!ifs_.is_open()) {
            throw std::runtime_error("Não foi possível abrir dataset DPO em: " + path);
        }
        ifs_.read(reinterpret_cast<char*>(&header_), sizeof(DPOHeader));
        if (header_.magic[0] != 'D' || header_.magic[1] != 'P' || header_.magic[2] != 'O') {
            throw std::runtime_error("Header DPO inválido em: " + path);
        }
        seq_len_ = header_.seq_len;
        num_pairs_ = header_.num_pairs;
        bytes_per_sample_ = 4 * seq_len_ * sizeof(uint16_t) + 2 * sizeof(float);
    }

    size_t num_pairs() const noexcept { return num_pairs_; }
    size_t seq_len() const noexcept { return seq_len_; }

    DPOPair read_sample(size_t index) {
        index = index % num_pairs_;
        size_t offset = sizeof(DPOHeader) + index * bytes_per_sample_;
        ifs_.seekg(offset, std::ios::beg);

        DPOPair pair;
        pair.chosen_inputs.resize(seq_len_);
        pair.chosen_targets.resize(seq_len_);
        pair.rejected_inputs.resize(seq_len_);
        pair.rejected_targets.resize(seq_len_);

        ifs_.read(reinterpret_cast<char*>(pair.chosen_inputs.data()), seq_len_ * sizeof(uint16_t));
        ifs_.read(reinterpret_cast<char*>(pair.chosen_targets.data()), seq_len_ * sizeof(uint16_t));
        ifs_.read(reinterpret_cast<char*>(pair.rejected_inputs.data()), seq_len_ * sizeof(uint16_t));
        ifs_.read(reinterpret_cast<char*>(pair.rejected_targets.data()), seq_len_ * sizeof(uint16_t));
        ifs_.read(reinterpret_cast<char*>(&pair.ref_logp_w), sizeof(float));
        ifs_.read(reinterpret_cast<char*>(&pair.ref_logp_l), sizeof(float));

        return pair;
    }

private:
    std::ifstream ifs_;
    DPOHeader header_;
    size_t seq_len_{256};
    size_t num_pairs_{0};
    size_t bytes_per_sample_{0};
};

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

    std::string in_ckpt = "checkpoints/sore_lm_150m_sft.bin";
    std::string dpo_path = "data/dpo_pairs.bin";
    std::string out_ckpt = "checkpoints/sore_lm_150m_dpo.bin";
    size_t dpo_steps = 300;
    float beta = 0.1f; // Temperatura de preferência DPO
    float max_lr = 1e-6f;
    float min_lr = 2e-7f;

    if (argc > 1) in_ckpt = argv[1];
    if (argc > 2) dpo_path = argv[2];
    if (argc > 3) out_ckpt = argv[3];
    if (argc > 4) dpo_steps = std::stoul(argv[4]);
    if (argc > 5) beta = std::stof(argv[5]);

    std::cout << "=================================================================" << std::endl;
    std::cout << "     soreRNN-LM: ALINHAMENTO POR PREFERÊNCIA DIRETA (DPO) NATIVO " << std::endl;
    std::cout << "               TREINADOR ZERO-OOM (100% GPU / C++20)             " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << " -> Checkpoint de Entrada (SFT): " << in_ckpt << std::endl;
    std::cout << " -> Dataset DPO:                 " << dpo_path << std::endl;
    std::cout << " -> Checkpoint de Saída:         " << out_ckpt << std::endl;
    std::cout << " -> Passos de Alinhamento:       " << dpo_steps << std::endl;
    std::cout << " -> Parâmetro Beta:              " << beta << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    if (!std::filesystem::exists(in_ckpt)) {
        std::cerr << "[Erro] Checkpoint SFT não encontrado: " << in_ckpt << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(dpo_path)) {
        std::cerr << "[Erro] Dataset de preferências DPO não encontrado: " << dpo_path << std::endl;
        std::cerr << "Execute 'python3 scripts/prepare_dpo_dataset.py' primeiro." << std::endl;
        return 1;
    }

    DPODatasetReader reader(dpo_path);
    size_t seq_len = reader.seq_len();
    std::cout << " -> Dataset DPO carregado: " << reader.num_pairs() 
              << " pares [Escolhido, Rejeitado], SeqLen=" << seq_len << std::endl;

    // 1. Carregar Modelo SFT
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
    } else {
        uintmax_t sz = std::filesystem::file_size(in_ckpt);
        if (sz > 1000000000ULL) {
            config.d_model = 1280; config.num_layers = 18; config.d_mlp = 3200;
        } else {
            config.d_model = 1024; config.num_layers = 12; config.d_mlp = 2560;
        }
    }

    sore::nn::StackedLinearRNNLM model(config);
    model.load_checkpoint(in_ckpt);

    auto all_params = model.parameters();
    auto all_grads = model.gradients();

    // 2. Alocar Tensores do Fused AdamW
    std::vector<sore::Tensor> exp_avg;
    std::vector<sore::Tensor> exp_avg_sq;
    for (auto* p : all_params) {
        exp_avg.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
        exp_avg_sq.push_back(sore::Tensor::zeros(p->shape(), p->dtype(), p->device()));
    }

    // 3. Loop de Treino DPO
    std::cout << "\n>>> INICIANDO OTIMIZAÇÃO POR PREFERÊNCIA DIRETA (DPO) <<<" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    size_t sample_idx = 0;
    float rolling_loss = 0.0f;
    float rolling_margin = 0.0f;
    size_t rolling_correct = 0;

    for (size_t step = 1; step <= dpo_steps; ++step) {
        DPOPair pair = reader.read_sample(sample_idx++);

        // Conta tokens ativos (não mascarados) em chosen e rejected
        size_t n_active_w = 0;
        for (auto t : pair.chosen_targets) if (t < config.vocab_size) n_active_w++;
        size_t n_active_l = 0;
        for (auto t : pair.rejected_targets) if (t < config.vocab_size) n_active_l++;

        if (n_active_w == 0 || n_active_l == 0) continue;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Forward Chosen (yw)
        sore::Tensor logits_w = model.forward(pair.chosen_inputs, 1, seq_len);
        sore::Tensor d_logits_w = sore::Tensor::zeros(logits_w.shape(), sore::DType::Float32, sore::Device::CUDA);
        float loss_w = sore::cuda::cross_entropy_loss_and_grad_cuda(logits_w, pair.chosen_targets, d_logits_w);
        float pi_logp_w = -loss_w * static_cast<float>(n_active_w);

        // 2. Forward Rejected (yl)
        sore::Tensor logits_l = model.forward(pair.rejected_inputs, 1, seq_len);
        sore::Tensor d_logits_l = sore::Tensor::zeros(logits_l.shape(), sore::DType::Float32, sore::Device::CUDA);
        float loss_l = sore::cuda::cross_entropy_loss_and_grad_cuda(logits_l, pair.rejected_targets, d_logits_l);
        float pi_logp_l = -loss_l * static_cast<float>(n_active_l);

        // 3. Cálculo da Margem Implícita DPO:
        //    h_w = pi_logp_w - ref_logp_w
        //    h_l = pi_logp_l - ref_logp_l
        //    margin = beta * (h_w - h_l)
        float h_w = pi_logp_w - pair.ref_logp_w;
        float h_l = pi_logp_l - pair.ref_logp_l;
        float margin = beta * (h_w - h_l);

        // Perda DPO: L = -log(sigmoid(margin)) = log(1 + exp(-margin))
        float dpo_loss = std::log1p(std::exp(-margin));
        // Sigmoid(-margin): fator de ponderação do gradiente
        float sig_neg = 1.0f / (1.0f + std::exp(std::min(std::max(margin, -20.0f), 20.0f)));

        // Escala analítica dos gradientes:
        // dL / dLogits_w = -beta * sig_neg * d_logits_w
        // dL / dLogits_l = +beta * sig_neg * d_logits_l
        float scale_w = -beta * sig_neg;
        float scale_l = +beta * sig_neg;

        sore::cuda::scale_tensor_cuda(d_logits_w, scale_w);
        sore::cuda::scale_tensor_cuda(d_logits_l, scale_l);

        // 4. Backward em ambos os ramos
        model.backward(d_logits_w, pair.chosen_inputs);
        model.backward(d_logits_l, pair.rejected_inputs);

        // 5. Gradient Clipping
        float grad_norm = sore::cuda::clip_grad_norm_cuda(all_grads, 1.0f);

        // 6. Passo AdamW
        float cur_lr = compute_lr(step, 10, dpo_steps, max_lr, min_lr);
        for (size_t i = 0; i < all_params.size(); ++i) {
            sore::optim::fused_adamw_cuda(
                *all_params[i], *all_grads[i], exp_avg[i], exp_avg_sq[i],
                cur_lr, 0.9f, 0.95f, 1e-8f, 0.01f, step
            );
        }
        model.zero_grad();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        rolling_loss += dpo_loss;
        rolling_margin += margin;
        if (margin > 0.0f) rolling_correct++;

        if (step % 10 == 0 || step == 1 || step == dpo_steps) {
            size_t win_count = (step % 10 == 0) ? 10 : (step == 1 ? 1 : (step % 10));
            float avg_loss = rolling_loss / static_cast<float>(win_count);
            float avg_margin = rolling_margin / static_cast<float>(win_count);
            float acc = static_cast<float>(rolling_correct) / static_cast<float>(win_count) * 100.0f;

            std::cout << "[DPO Passo " << std::setw(4) << step << "/" << dpo_steps 
                      << "] Loss: " << std::fixed << std::setprecision(4) << avg_loss 
                      << " | Margem: " << std::showpos << std::fixed << std::setprecision(2) << avg_margin << std::noshowpos
                      << " | Acc: " << std::fixed << std::setprecision(1) << acc << "%"
                      << " | Norm: " << std::fixed << std::setprecision(2) << grad_norm
                      << " | LR: " << std::scientific << std::setprecision(2) << cur_lr
                      << " | " << std::fixed << std::setprecision(1) << ms << " ms/par" << std::endl;

            rolling_loss = 0.0f;
            rolling_margin = 0.0f;
            rolling_correct = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Salvar Modelo Alinhado
    std::filesystem::create_directories("checkpoints");
    model.save_checkpoint(out_ckpt);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << ">>> SUCESSO: ALINHAMENTO DPO NATIVO CONCLUÍDO! <<<" << std::endl;
    std::cout << " -> Modelo alinhado salvo em: " << out_ckpt << std::endl;
    std::cout << " -> Tempo total de alinhamento: " << std::fixed << std::setprecision(1) << total_sec << "s" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
