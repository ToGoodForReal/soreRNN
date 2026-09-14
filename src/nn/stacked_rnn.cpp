#include "sore/nn/stacked_rnn.hpp"
#include "sore/data/dataloader.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <filesystem>

namespace sore {
namespace nn {

StackedLinearRNNLM::StackedLinearRNNLM(StackedRNNConfig config)
    : config_(config) {
    size_t V = config_.vocab_size;
    size_t D = config_.d_model;
    size_t L = config_.num_layers;
    size_t d_mlp = config_.d_mlp;
    size_t K = config_.conv_kernel;
    Device dev = config_.device;

    // 1. Embedding (Weight Tied com o LM Head)
    w_emb_ = Tensor({V, D}, DType::Float32, dev);
    dw_emb_ = Tensor::zeros({V, D}, DType::Float32, dev);

    // 2. 12 Camadas Recorrentes com Conv1D e MLP
    layers_.resize(L);
    layer_grads_.resize(L);

    for (size_t l = 0; l < L; ++l) {
        // Bloco 1: RMSNorm1 + Conv1D + RG-LRU
        layers_[l].norm1_gamma = Tensor({D}, DType::Float32, dev);
        layers_[l].w_conv = Tensor({D, K}, DType::Float32, dev);
        layers_[l].b_conv = Tensor({D}, DType::Float32, dev);

        layers_[l].w_gate = Tensor({D, D}, DType::Float32, dev);
        layers_[l].b_gate = Tensor({D}, DType::Float32, dev);
        layers_[l].w_in = Tensor({D, D}, DType::Float32, dev);
        layers_[l].b_in = Tensor({D}, DType::Float32, dev);
        layers_[l].w_out = Tensor({D, D}, DType::Float32, dev);
        layers_[l].b_out = Tensor({D}, DType::Float32, dev);

        layer_grads_[l].dnorm1_gamma = Tensor::zeros({D}, DType::Float32, dev);
        layer_grads_[l].dw_conv = Tensor::zeros({D, K}, DType::Float32, dev);
        layer_grads_[l].db_conv = Tensor::zeros({D}, DType::Float32, dev);
        layer_grads_[l].dw_gate = Tensor::zeros({D, D}, DType::Float32, dev);
        layer_grads_[l].db_gate = Tensor::zeros({D}, DType::Float32, dev);
        layer_grads_[l].dw_in = Tensor::zeros({D, D}, DType::Float32, dev);
        layer_grads_[l].db_in = Tensor::zeros({D}, DType::Float32, dev);
        layer_grads_[l].dw_out = Tensor::zeros({D, D}, DType::Float32, dev);
        layer_grads_[l].db_out = Tensor::zeros({D}, DType::Float32, dev);

        // Bloco 2: RMSNorm2 + MLP Channel Mixing
        layers_[l].norm2_gamma = Tensor({D}, DType::Float32, dev);
        layers_[l].w_mlp1 = Tensor({d_mlp, D}, DType::Float32, dev);
        layers_[l].b_mlp1 = Tensor({d_mlp}, DType::Float32, dev);
        layers_[l].w_mlp2 = Tensor({D, d_mlp}, DType::Float32, dev);
        layers_[l].b_mlp2 = Tensor({D}, DType::Float32, dev);

        layer_grads_[l].dnorm2_gamma = Tensor::zeros({D}, DType::Float32, dev);
        layer_grads_[l].dw_mlp1 = Tensor::zeros({d_mlp, D}, DType::Float32, dev);
        layer_grads_[l].db_mlp1 = Tensor::zeros({d_mlp}, DType::Float32, dev);
        layer_grads_[l].dw_mlp2 = Tensor::zeros({D, d_mlp}, DType::Float32, dev);
        layer_grads_[l].db_mlp2 = Tensor::zeros({D}, DType::Float32, dev);
    }

    // 3. Final RMSNorm
    final_norm_gamma_ = Tensor({D}, DType::Float32, dev);
    dfinal_norm_gamma_ = Tensor::zeros({D}, DType::Float32, dev);

    // 4. LM Head Bias (w_head compartilhado com w_emb_)
    b_head_ = Tensor({V}, DType::Float32, dev);
    db_head_ = Tensor::zeros({V}, DType::Float32, dev);

    init_weights(42);
}

void StackedLinearRNNLM::init_weights(uint64_t seed) {
    std::mt19937 rng(seed);
    constexpr float stddev = 0.02f;
    std::normal_distribution<float> dist(0.0f, stddev);

    auto init_tensor = [&](Tensor& t, float scale = 1.0f) {
        Tensor cpu_t(t.shape(), DType::Float32, Device::CPU);
        float* ptr = cpu_t.data<float>();
        for (size_t i = 0; i < t.numel(); ++i) {
            ptr[i] = dist(rng) * scale;
        }
        if (t.device() == Device::CUDA) {
            t = cpu_t.cuda();
        } else {
            t = cpu_t;
        }
    };

    auto fill_val = [&](Tensor& t, float val) {
        Tensor cpu_t(t.shape(), DType::Float32, Device::CPU);
        cpu_t.fill_(val);
        if (t.device() == Device::CUDA) {
            t = cpu_t.cuda();
        } else {
            t = cpu_t;
        }
    };

    init_tensor(w_emb_);
    fill_val(b_head_, 0.0f);
    fill_val(final_norm_gamma_, 1.0f);

    float residual_scale = 1.0f / std::sqrt(4.0f * static_cast<float>(config_.num_layers));

    for (size_t l = 0; l < config_.num_layers; ++l) {
        fill_val(layers_[l].norm1_gamma, 1.0f);
        init_tensor(layers_[l].w_conv, 0.02f);
        fill_val(layers_[l].b_conv, 0.0f);

        init_tensor(layers_[l].w_gate);
        // Inicialização de Gates de Decaimento com Multi-Escala Temporal (Griffin/LRU)
        // Evita que o modelo nasça com decay=0.5 (esquecimento rápido) e cria
        // horizontes de memória de curto, médio e longo alcance (5 a 500+ tokens).
        {
            Tensor cpu_bg(layers_[l].b_gate.shape(), DType::Float32, Device::CPU);
            float* bg_ptr = cpu_bg.data<float>();
            size_t D = config_.d_model;
            for (size_t i = 0; i < D; ++i) {
                float frac = static_cast<float>(i) / static_cast<float>(D > 1 ? D - 1 : 1);
                float log_tau_min = std::log(5.0f);    // memória mínima: ~5 tokens
                float log_tau_max = std::log(500.0f);  // memória máxima: ~500 tokens
                float tau = std::exp(log_tau_min + frac * (log_tau_max - log_tau_min));
                float lambda_val = 1.0f - (1.0f / tau);
                lambda_val = std::clamp(lambda_val, 0.01f, 0.999f);
                bg_ptr[i] = std::log(lambda_val / (1.0f - lambda_val));
            }
            if (layers_[l].b_gate.device() == Device::CUDA) {
                layers_[l].b_gate = cpu_bg.cuda();
            } else {
                layers_[l].b_gate = cpu_bg;
            }
        }
        init_tensor(layers_[l].w_in);
        fill_val(layers_[l].b_in, 0.0f);
        init_tensor(layers_[l].w_out, residual_scale);
        fill_val(layers_[l].b_out, 0.0f);

        fill_val(layers_[l].norm2_gamma, 1.0f);
        init_tensor(layers_[l].w_mlp1);
        fill_val(layers_[l].b_mlp1, 0.0f);
        init_tensor(layers_[l].w_mlp2, residual_scale);
        fill_val(layers_[l].b_mlp2, 0.0f);
    }
}

Tensor StackedLinearRNNLM::forward(const std::vector<uint16_t>& tokens, size_t B, size_t T) {
    size_t D = config_.d_model;
    Tensor x = cuda::embedding_forward_cuda(tokens, w_emb_, B, T);

    cache_.resize(config_.num_layers);

    for (size_t l = 0; l < config_.num_layers; ++l) {
        // --- Sub-bloco 1: Recorrência Linear com Conv1D ---
        cache_[l].x_in1 = x.clone();
        cache_[l].x_norm1 = cuda::rmsnorm_forward_cuda(x, layers_[l].norm1_gamma, cache_[l].rstd1, config_.eps);
        
        // Causal Depthwise Conv1D (K=4)
        cache_[l].x_conv = cuda::conv1d_causal_depthwise_forward_cuda(
            cache_[l].x_norm1, layers_[l].w_conv, &layers_[l].b_conv, B, T, D
        );

        // Gate Sigmoid e Projeção de Entrada (G é temporário, apenas A e U são mantidos)
        Tensor G = cuda::linear_cuda(cache_[l].x_conv, layers_[l].w_gate, &layers_[l].b_gate);
        cache_[l].A = cuda::sigmoid_cuda(G);
        cache_[l].U = cuda::linear_cuda(cache_[l].x_conv, layers_[l].w_in, &layers_[l].b_in);

        // Recorrência Linear Associativa (Lockstep / Parallel Scan)
        cache_[l].H = cuda::linear_rnn_forward_cuda(cache_[l].A, cache_[l].U);
        Tensor Y_rnn = cuda::linear_cuda(cache_[l].H, layers_[l].w_out, &layers_[l].b_out);

        // Conexão Residual 1: x = x + Y_rnn (Y_rnn é liberado imediatamente)
        cuda::add_residual_cuda(x, Y_rnn);

        // --- Sub-bloco 2: MLP (Channel-Mixing) com Ativação GELU ---
        cache_[l].x_in2 = x.clone();
        cache_[l].x_norm2 = cuda::rmsnorm_forward_cuda(x, layers_[l].norm2_gamma, cache_[l].rstd2, config_.eps);

        cache_[l].mlp_in = cuda::linear_cuda(cache_[l].x_norm2, layers_[l].w_mlp1, &layers_[l].b_mlp1);
        cache_[l].mlp_act = cuda::gelu_cuda(cache_[l].mlp_in);
        Tensor Y_mlp = cuda::linear_cuda(cache_[l].mlp_act, layers_[l].w_mlp2, &layers_[l].b_mlp2);

        // Conexão Residual 2: x = x + Y_mlp (Y_mlp é liberado imediatamente)
        cuda::add_residual_cuda(x, Y_mlp);
    }

    // 3. RMSNorm Final
    final_norm_input_ = x.clone();
    final_norm_x_ = cuda::rmsnorm_forward_cuda(final_norm_input_, final_norm_gamma_, final_norm_rstd_, config_.eps);

    // 4. LM Head Projeção com Weight Tying (reutiliza w_emb_ como matriz de projeção)
    Tensor logits = cuda::linear_cuda(final_norm_x_, w_emb_, &b_head_);
    return logits;
}

void StackedLinearRNNLM::backward(const Tensor& d_logits, const std::vector<uint16_t>& tokens) {
    size_t B = cache_[0].x_in1.dim(0);
    size_t T = cache_[0].x_in1.dim(1);
    size_t D = config_.d_model;

    // 1. LM Head Backward com Weight Tying
    // Gradiente dW acumula DIRETAMENTE no tensor dw_emb_
    Tensor d_final_norm_x = Tensor::zeros(final_norm_x_.shape(), DType::Float32, Device::CUDA);
    cuda::linear_backward_cuda(d_logits, final_norm_x_, w_emb_, d_final_norm_x, dw_emb_, &db_head_);

    // 2. Final RMSNorm Backward
    Tensor d_x = Tensor::zeros(d_final_norm_x.shape(), DType::Float32, Device::CUDA);
    cuda::rmsnorm_backward_cuda(
        d_final_norm_x, final_norm_input_, final_norm_gamma_, final_norm_rstd_,
        d_x, dfinal_norm_gamma_
    );

    // 3. Retropropagação através das 12 camadas (Ordem Reversa)
    for (int64_t l = static_cast<int64_t>(config_.num_layers) - 1; l >= 0; --l) {
        // --- MLP Backward ---
        // Residual 2: x = x_in2 + Y_mlp => dY_mlp = d_x
        Tensor d_mlp_act = Tensor::zeros(cache_[l].mlp_act.shape(), DType::Float32, Device::CUDA);
        cuda::linear_backward_cuda(
            d_x, cache_[l].mlp_act, layers_[l].w_mlp2, d_mlp_act,
            layer_grads_[l].dw_mlp2, &layer_grads_[l].db_mlp2
        );

        Tensor d_mlp_in = cuda::gelu_backward_cuda(d_mlp_act, cache_[l].mlp_in);

        Tensor d_x_norm2 = Tensor::zeros(cache_[l].x_norm2.shape(), DType::Float32, Device::CUDA);
        cuda::linear_backward_cuda(
            d_mlp_in, cache_[l].x_norm2, layers_[l].w_mlp1, d_x_norm2,
            layer_grads_[l].dw_mlp1, &layer_grads_[l].db_mlp1
        );

        Tensor dx_in2 = Tensor::zeros(cache_[l].x_in2.shape(), DType::Float32, Device::CUDA);
        cuda::rmsnorm_backward_cuda(
            d_x_norm2, cache_[l].x_in2, layers_[l].norm2_gamma, cache_[l].rstd2,
            dx_in2, layer_grads_[l].dnorm2_gamma
        );

        // Acumula no gradiente residual de x_in2: d_x = d_x + dx_in2
        cuda::add_residual_cuda(d_x, dx_in2);

        // --- Recorrência Linear + Conv1D Backward ---
        // Residual 1: x_in2 = x_in1 + Y_rnn => dY_rnn = d_x
        Tensor dH = Tensor::zeros(cache_[l].H.shape(), DType::Float32, Device::CUDA);
        cuda::linear_backward_cuda(
            d_x, cache_[l].H, layers_[l].w_out, dH,
            layer_grads_[l].dw_out, &layer_grads_[l].db_out
        );

        Tensor dA = Tensor::zeros(cache_[l].A.shape(), DType::Float32, Device::CUDA);
        Tensor dU = Tensor::zeros(cache_[l].U.shape(), DType::Float32, Device::CUDA);
        autograd::linear_rnn_backward_cuda(dH, cache_[l].A, cache_[l].H, nullptr, dA, dU);

        Tensor dG = autograd::sigmoid_backward_cuda(dA, cache_[l].A);

        Tensor dx_conv_u = Tensor::zeros(cache_[l].x_conv.shape(), DType::Float32, Device::CUDA);
        cuda::linear_backward_cuda(
            dU, cache_[l].x_conv, layers_[l].w_in, dx_conv_u,
            layer_grads_[l].dw_in, &layer_grads_[l].db_in
        );

        Tensor dx_conv_g = Tensor::zeros(cache_[l].x_conv.shape(), DType::Float32, Device::CUDA);
        cuda::linear_backward_cuda(
            dG, cache_[l].x_conv, layers_[l].w_gate, dx_conv_g,
            layer_grads_[l].dw_gate, &layer_grads_[l].db_gate
        );

        cuda::add_residual_cuda(dx_conv_u, dx_conv_g); // dx_conv total

        // Conv1D Backward
        Tensor dx_norm1 = Tensor::zeros(cache_[l].x_norm1.shape(), DType::Float32, Device::CUDA);
        cuda::conv1d_causal_depthwise_backward_cuda(
            dx_conv_u, cache_[l].x_norm1, layers_[l].w_conv, dx_norm1,
            layer_grads_[l].dw_conv, &layer_grads_[l].db_conv, B, T, D
        );

        // RMSNorm 1 Backward
        Tensor dx_in1 = Tensor::zeros(cache_[l].x_in1.shape(), DType::Float32, Device::CUDA);
        cuda::rmsnorm_backward_cuda(
            dx_norm1, cache_[l].x_in1, layers_[l].norm1_gamma, cache_[l].rstd1,
            dx_in1, layer_grads_[l].dnorm1_gamma
        );

        // Acumula no gradiente residual de x_in1: d_x = d_x + dx_in1
        cuda::add_residual_cuda(d_x, dx_in1);
    }

    // 4. Embedding Backward (Acumula na mesma matriz dw_emb_ do LM Head!)
    cuda::embedding_backward_cuda(d_x, tokens, dw_emb_);
}

void StackedLinearRNNLM::zero_grad() {
    dw_emb_.zero_();
    dfinal_norm_gamma_.zero_();
    db_head_.zero_();

    for (size_t l = 0; l < config_.num_layers; ++l) {
        layer_grads_[l].dnorm1_gamma.zero_();
        layer_grads_[l].dw_conv.zero_();
        layer_grads_[l].db_conv.zero_();
        layer_grads_[l].dw_gate.zero_();
        layer_grads_[l].db_gate.zero_();
        layer_grads_[l].dw_in.zero_();
        layer_grads_[l].db_in.zero_();
        layer_grads_[l].dw_out.zero_();
        layer_grads_[l].db_out.zero_();

        layer_grads_[l].dnorm2_gamma.zero_();
        layer_grads_[l].dw_mlp1.zero_();
        layer_grads_[l].db_mlp1.zero_();
        layer_grads_[l].dw_mlp2.zero_();
        layer_grads_[l].db_mlp2.zero_();
    }
}

std::vector<Tensor*> StackedLinearRNNLM::parameters() {
    std::vector<Tensor*> params;
    params.push_back(&w_emb_);
    params.push_back(&b_head_);
    params.push_back(&final_norm_gamma_);

    for (size_t l = 0; l < config_.num_layers; ++l) {
        params.push_back(&layers_[l].norm1_gamma);
        params.push_back(&layers_[l].w_conv);
        params.push_back(&layers_[l].b_conv);
        params.push_back(&layers_[l].w_gate);
        params.push_back(&layers_[l].b_gate);
        params.push_back(&layers_[l].w_in);
        params.push_back(&layers_[l].b_in);
        params.push_back(&layers_[l].w_out);
        params.push_back(&layers_[l].b_out);

        params.push_back(&layers_[l].norm2_gamma);
        params.push_back(&layers_[l].w_mlp1);
        params.push_back(&layers_[l].b_mlp1);
        params.push_back(&layers_[l].w_mlp2);
        params.push_back(&layers_[l].b_mlp2);
    }
    return params;
}

std::vector<Tensor*> StackedLinearRNNLM::gradients() {
    std::vector<Tensor*> grads;
    grads.push_back(&dw_emb_);
    grads.push_back(&db_head_);
    grads.push_back(&dfinal_norm_gamma_);

    for (size_t l = 0; l < config_.num_layers; ++l) {
        grads.push_back(&layer_grads_[l].dnorm1_gamma);
        grads.push_back(&layer_grads_[l].dw_conv);
        grads.push_back(&layer_grads_[l].db_conv);
        grads.push_back(&layer_grads_[l].dw_gate);
        grads.push_back(&layer_grads_[l].db_gate);
        grads.push_back(&layer_grads_[l].dw_in);
        grads.push_back(&layer_grads_[l].db_in);
        grads.push_back(&layer_grads_[l].dw_out);
        grads.push_back(&layer_grads_[l].db_out);

        grads.push_back(&layer_grads_[l].dnorm2_gamma);
        grads.push_back(&layer_grads_[l].dw_mlp1);
        grads.push_back(&layer_grads_[l].db_mlp1);
        grads.push_back(&layer_grads_[l].dw_mlp2);
        grads.push_back(&layer_grads_[l].db_mlp2);
    }
    return grads;
}

size_t StackedLinearRNNLM::total_parameters() const noexcept {
    size_t count = w_emb_.numel() + b_head_.numel() + final_norm_gamma_.numel();
    for (size_t l = 0; l < config_.num_layers; ++l) {
        count += layers_[l].norm1_gamma.numel();
        count += layers_[l].w_conv.numel() + layers_[l].b_conv.numel();
        count += layers_[l].w_gate.numel() + layers_[l].b_gate.numel();
        count += layers_[l].w_in.numel() + layers_[l].b_in.numel();
        count += layers_[l].w_out.numel() + layers_[l].b_out.numel();

        count += layers_[l].norm2_gamma.numel();
        count += layers_[l].w_mlp1.numel() + layers_[l].b_mlp1.numel();
        count += layers_[l].w_mlp2.numel() + layers_[l].b_mlp2.numel();
    }
    return count;
}

void StackedLinearRNNLM::save_checkpoint(const std::string& filepath, const TrainingState* state) const {
    std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Não foi possível salvar checkpoint em: " + filepath);
    }

    CheckpointHeader header;
    header.magic[0] = 'S'; header.magic[1] = 'O'; header.magic[2] = 'R'; header.magic[3] = 'E';
    header.version = 2;
    header.header_size = sizeof(CheckpointHeader);
    header.vocab_size = static_cast<uint32_t>(config_.vocab_size);
    header.d_model = static_cast<uint32_t>(config_.d_model);
    header.num_layers = static_cast<uint32_t>(config_.num_layers);
    header.d_mlp = static_cast<uint32_t>(config_.d_mlp);
    header.conv_kernel = static_cast<uint32_t>(config_.conv_kernel);
    header.tie_weights = config_.tie_weights ? 1 : 0;
    header.eps = config_.eps;
    header.num_param_tensors = static_cast<uint32_t>(num_parameter_tensors());
    header.total_params = total_parameters();

    if (state != nullptr) {
        header.step = state->step;
        header.total_trained_tokens = state->total_trained_tokens;
        header.dataloader_cursor = state->dataloader_cursor;
        header.current_lr = state->current_lr;
        header.has_optimizer_state = (state->has_optimizer && !state->exp_avg.empty()) ? 1 : 0;
    }

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(CheckpointHeader));

    auto save_tensor = [&](const Tensor& t) {
        Tensor cpu_t = (t.device() == Device::CUDA) ? t.cpu() : t;
        ofs.write(reinterpret_cast<const char*>(cpu_t.data<float>()), cpu_t.numel() * sizeof(float));
    };

    save_tensor(w_emb_);
    save_tensor(b_head_);
    save_tensor(final_norm_gamma_);

    for (size_t l = 0; l < config_.num_layers; ++l) {
        save_tensor(layers_[l].norm1_gamma);
        save_tensor(layers_[l].w_conv);
        save_tensor(layers_[l].b_conv);
        save_tensor(layers_[l].w_gate);
        save_tensor(layers_[l].b_gate);
        save_tensor(layers_[l].w_in);
        save_tensor(layers_[l].b_in);
        save_tensor(layers_[l].w_out);
        save_tensor(layers_[l].b_out);

        save_tensor(layers_[l].norm2_gamma);
        save_tensor(layers_[l].w_mlp1);
        save_tensor(layers_[l].b_mlp1);
        save_tensor(layers_[l].w_mlp2);
        save_tensor(layers_[l].b_mlp2);
    }

    if (header.has_optimizer_state && state != nullptr) {
        uint32_t num_opt_tensors = static_cast<uint32_t>(state->exp_avg.size());
        ofs.write(reinterpret_cast<const char*>(&num_opt_tensors), sizeof(uint32_t));
        for (const auto& t : state->exp_avg) {
            save_tensor(t);
        }
        for (const auto& t : state->exp_avg_sq) {
            save_tensor(t);
        }
    }
}

bool StackedLinearRNNLM::load_checkpoint(const std::string& filepath, TrainingState* state) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Não foi possível abrir checkpoint em: " + filepath);
    }

    char magic[4]{0, 0, 0, 0};
    ifs.read(magic, 4);
    bool has_header = (magic[0] == 'S' && magic[1] == 'O' && magic[2] == 'R' && magic[3] == 'E');

    CheckpointHeader header;
    if (has_header) {
        ifs.seekg(0, std::ios::beg);
        ifs.read(reinterpret_cast<char*>(&header), sizeof(CheckpointHeader));

        // Validação estrita de compatibilidade de arquitetura contra o modelo instanciado
        if (header.vocab_size != config_.vocab_size ||
            header.d_model != config_.d_model ||
            header.num_layers != config_.num_layers ||
            header.d_mlp != config_.d_mlp ||
            header.conv_kernel != config_.conv_kernel) {
            std::ostringstream oss;
            oss << "[Checkpoint Error] Incompatibilidade de arquitetura ao carregar checkpoint '"
                << filepath << "':\n"
                << "  Modelo atual: vocab=" << config_.vocab_size
                << ", d_model=" << config_.d_model
                << ", num_layers=" << config_.num_layers
                << ", d_mlp=" << config_.d_mlp
                << ", conv_kernel=" << config_.conv_kernel << "\n"
                << "  Checkpoint:   vocab=" << header.vocab_size
                << ", d_model=" << header.d_model
                << ", num_layers=" << header.num_layers
                << ", d_mlp=" << header.d_mlp
                << ", conv_kernel=" << header.conv_kernel;
            throw std::runtime_error(oss.str());
        }

        std::cout << "[Checkpoint] Formato soreRNN v" << header.version
                  << " detectado (passo=" << header.step << ", tokens=" << header.total_trained_tokens
                  << ", lr=" << header.current_lr << ", otimizador=" << (header.has_optimizer_state ? "Sim" : "Não") << ")." << std::endl;

        if (state != nullptr) {
            state->step = header.step;
            state->total_trained_tokens = header.total_trained_tokens;
            state->dataloader_cursor = header.dataloader_cursor;
            state->current_lr = header.current_lr;
            state->has_optimizer = (header.has_optimizer_state != 0);
        }
        ifs.seekg(header.header_size, std::ios::beg);
    } else {
        std::cout << "[Checkpoint] Formato legado detectado (sem header binário). Carregando pesos crus..." << std::endl;
        ifs.seekg(0, std::ios::beg);
        if (state != nullptr) {
            state->step = 0;
            state->total_trained_tokens = 0;
            state->dataloader_cursor = 0;
            state->current_lr = 0.0f;
            state->has_optimizer = false;
        }
    }

    auto load_tensor = [&](Tensor& t) {
        size_t n = t.numel();
        size_t bytes = n * sizeof(float);
        if (t.device() == Device::CUDA) {
            std::vector<float> host_buf(n);
            ifs.read(reinterpret_cast<char*>(host_buf.data()), bytes);
            CUDA_CHECK(cudaMemcpy(t.raw_data(), host_buf.data(), bytes, cudaMemcpyHostToDevice));
        } else {
            ifs.read(reinterpret_cast<char*>(t.data<float>()), bytes);
        }
    };

    load_tensor(w_emb_);
    load_tensor(b_head_);
    load_tensor(final_norm_gamma_);

    for (size_t l = 0; l < config_.num_layers; ++l) {
        load_tensor(layers_[l].norm1_gamma);
        load_tensor(layers_[l].w_conv);
        load_tensor(layers_[l].b_conv);
        load_tensor(layers_[l].w_gate);
        load_tensor(layers_[l].b_gate);
        load_tensor(layers_[l].w_in);
        load_tensor(layers_[l].b_in);
        load_tensor(layers_[l].w_out);
        load_tensor(layers_[l].b_out);

        load_tensor(layers_[l].norm2_gamma);
        load_tensor(layers_[l].w_mlp1);
        load_tensor(layers_[l].b_mlp1);
        load_tensor(layers_[l].w_mlp2);
        load_tensor(layers_[l].b_mlp2);
    }

    if (has_header && header.has_optimizer_state && state != nullptr) {
        uint32_t num_opt_tensors = 0;
        ifs.read(reinterpret_cast<char*>(&num_opt_tensors), sizeof(uint32_t));
        auto params = parameters();
        if (num_opt_tensors == params.size()) {
            state->exp_avg.clear();
            state->exp_avg_sq.clear();
            for (size_t i = 0; i < num_opt_tensors; ++i) {
                Tensor t(params[i]->shape(), DType::Float32, params[i]->device());
                load_tensor(t);
                state->exp_avg.push_back(std::move(t));
            }
            for (size_t i = 0; i < num_opt_tensors; ++i) {
                Tensor t(params[i]->shape(), DType::Float32, params[i]->device());
                load_tensor(t);
                state->exp_avg_sq.push_back(std::move(t));
            }
            std::cout << " -> Momentos do AdamW (" << num_opt_tensors << " tensores m e v) restaurados com fidelidade total!" << std::endl;
        }
    }

    return has_header;
}

float StackedLinearRNNLM::evaluate(data::DataLoader& val_loader, size_t max_batches) {
    size_t batch_count = 0;
    double sum_loss = 0.0;
    size_t initial_cursor = val_loader.cursor();

    val_loader.reset();
    while (val_loader.has_next()) {
        if (max_batches > 0 && batch_count >= max_batches) break;
        auto batch = val_loader.next();
        Tensor logits = forward(batch.inputs, val_loader.batch_size(), val_loader.seq_len());
        Tensor dummy_d_logits = Tensor::zeros(logits.shape(), DType::Float32, Device::CUDA);
        float loss = cuda::cross_entropy_loss_and_grad_cuda(logits, batch.targets, dummy_d_logits);
        sum_loss += loss;
        batch_count++;
    }

    val_loader.set_cursor(initial_cursor);
    return batch_count > 0 ? static_cast<float>(sum_loss / static_cast<double>(batch_count)) : 0.0f;
}

} // namespace nn
} // namespace sore
