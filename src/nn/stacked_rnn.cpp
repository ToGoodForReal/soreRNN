#include "sore/nn/stacked_rnn.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <stdexcept>
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
        fill_val(layers_[l].b_gate, 0.0f);
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

        // Gate Sigmoid e Projeção de Entrada
        cache_[l].G = cuda::linear_cuda(cache_[l].x_conv, layers_[l].w_gate, &layers_[l].b_gate);
        cache_[l].A = cuda::sigmoid_cuda(cache_[l].G);
        cache_[l].U = cuda::linear_cuda(cache_[l].x_conv, layers_[l].w_in, &layers_[l].b_in);

        // Recorrência Linear Associativa (Lockstep / Parallel Scan)
        cache_[l].H = cuda::linear_rnn_forward_cuda(cache_[l].A, cache_[l].U);
        cache_[l].Y_rnn = cuda::linear_cuda(cache_[l].H, layers_[l].w_out, &layers_[l].b_out);

        // Conexão Residual 1: x = x + Y_rnn
        cuda::add_residual_cuda(x, cache_[l].Y_rnn);

        // --- Sub-bloco 2: MLP (Channel-Mixing) com Ativação GELU ---
        cache_[l].x_in2 = x.clone();
        cache_[l].x_norm2 = cuda::rmsnorm_forward_cuda(x, layers_[l].norm2_gamma, cache_[l].rstd2, config_.eps);

        cache_[l].mlp_in = cuda::linear_cuda(cache_[l].x_norm2, layers_[l].w_mlp1, &layers_[l].b_mlp1);
        cache_[l].mlp_act = cuda::gelu_cuda(cache_[l].mlp_in);
        cache_[l].Y_mlp = cuda::linear_cuda(cache_[l].mlp_act, layers_[l].w_mlp2, &layers_[l].b_mlp2);

        // Conexão Residual 2: x = x + Y_mlp
        cuda::add_residual_cuda(x, cache_[l].Y_mlp);
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

void StackedLinearRNNLM::save_checkpoint(const std::string& filepath) const {
    std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Não foi possível salvar checkpoint em: " + filepath);
    }

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
}

void StackedLinearRNNLM::load_checkpoint(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Não foi possível abrir checkpoint em: " + filepath);
    }

    auto load_tensor = [&](Tensor& t) {
        Tensor cpu_t(t.shape(), DType::Float32, Device::CPU);
        ifs.read(reinterpret_cast<char*>(cpu_t.data<float>()), cpu_t.numel() * sizeof(float));
        if (t.device() == Device::CUDA) {
            t = cpu_t.cuda();
        } else {
            t = cpu_t;
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
}

} // namespace nn
} // namespace sore
