#pragma once

#include "sore/core/tensor.hpp"
#include "sore/cuda/ops.cuh"
#include "sore/cuda/scan.cuh"
#include "sore/nn/autograd.hpp"
#include <vector>
#include <memory>
#include <string>

namespace sore {
namespace nn {

struct StackedRNNConfig {
    size_t vocab_size{50257};
    size_t d_model{1024};
    size_t num_layers{12};
    size_t d_mlp{2560};
    size_t conv_kernel{4};
    bool tie_weights{true};
    Device device{Device::CUDA};
    float eps{1e-5f};
};

struct LayerWeights {
    // 1. RMSNorm 1
    Tensor norm1_gamma;
    // 2. Causal Depthwise Conv1D
    Tensor w_conv; // [D, 4]
    Tensor b_conv; // [D]
    // 3. LinearRNN (RG-LRU)
    Tensor w_gate; // [D, D]
    Tensor b_gate; // [D]
    Tensor w_in;   // [D, D]
    Tensor b_in;   // [D]
    Tensor w_out;  // [D, D]
    Tensor b_out;  // [D]
    // 4. RMSNorm 2
    Tensor norm2_gamma;
    // 5. MLP (Feed-Forward)
    Tensor w_mlp1; // [d_mlp, D]
    Tensor b_mlp1; // [d_mlp]
    Tensor w_mlp2; // [D, d_mlp]
    Tensor b_mlp2; // [D]
};

struct LayerGrads {
    Tensor dnorm1_gamma;
    Tensor dw_conv;
    Tensor db_conv;
    Tensor dw_gate;
    Tensor db_gate;
    Tensor dw_in;
    Tensor db_in;
    Tensor dw_out;
    Tensor db_out;
    Tensor dnorm2_gamma;
    Tensor dw_mlp1;
    Tensor db_mlp1;
    Tensor dw_mlp2;
    Tensor db_mlp2;
};

// Cache de ativações para a passagem reversa (Backward BPTT)
struct LayerActivations {
    Tensor x_in1;       // [B, T, D] entrada do bloco de recorrência
    Tensor x_norm1;     // [B, T, D]
    Tensor rstd1;       // [B*T]
    Tensor x_conv;      // [B, T, D]
    Tensor G;           // [B, T, D]
    Tensor A;           // [B, T, D]
    Tensor U;           // [B, T, D]
    Tensor H;           // [B, T, D]
    Tensor Y_rnn;       // [B, T, D]
    Tensor x_in2;       // [B, T, D] entrada do bloco MLP (x_in1 + Y_rnn)
    Tensor x_norm2;     // [B, T, D]
    Tensor rstd2;       // [B*T]
    Tensor mlp_in;      // [B*T, d_mlp]
    Tensor mlp_act;     // [B*T, d_mlp]
    Tensor Y_mlp;       // [B, T, D]
};

/**
 * @brief Language Model Recorrente Linear Profundo (152.3M Parâmetros).
 * 
 * Implementa 12 camadas ricas com:
 * - Causal Depthwise Conv1D (K=4)
 * - Recorrência Linear Real-Gated (RG-LRU)
 * - Bloco MLP (Channel-Mixing) com ativação GELU
 * - Conexões Residuais Duplas por camada
 * - Weight Tying entre Embedding e LM Head
 */
class StackedLinearRNNLM {
public:
    explicit StackedLinearRNNLM(StackedRNNConfig config = {});

    // Forward pass de treinamento (guarda ativações para backward)
    Tensor forward(const std::vector<uint16_t>& tokens, size_t B, size_t T);

    // Backward pass: calcula gradientes analíticos de ponta a ponta na GPU
    void backward(const Tensor& d_logits, const std::vector<uint16_t>& tokens);

    // Passo de inferência autoregressiva O(1) por token
    // h_states: vetor com os 12 tensores de estado oculto [1, D]
    // conv_buffers: vetor com os 12 buffers de contexto temporal [3, D]
    Tensor step(
        uint16_t token,
        std::vector<Tensor>& h_states,
        std::vector<Tensor>& conv_buffers
    );

    // Inicialização de pesos com desvios padrão calibrados
    void init_weights(uint64_t seed = 42);

    // Zera todos os tensores de gradientes acumulados
    void zero_grad();

    // Coletores de ponteiros para o otimizador AdamW
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    // Salvamento e carregamento de checkpoints binários
    void save_checkpoint(const std::string& filepath) const;
    void load_checkpoint(const std::string& filepath);

    [[nodiscard]] const StackedRNNConfig& config() const noexcept { return config_; }
    [[nodiscard]] size_t total_parameters() const noexcept;

private:
    StackedRNNConfig config_;

    // Pesos e Gradientes
    Tensor w_emb_;
    Tensor dw_emb_;

    std::vector<LayerWeights> layers_;
    std::vector<LayerGrads> layer_grads_;

    Tensor final_norm_gamma_;
    Tensor dfinal_norm_gamma_;

    Tensor b_head_;
    Tensor db_head_;

    // Cache para backward
    std::vector<LayerActivations> cache_;
    Tensor final_norm_input_;
    Tensor final_norm_x_;
    Tensor final_norm_rstd_;
};

} // namespace nn
} // namespace sore
