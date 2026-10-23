#include "sore/nn/linear_rnn.hpp"
#include <cmath>

namespace sore {
namespace nn {

LinearRNN::LinearRNN(size_t in_dim, size_t hidden_dim, size_t out_dim, Device device)
    : in_dim_(in_dim),
      hidden_dim_(hidden_dim),
      out_dim_(out_dim),
      device_(device),
      w_gate_({hidden_dim, in_dim}, DType::Float32, device),
      b_gate_({hidden_dim}, DType::Float32, device),
      w_in_({hidden_dim, in_dim}, DType::Float32, device),
      b_in_({hidden_dim}, DType::Float32, device),
      w_out_({out_dim, hidden_dim}, DType::Float32, device),
      b_out_({out_dim}, DType::Float32, device) {
    init_weights(42);
}

void LinearRNN::init_weights(uint64_t seed) {
    std::mt19937_64 rng(seed);

    auto xavier_init = [&](Tensor& t, size_t fan_in, size_t fan_out) {
        float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        float* ptr = t.data<float>();
        for (size_t i = 0; i < t.numel(); ++i) {
            ptr[i] = dist(rng);
        }
    };

    xavier_init(w_gate_, in_dim_, hidden_dim_);
    xavier_init(w_in_, in_dim_, hidden_dim_);
    xavier_init(w_out_, hidden_dim_, out_dim_);

    // Inicialização do viés de gate com +1.0f para favorecer retenção de memória no início (Griffin / LRU trick)
    b_gate_.fill_(1.0f);
    b_in_.zero_();
    b_out_.zero_();
}

Tensor LinearRNN::forward(const Tensor& x) {
    if (x.rank() != 3) {
        throw std::runtime_error("Entrada da LinearRNN deve ser 3D [Batch, SeqLen, InDim].");
    }
    if (x.dim(2) != in_dim_) {
        throw std::runtime_error("Dimensão in_dim de x não coincide com a camada.");
    }

    if (device_ == Device::CPU) {
        // 1. Projeção de decay gate: G = X @ W_gate^T + b_gate
        Tensor G = functional::linear_cpu(x, w_gate_, &b_gate_);

        // 2. Gate de decaimento em (0, 1): A = \sigma(G)
        Tensor A = functional::sigmoid_cpu(G);

        // 3. Projeção de entrada: U = X @ W_in^T + b_in
        Tensor U = functional::linear_cpu(x, w_in_, &b_in_);

        // 4. Recorrência Linear no tempo: h_t = a_t * h_{t-1} + u_t
        Tensor H = functional::linear_rnn_forward_cpu(A, U, nullptr);

        // 5. Projeção de saída: Y = H @ W_out^T + b_out
        Tensor Y = functional::linear_cpu(H, w_out_, &b_out_);

        return Y;
    } else {
        throw std::runtime_error("Execução GPU será ativada nos Módulos 3 e 4 via cuBLAS e Parallel Scan.");
    }
}

} // namespace nn
} // namespace sore
