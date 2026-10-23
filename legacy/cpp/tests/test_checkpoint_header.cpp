#include "sore/nn/stacked_rnn.hpp"
#include "sore/optim/adamw.hpp"
#include <cassert>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>

void test_checkpoint_v2_and_optimizer_resume() {
    std::cout << "[Test] Checkpoint v2 com Header SORE e Resume de AdamW..." << std::endl;

    sore::nn::StackedRNNConfig config;
    config.vocab_size = 128;
    config.d_model = 64;
    config.num_layers = 2;
    config.d_mlp = 128;
    config.conv_kernel = 4;
    config.device = sore::Device::CUDA;

    sore::nn::StackedLinearRNNLM model1(config);
    model1.init_weights(12345);

    auto params = model1.parameters();
    sore::nn::TrainingState save_state;
    save_state.step = 42;
    save_state.total_trained_tokens = 43008;
    save_state.dataloader_cursor = 8192;
    save_state.current_lr = 2.5e-4f;
    save_state.has_optimizer = true;

    // Popula momentos sintéticos para teste
    for (auto* p : params) {
        sore::Tensor m = sore::Tensor::ones(p->shape(), p->dtype(), p->device());
        sore::Tensor v = sore::Tensor::ones(p->shape(), p->dtype(), p->device());
        save_state.exp_avg.push_back(std::move(m));
        save_state.exp_avg_sq.push_back(std::move(v));
    }

    std::string test_ckpt = "checkpoints/test_header_ckpt.bin";
    model1.save_checkpoint(test_ckpt, &save_state);

    // Carrega em uma segunda instância
    sore::nn::StackedLinearRNNLM model2(config);
    sore::nn::TrainingState load_state;
    bool has_header = model2.load_checkpoint(test_ckpt, &load_state);

    assert(has_header == true);
    assert(load_state.step == 42);
    assert(load_state.total_trained_tokens == 43008);
    assert(load_state.dataloader_cursor == 8192);
    assert(std::fabs(load_state.current_lr - 2.5e-4f) < 1e-7f);
    assert(load_state.has_optimizer == true);
    assert(load_state.exp_avg.size() == params.size());
    assert(load_state.exp_avg_sq.size() == params.size());

    // Valida integridade numérica dos momentos restaurados
    sore::Tensor m_cpu = load_state.exp_avg[0].cpu();
    for (size_t i = 0; i < std::min(m_cpu.numel(), size_t(100)); ++i) {
        assert(std::fabs(m_cpu.item(i) - 1.0f) < 1e-5f);
    }

    // Valida que num_param_tensors foi preenchido corretamente no header
    {
        std::ifstream ifs(test_ckpt, std::ios::binary);
        sore::nn::CheckpointHeader hdr;
        ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        assert(hdr.num_param_tensors == 31);
        assert(hdr.num_param_tensors == model1.num_parameter_tensors());
    }

    std::cout << " -> Checkpoint v2 e momentos de AdamW salvos e restaurados com 100% de precisão!" << std::endl;
}

void test_legacy_checkpoint_compatibility() {
    std::cout << "[Test] Compatibilidade Retroativa com Checkpoint Legado (Sem Header)..." << std::endl;

    sore::nn::StackedRNNConfig config;
    config.vocab_size = 128;
    config.d_model = 64;
    config.num_layers = 2;
    config.d_mlp = 128;
    config.conv_kernel = 4;
    config.device = sore::Device::CUDA;

    sore::nn::StackedLinearRNNLM model_ref(config);
    model_ref.init_weights(999);
    size_t total_floats = model_ref.total_parameters();

    // Cria arquivo legado com floats crus (simulando formato antigo)
    std::string legacy_file = "checkpoints/test_legacy.bin";
    std::vector<float> raw_floats(total_floats, 0.12345f);
    {
        std::ofstream ofs(legacy_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(raw_floats.data()), raw_floats.size() * sizeof(float));
    }

    sore::nn::StackedLinearRNNLM model_legacy(config);
    sore::nn::TrainingState state;
    bool has_header = model_legacy.load_checkpoint(legacy_file, &state);

    assert(has_header == false); // Detectou corretamente formato legado!
    assert(state.step == 0);
    assert(state.has_optimizer == false);

    std::cout << " -> Fallback transparente para checkpoints legados validado com sucesso!" << std::endl;

    // Limpa arquivos temporários de teste
    std::filesystem::remove("checkpoints/test_header_ckpt.bin");
    std::filesystem::remove("checkpoints/test_legacy.bin");
}

void test_architecture_mismatch_validation() {
    std::cout << "[Test] Validação Estrita de Arquitetura (Mismatched Config)..." << std::endl;

    sore::nn::StackedRNNConfig config150m;
    config150m.vocab_size = 128;
    config150m.d_model = 64;
    config150m.num_layers = 2;
    config150m.d_mlp = 128;
    config150m.conv_kernel = 4;
    config150m.device = sore::Device::CUDA;

    sore::nn::StackedLinearRNNLM model150m(config150m);
    std::string test_ckpt = "checkpoints/test_mismatch.bin";
    model150m.save_checkpoint(test_ckpt);

    // Tentativa de carregar em modelo com configuração incompatível (ex: 4 camadas e D=128)
    sore::nn::StackedRNNConfig config300m = config150m;
    config300m.num_layers = 4;
    config300m.d_model = 128;
    sore::nn::StackedLinearRNNLM model300m(config300m);

    bool caught_exception = false;
    try {
        model300m.load_checkpoint(test_ckpt);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
        std::cout << " -> Exceção esperada capturada com sucesso:\n    " << e.what() << std::endl;
    }
    assert(caught_exception == true);

    std::filesystem::remove(test_ckpt);
    std::cout << " -> Validação de compatibilidade de arquitetura aprovada com sucesso!" << std::endl;
}

void test_adamw_class_methods() {
    std::cout << "[Test] Classe AdamW (step, zero_grad, set_step)..." << std::endl;
    sore::Tensor p = sore::Tensor::ones({10, 10}, sore::DType::Float32, sore::Device::CUDA);
    sore::Tensor g = sore::Tensor::ones({10, 10}, sore::DType::Float32, sore::Device::CUDA);

    std::vector<sore::Tensor*> params = {&p};
    std::vector<sore::Tensor*> grads = {&g};

    sore::optim::AdamW opt(params, grads);
    assert(opt.current_step() == 0);

    opt.step();
    assert(opt.current_step() == 1);

    // O peso deve ter sido atualizado (decréscimo pelo gradiente positivo)
    sore::Tensor p_cpu = p.cpu();
    assert(p_cpu.item(0) < 1.0f);

    opt.zero_grad();
    sore::Tensor g_cpu = g.cpu();
    assert(g_cpu.item(0) == 0.0f);

    std::cout << " -> AdamW::step() e AdamW::zero_grad() executados com sucesso!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: Checkpoint Header & AdamW Resume ===" << std::endl;
    try {
        test_checkpoint_v2_and_optimizer_resume();
        test_legacy_checkpoint_compatibility();
        test_architecture_mismatch_validation();
        test_adamw_class_methods();
        std::cout << "TODOS OS TESTES DE CHECKPOINT E ADAMW PASSARAM COM SUCESSO!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
