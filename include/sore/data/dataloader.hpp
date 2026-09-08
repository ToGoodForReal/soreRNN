#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include "sore/core/tensor.hpp"

namespace sore {
namespace data {

/**
 * @brief Leitor de dataset binário de tokens de alto desempenho via POSIX mmap.
 * 
 * Mapeia o arquivo binário diretamente no espaço de endereçamento do processo,
 * evitando cópias no espaço de usuário e habilitando prefetching via madvise(MADV_SEQUENTIAL).
 */
class MMapDataset {
public:
    explicit MMapDataset(const std::string& filepath);
    ~MMapDataset();

    MMapDataset(const MMapDataset&) = delete;
    MMapDataset& operator=(const MMapDataset&) = delete;

    MMapDataset(MMapDataset&& other) noexcept;
    MMapDataset& operator=(MMapDataset&& other) noexcept;

    [[nodiscard]] size_t total_tokens() const noexcept { return num_tokens_; }
    [[nodiscard]] const uint16_t* token_data() const noexcept { return tokens_; }

private:
    int fd_{-1};
    size_t file_size_{0};
    size_t num_tokens_{0};
    uint16_t* tokens_{nullptr};

    void cleanup() noexcept;
};

/**
 * @brief DataLoader em C++ para geração de batches (Input, Target) para Modelos de Linguagem
 */
class DataLoader {
public:
    DataLoader(std::shared_ptr<MMapDataset> dataset, size_t batch_size, size_t seq_len, bool is_paired = false);

    struct Batch {
        std::vector<uint16_t> inputs;  // [B, T]
        std::vector<uint16_t> targets; // [B, T] (offset de +1 token no tempo ou mascarado com 65535)
    };

    [[nodiscard]] bool has_next() const noexcept;
    Batch next();
    void reset() noexcept { current_cursor_ = 0; }

    [[nodiscard]] size_t total_batches() const noexcept;

private:
    std::shared_ptr<MMapDataset> dataset_;
    size_t batch_size_;
    size_t seq_len_;
    bool is_paired_{false};
    size_t current_cursor_{0};
};

} // namespace data
} // namespace sore
