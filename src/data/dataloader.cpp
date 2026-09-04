#include "sore/data/dataloader.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace sore {
namespace data {

MMapDataset::MMapDataset(const std::string& filepath) {
    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::ostringstream oss;
        oss << "Falha ao abrir arquivo para mmap: " << filepath << " (" << strerror(errno) << ")";
        throw std::runtime_error(oss.str());
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Falha ao consultar fstat do arquivo de dados.");
    }

    file_size_ = static_cast<size_t>(sb.st_size);
    num_tokens_ = file_size_ / sizeof(uint16_t);

    if (file_size_ == 0) {
        close(fd_);
        fd_ = -1;
        return;
    }

    // Mapeamento virtual direto no espaço de endereçamento (Zero-Copy)
    void* addr = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Falha na chamada de sistema mmap.");
    }

    tokens_ = static_cast<uint16_t*>(addr);

    // Instrução ao kernel Linux para prefetch de páginas sequenciais
    madvise(tokens_, file_size_, MADV_SEQUENTIAL | MADV_WILLNEED);
}

MMapDataset::~MMapDataset() {
    cleanup();
}

MMapDataset::MMapDataset(MMapDataset&& other) noexcept
    : fd_(other.fd_), file_size_(other.file_size_), num_tokens_(other.num_tokens_), tokens_(other.tokens_) {
    other.fd_ = -1;
    other.file_size_ = 0;
    other.num_tokens_ = 0;
    other.tokens_ = nullptr;
}

MMapDataset& MMapDataset::operator=(MMapDataset&& other) noexcept {
    if (this != &other) {
        cleanup();
        fd_ = other.fd_;
        file_size_ = other.file_size_;
        num_tokens_ = other.num_tokens_;
        tokens_ = other.tokens_;
        other.fd_ = -1;
        other.file_size_ = 0;
        other.num_tokens_ = 0;
        other.tokens_ = nullptr;
    }
    return *this;
}

void MMapDataset::cleanup() noexcept {
    if (tokens_ != nullptr && file_size_ > 0) {
        munmap(tokens_, file_size_);
        tokens_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
    num_tokens_ = 0;
}

DataLoader::DataLoader(std::shared_ptr<MMapDataset> dataset, size_t batch_size, size_t seq_len)
    : dataset_(std::move(dataset)), batch_size_(batch_size), seq_len_(seq_len) {
    if (!dataset_ || dataset_->total_tokens() == 0) {
        throw std::runtime_error("DataLoader inicializado com dataset vazio.");
    }
}

bool DataLoader::has_next() const noexcept {
    if (!dataset_) return false;
    size_t tokens_needed = batch_size_ * seq_len_ + 1;
    return (current_cursor_ + tokens_needed) <= dataset_->total_tokens();
}

size_t DataLoader::total_batches() const noexcept {
    if (!dataset_ || dataset_->total_tokens() <= 1) return 0;
    size_t tokens_per_batch = batch_size_ * seq_len_;
    return (dataset_->total_tokens() - 1) / tokens_per_batch;
}

DataLoader::Batch DataLoader::next() {
    if (!has_next()) {
        throw std::out_of_range("Fim do dataset atingido no DataLoader.");
    }

    Batch batch;
    batch.inputs.resize(batch_size_ * seq_len_);
    batch.targets.resize(batch_size_ * seq_len_);

    const uint16_t* raw_tokens = dataset_->token_data();

    for (size_t b = 0; b < batch_size_; ++b) {
        size_t start = current_cursor_ + b * seq_len_;
        for (size_t t = 0; t < seq_len_; ++t) {
            size_t out_idx = b * seq_len_ + t;
            batch.inputs[out_idx] = raw_tokens[start + t];
            batch.targets[out_idx] = raw_tokens[start + t + 1]; // Próximo token no tempo
        }
    }

    current_cursor_ += batch_size_ * seq_len_;
    return batch;
}

} // namespace data
} // namespace sore
