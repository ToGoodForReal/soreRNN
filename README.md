# soreRNN

A lightweight C++20 / CUDA framework for recurrent neural networks, built from
scratch. It provides a zero-copy tensor core, an automatic differentiation
engine, a LinearRNN layer (real-gated, Griffin-style), and GPU-accelerated
training primitives (cuBLAS, parallel scan, BPTT, fused AdamW).

## Features

- **Tensor core** (`sore/core`): multidimensional tensors with shape, strides,
  rank and offset. Views, slices and reshape are zero-copy (no allocation).
  CPU and CUDA backends behind a single `Device` abstraction.
- **Automatic differentiation** (`sore/nn/autograd`): reverse-mode engine for
  building and training models.
- **LinearRNN** (`sore/nn/linear_rnn`): real-gated linear recurrence layer
  (RG-LRU / Griffin-style):

  ```
  a_t = sigma(W_gate . x_t + b_gate)      # decay gate in (0, 1)
  u_t = W_in . x_t + b_in                 # input projection
  h_t = a_t * h_{t-1} + u_t               # linear associative recurrence
  y_t = W_out . h_t + b_out               # output projection
  ```
- **CUDA kernels & Memory Pool** (`sore/cuda`): cuBLAS GEMM wrappers, parallel prefix scan,
  BPTT for linear recurrence, fused AdamW step, and a high-performance **CUDACachingAllocator**
  (memory pool) to eliminate runtime cudaMalloc/cudaFree stalls and memory fragmentation.
- **Optimizer** (`sore/optim`): Fused AdamW with decoupled weight decay and full state persistence (moments m and v).
- **Checkpoints & Validation** (`sore/nn`): Binary checkpoints with structured header (`SORE`),
  architecture metadata, faithful training resume (step, tokens, dataloader cursor, LR, AdamW moments),
  and held-out validation evaluation.
- **Data loading** (`sore/data`): mmap-based dataloader for large datasets with cursor serialization.
- **Embeddings and losses** (`sore/nn`): embedding layer and cross-entropy with active token masking.

## Requirements

- CMake >= 3.20
- C++20 compiler (GCC/Clang)
- CUDA Toolkit (arquitetura padrão `sm_86` para Ampere como RTX 3050; suporta override com `-DCMAKE_CUDA_ARCHITECTURES`)
- cuBLAS (incluso no CUDA Toolkit)

### Instalação dos Pré-requisitos (Arch Linux)

```bash
# Driver NVIDIA já deve estar ativo; instalar o toolkit de compilação CUDA:
sudo pacman -S cuda
```

## Build

```bash
# Para a NVIDIA GeForce RTX 3050 (arquitetura sm_86 padrão):
cmake -S . -B build
cmake --build build -j

# Caso queira especificar outra arquitetura explicitamente:
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Tests

Execute toda a suíte de testes unitários e de integração via CTest:

```bash
ctest --test-dir build --output-on-failure
```

Ou execute cada binário de teste individualmente:

```bash
./build/test_storage
./build/test_tensor
./build/test_caching_allocator
./build/test_linear_rnn_cpu
./build/test_cublas_ops
./build/test_parallel_scan
./build/test_autograd_adamw
./build/test_mmap_dataloader
./build/test_conv1d_mlp
./build/test_checkpoint_header
```

## Training examples

```bash
# Runner unificado (pretrain -> SFT -> DPO -> aval):
#   scripts/train.sh [MODE] [MODEL] [PASSOS]   MODE=pretrain|sft|dpo|all|test
scripts/train.sh pretrain 150m 10000      # pre-treino de base (150M)
scripts/train.sh pretrain 300m 15000      # pre-treino (300M, TF32 + grad accum)
scripts/train.sh all 150m 10000           # pipeline completo: pretrain+SFT+DPO+test

# Gerar dados (dedup MinHash + EOS + val held-out, tokenizer PT):
python3 scripts/prepare_4b_pretrain.py --out data/pretrain_75pt_25en.bin --val_out data/val.bin
python3 scripts/prepare_500m_sft.py --out data/sft_chat_pt.bin --val_out data/sft_val.bin

# Executaveis diretos (se preferir):
./build/train_lm_150m 10000 4 1 1024
./build/train_lm_300m 15000 2 4 512

# Inference e avaliacao rapida:
python3 scripts/chat.py --checkpoint checkpoints/sore_lm_150m_pretrain.bin
python3 scripts/evaluate.py --checkpoint checkpoints/sore_lm_150m_sft.bin
```

## Project layout

```
include/sore/
  core/      tensor, storage, common types
  nn/        linear_rnn, autograd, embedding, loss, functional, stacked_rnn
  cuda/      device kernels, caching allocator, scan, cublas handle
  optim/     adamw
  data/      dataloader
src/         host and device implementations
tests/       per-module test executables
examples/    training examples
```

## License

MIT
