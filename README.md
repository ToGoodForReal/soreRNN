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
- **CUDA kernels** (`sore/cuda`): cuBLAS GEMM wrappers, parallel prefix scan,
  BPTT for the linear recurrence, and a fused AdamW step.
- **Optimizer** (`sore/optim`): AdamW with weight decay.
- **Data loading** (`sore/data`): mmap-based dataloader for large datasets.
- **Embeddings and losses** (`sore/nn`): embedding layer and common loss
  functions.

## Requirements

- CMake >= 3.20
- C++20 compiler (GCC/Clang)
- CUDA Toolkit (target architecture `sm_89`, Ada Lovelace: L40S / RTX 4090)
- cuBLAS (bundled with the CUDA Toolkit)

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Tests

Each module has a standalone test executable:

```bash
./build/test_storage
./build/test_tensor
./build/test_linear_rnn_cpu
./build/test_cublas_ops
./build/test_parallel_scan
./build/test_autograd_adamw
./build/test_mmap_dataloader
```

## Training example

```bash
./build/train_linear_rnn
```

See `examples/train_linear_rnn.cpp` for a full training loop using the
LinearRNN layer, AdamW, and the CUDA BPTT kernel.

## Project layout

```
include/sore/
  core/      tensor, storage, common types
  nn/        linear_rnn, autograd, embedding, loss, functional
  cuda/      device kernels (ops, scan, cublas handle)
  optim/     adamw
  data/      dataloader
src/         host and device implementations
tests/       per-module test executables
examples/    training examples
```

## License

MIT
