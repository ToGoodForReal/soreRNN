cmake_minimum_required(VERSION 3.20)

# Localização automática do compilador nvcc no Arch Linux (/opt/cuda)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/opt/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
endif()

project(soreRNN LANGUAGES CXX CUDA)

# Padrão C++20 para Host e Device
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Arquitetura CUDA: sm_86 para Ampere (NVIDIA GeForce RTX 3050)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Flags de otimização e rigor de compilador
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall,-Wextra")

# Diretórios de include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Biblioteca core do framework
add_library(sore_core
    src/core/storage.cpp
    src/core/tensor.cpp
    src/nn/functional.cpp
    src/nn/linear_rnn.cpp
    src/nn/autograd.cpp
    src/nn/embedding.cpp
    src/nn/loss.cpp
    src/nn/stacked_rnn.cpp
    src/data/dataloader.cpp
    src/cuda/caching_allocator.cpp
    src/cuda/ops.cu
    src/cuda/scan.cu
    src/cuda/bptt.cu
    src/cuda/adamw.cu
    src/optim/adamw.cpp
)

target_include_directories(sore_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link com CUDA Runtime e cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(sore_core PUBLIC CUDA::cudart CUDA::cublas)

# Testes do Módulo 1
add_executable(test_storage tests/test_storage.cpp)
target_link_libraries(test_storage PRIVATE sore_core)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE sore_core)

add_executable(test_caching_allocator tests/test_caching_allocator.cpp)
target_link_libraries(test_caching_allocator PRIVATE sore_core)

# Teste do Módulo 2
add_executable(test_linear_rnn_cpu tests/test_linear_rnn_cpu.cpp)
target_link_libraries(test_linear_rnn_cpu PRIVATE sore_core)

# Teste do Módulo 3
add_executable(test_cublas_ops tests/test_cublas_ops.cpp)
target_link_libraries(test_cublas_ops PRIVATE sore_core)

# Teste do Módulo 4
add_executable(test_parallel_scan tests/test_parallel_scan.cpp)
target_link_libraries(test_parallel_scan PRIVATE sore_core)

# Teste do Módulo 5
add_executable(test_autograd_adamw tests/test_autograd_adamw.cpp)
target_link_libraries(test_autograd_adamw PRIVATE sore_core)

# Testes do Módulo 6
add_executable(test_mmap_dataloader tests/test_mmap_dataloader.cpp)
target_link_libraries(test_mmap_dataloader PRIVATE sore_core)

# Teste de Conv1D e GELU MLP
add_executable(test_conv1d_mlp tests/test_conv1d_mlp.cpp)
target_link_libraries(test_conv1d_mlp PRIVATE sore_core)

# Teste de Checkpoint Header e Resume Fiel
add_executable(test_checkpoint_header tests/test_checkpoint_header.cpp)
target_link_libraries(test_checkpoint_header PRIVATE sore_core)

# Executáveis de Treinamento
add_executable(train_linear_rnn examples/train_linear_rnn.cpp)
target_link_libraries(train_linear_rnn PRIVATE sore_core)

add_executable(train_sequential examples/train_sequential.cpp)
target_link_libraries(train_sequential PRIVATE sore_core)

add_executable(train_lm_150m examples/train_lm_150m.cpp)
target_link_libraries(train_lm_150m PRIVATE sore_core)

add_executable(train_sft examples/train_sft.cpp)
target_link_libraries(train_sft PRIVATE sore_core)

add_executable(train_lm_300m examples/train_lm_300m.cpp)
target_link_libraries(train_lm_300m PRIVATE sore_core)

add_executable(train_dpo examples/train_dpo.cpp)
target_link_libraries(train_dpo PRIVATE sore_core)

# Configuração CTest / Runner Unificado
enable_testing()
add_test(NAME test_storage COMMAND test_storage)
add_test(NAME test_tensor COMMAND test_tensor)
add_test(NAME test_caching_allocator COMMAND test_caching_allocator)
add_test(NAME test_linear_rnn_cpu COMMAND test_linear_rnn_cpu)
add_test(NAME test_cublas_ops COMMAND test_cublas_ops)
add_test(NAME test_parallel_scan COMMAND test_parallel_scan)
add_test(NAME test_autograd_adamw COMMAND test_autograd_adamw)
add_test(NAME test_mmap_dataloader COMMAND test_mmap_dataloader)
add_test(NAME test_conv1d_mlp COMMAND test_conv1d_mlp)
add_test(NAME test_checkpoint_header COMMAND test_checkpoint_header)



