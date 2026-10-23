#include "sore/cuda/caching_allocator.hpp"
#include "sore/core/storage.hpp"
#include "sore/core/tensor.hpp"
#include <cassert>
#include <iostream>

void test_caching_and_reuse() {
    std::cout << "[Test] CUDACachingAllocator: Reúso de blocos livres..." << std::endl;
    auto& alloc = sore::cuda::CUDACachingAllocator::instance();
    alloc.empty_cache();

    size_t size1 = 1024 * 1024 * sizeof(float); // 4 MB
    void* ptr1 = alloc.allocate(size1);
    assert(ptr1 != nullptr);

    auto stats1 = alloc.stats();
    assert(stats1.num_cuda_malloc_calls == 1);
    assert(stats1.allocated_bytes >= size1);
    assert(stats1.cached_bytes == 0);

    // Devolve para o cache do pool
    alloc.deallocate(ptr1);
    auto stats2 = alloc.stats();
    assert(stats2.allocated_bytes == 0);
    assert(stats2.cached_bytes >= size1);

    // Aloca novamente o mesmo tamanho - deve reusar o bloco sem chamar cudaMalloc!
    void* ptr2 = alloc.allocate(size1);
    assert(ptr2 == ptr1); // Ponteiro idêntico reaproveitado do cache
    auto stats3 = alloc.stats();
    assert(stats3.num_cuda_malloc_calls == 1); // Nenhuma chamada nova ao driver CUDA!
    assert(stats3.cached_bytes == 0);

    alloc.deallocate(ptr2);
    std::cout << " -> Reúso de bloco comprovado: ponteiro " << ptr1 << " reaproveitado com zero overhead de driver!" << std::endl;
}

void test_storage_tensor_integration() {
    std::cout << "[Test] CUDACachingAllocator: Integração com Storage e Tensor..." << std::endl;
    auto& alloc = sore::cuda::CUDACachingAllocator::instance();
    alloc.empty_cache();
    size_t initial_mallocs = alloc.stats().num_cuda_malloc_calls;

    {
        // Cria tensor na GPU
        sore::Tensor t1 = sore::Tensor::zeros({2, 1024, 1024}, sore::DType::Float32, sore::Device::CUDA);
        assert(t1.device() == sore::Device::CUDA);
        assert(t1.data<float>() != nullptr);
        assert(alloc.stats().num_cuda_malloc_calls == initial_mallocs + 1);
    } // t1 sai de escopo e é destruído aqui

    assert(alloc.stats().cached_bytes > 0);

    {
        // Cria novo tensor com mesmo shape - deve cair 100% no cache sem nova chamada cudaMalloc
        sore::Tensor t2 = sore::Tensor::zeros({2, 1024, 1024}, sore::DType::Float32, sore::Device::CUDA);
        assert(alloc.stats().num_cuda_malloc_calls == initial_mallocs + 1);
    }

    std::cout << " -> Integração Tensor <-> Storage <-> CachingAllocator validada com sucesso!" << std::endl;
}

void test_empty_cache() {
    std::cout << "[Test] CUDACachingAllocator: empty_cache()..." << std::endl;
    auto& alloc = sore::cuda::CUDACachingAllocator::instance();
    void* p = alloc.allocate(2 * 1024 * 1024);
    alloc.deallocate(p);

    assert(alloc.stats().cached_bytes > 0);
    alloc.empty_cache();
    assert(alloc.stats().cached_bytes == 0);
    std::cout << " -> empty_cache() limpou todos os blocos com sucesso!" << std::endl;
}

int main() {
    std::cout << "=== Bateria de Testes: CUDACachingAllocator ===" << std::endl;
    try {
        test_caching_and_reuse();
        test_storage_tensor_integration();
        test_empty_cache();
        std::cout << "TODOS OS TESTES DO CACHING ALLOCATOR PASSARAM!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
