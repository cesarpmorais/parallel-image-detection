#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// ===== CUDA Error Checking Macros =====

#define CUDA_CHECK(err)                                                            \
    do                                                                             \
    {                                                                              \
        cudaError_t err__ = (err);                                                 \
        if (err__ != cudaSuccess)                                                  \
        {                                                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err__) << std::endl; \
            std::cerr << "File: " << __FILE__ << std::endl;                        \
            std::cerr << "Line: " << __LINE__ << std::endl;                        \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (false)

#define CUDA_CHECK_KERNEL()                                                             \
    do                                                                                  \
    {                                                                                   \
        cudaError_t err = cudaGetLastError();                                           \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl; \
            std::cerr << "File: " << __FILE__ << std::endl;                             \
            std::cerr << "Line: " << __LINE__ << std::endl;                             \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (false)

// ===== Utility Functions =====

inline size_t get_gpu_memory_total()
{
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    return total_memory;
}

inline size_t get_gpu_memory_free()
{
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    return free_memory;
}

inline void print_gpu_memory_info()
{
    size_t free = get_gpu_memory_free();
    size_t total = get_gpu_memory_total();
    std::cout << "GPU Memory: " << (free / 1024.0 / 1024.0) << " MB free / "
              << (total / 1024.0 / 1024.0) << " MB total" << std::endl;
}
