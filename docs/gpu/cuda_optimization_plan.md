# CUDA Parallelization Plan for ResNet-18

## Overview

This document outlines a strategy to accelerate ResNet-18 inference using NVIDIA GPUs via CUDA. The goal is to match or exceed CPU+OpenMP performance by offloading compute-intensive operations (convolution, matrix multiplication, batch normalization) to the GPU.

---

## 1. Architecture Overview

### Current Bottlenecks (from sequential CPU)
Based on per-layer timing analysis:

| Layer | Time (ms) | % of Total | Computation |
|-------|-----------|-----------|------------|
| conv1 | 315 | 6.8% | 3×3 convolutions (1 layer) |
| layer1-4 | ~4000 | 86% | Convolutions + BatchNorm (8 blocks) |
| avgpool | 0.05 | <0.1% | Reduction |
| fc | 1.3 | 0.03% | Matrix multiplication |
| **Total** | **4647** | **100%** | |

**Key Insight:** Convolution and residual blocks dominate execution time. These are highly parallelizable on GPUs.

### GPU Acceleration Strategy

**Phase 1 (High Priority - 80% speedup potential)**
- Conv2D: Parallelize convolution on GPU
- BatchNorm2D: Fuse with preceding convolution
- ReLU: Simple activation (kernel cost negligible)

**Phase 2 (Medium Priority - 15% speedup potential)**
- Linear: GPU matrix multiplication
- AdaptiveAvgPool: Reduction kernel
- Memory management: Minimize CPU↔GPU transfers

**Phase 3 (Low Priority - optimization)**
- Layer fusion: Combine Conv+BN+ReLU into single kernel
- Quantization: Reduce memory bandwidth (optional)

---

## 2. Implementation Plan

### 2.1 Project Structure

Create a new directory mirroring `cpp/`:

```
cpp_cuda/
├── CMakeLists.txt           # CUDA configuration
├── src/
│   ├── main.cu             # Main program (host code)
│   ├── conv2d.cu           # GPU convolution kernel + host wrapper
│   ├── batchnorm.cu        # GPU batch norm kernel
│   ├── relu.cu             # Simple activation kernel
│   ├── linear.cu           # GPU matrix multiply
│   ├── tensor.cu           # CUDA tensor management
│   ├── maxpool.cu          # Max pooling kernel
│   ├── adaptiveavgpool.cu  # Adaptive average pool kernel
│   └── basicblock.cu       # GPU residual block orchestrator
├── include/
│   ├── tensor.h            # Tensor class with GPU memory management
│   ├── conv2d.h
│   ├── batchnorm.h
│   ├── ... (headers for all layers)
│   └── cuda_utils.h        # CUDA error checking macros
└── build/
    └── resnet18            # Compiled CUDA binary
```

### 2.2 Core Components to Implement

#### A. Tensor Class with GPU Memory Management

**File:** `cpp_cuda/include/tensor.h` and `cpp_cuda/src/tensor.cu`

```cpp
class Tensor {
public:
    // CPU memory
    std::vector<float> data;        // Host-side data
    std::vector<int> shape;
    
    // GPU memory
    float* gpu_data;                // Device-side pointer
    bool on_gpu;                    // Tracks if data is on GPU
    
    // Methods
    void to_gpu();                  // Transfer CPU → GPU
    void to_cpu();                  // Transfer GPU → CPU
    void allocate_gpu();            // Allocate GPU memory
    void free_gpu();                // Free GPU memory
    
    // Forward declaration for CUDA kernels
    friend class Conv2D;
    friend class BatchNorm2D;
    // ... etc
};
```

#### B. Conv2D Kernel (Highest Priority)

**Approach: Use cuDNN for convolution**

cuDNN (CUDA Deep Neural Network library) provides highly optimized convolution kernels:

```cpp
// cpp_cuda/src/conv2d.cu
#include <cudnn.h>

class Conv2D {
private:
    float* gpu_weights;            // GPU memory for weights
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
public:
    Tensor forward(const Tensor& input);
    // Implementation uses cudnnConvolutionForward()
};
```

**Alternative (if cuDNN unavailable):** Write custom CUDA kernel
- Parallelize over output pixels/channels
- Use shared memory for input tile caching
- Thread block: 16×16 threads (256 threads)
- Each thread: compute one output pixel

#### C. BatchNorm2D Kernel

**Key Optimization: Fuse with preceding Conv**

Instead of: Conv → BN → ReLU (3 kernel calls)
- **Fused:** Conv → (BN+ReLU) (single GPU pass)

```cpp
// cpp_cuda/src/batchnorm.cu
class BatchNorm2D {
public:
    // Batch norm: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Tensor forward(const Tensor& input);
    
    // Fused kernel: apply BN directly after convolution
    void forward_fused_with_conv(const Tensor& conv_output, 
                                 float* output_gpu);
};
```

**Batch Norm Formula (fused with conv):**
```
For each pixel (c, h, w):
    normalized = (conv_output - running_mean[c]) / sqrt(running_var[c] + eps)
    output = gamma[c] * normalized + beta[c]
```

Thread layout: One thread per output element (parallelizable)

#### D. ReLU Kernel

**Trivial:** Element-wise max(0, x)

```cpp
// cpp_cuda/src/relu.cu
__global__ void relu_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = max(0.0f, data[idx]);
    }
}
```

#### E. Linear (Matrix Multiplication) Kernel

**Use cuBLAS (CUDA Basic Linear Algebra Subroutines)**

```cpp
// cpp_cuda/src/linear.cu
#include <cublas_v2.h>

class Linear {
private:
    float* gpu_weights;            // [out_features, in_features]
    float* gpu_bias;               // [out_features]
    cublasHandle_t cublas_handle;
    
public:
    Tensor forward(const Tensor& input);
    // Uses cublasSgemm for matrix multiply: output = weights @ input.T + bias
};
```

#### F. MaxPool2D and AdaptiveAvgPool2D

**MaxPool:** Custom CUDA kernel
- Thread block computes max over pool region
- Simple parallel reduction

```cpp
// cpp_cuda/src/maxpool.cu
__global__ void maxpool_kernel(float* input, int input_h, int input_w,
                               float* output, int output_h, int output_w,
                               int pool_size, int stride) {
    // Each thread computes one output pixel
    // Iterate over pool region to find max
}
```

**AdaptiveAvgPool:** Parallel reduction
- Compute average over spatial dimensions efficiently

---

## 3. Memory Management Strategy

### 3.1 Data Transfer Minimization

**Current (CPU):** Load weights once, process multiple images sequentially

**GPU Approach:**
1. **Persistent GPU Memory:** Load all weights to GPU once (at program start)
   - Conv1, BN1, Layer1-4 weights: ~60 MB total
   - Transfer once, keep on GPU for all images

2. **Per-Image GPU Pipeline:**
   - Transfer image (3×224×224 float32 = 589 KB) to GPU
   - Run entire forward pass on GPU
   - Transfer output (1000 floats = 4 KB) back to CPU
   - **Total transfer per image:** ~600 KB (negligible vs. compute time)

3. **Intermediate Buffers:** Keep on GPU
   - after_conv1: 64×112×112×4 = 3.2 MB
   - after_layer1: 64×56×56×4 = 802 KB
   - Etc.
   - **Total intermediate memory:** ~30 MB (for single image batching)

### 3.2 GPU Memory Layout

**Row-major (C-style):** Same as CPU for consistency
```
Shape: [batch=1, channels, height, width]
Memory: contiguous [C, H, W] for each batch
```

---

## 4. Expected Performance Gains

### Theoretical Analysis

| Operation | CPU Time | GPU Time | Speedup | Notes |
|-----------|----------|----------|---------|-------|
| Conv1 (315 ms) | 315 ms | ~30 ms | **10x** | cuDNN optimizations |
| Layer1 (1200 ms) | 1200 ms | ~100 ms | **12x** | 2× Conv2D blocks |
| Layer2-4 (3000 ms) | 3000 ms | ~250 ms | **12x** | Same pattern |
| BN+ReLU (overhead) | ~50 ms | ~5 ms | **10x** | Fused kernels |
| Linear (1.3 ms) | 1.3 ms | ~0.5 ms | **2.6x** | cuBLAS |
| **Total** | **4647 ms** | **~385 ms** | **~12x** | Optimistic estimate |

**Realistic estimate:** 8-10x speedup (after considering memory transfers and overhead)

### Bottleneck After GPU Acceleration

Once compute is on GPU, bottlenecks become:
1. **GPU memory bandwidth:** Convolution reads/writes large tensors
2. **cuDNN efficiency:** Library-specific optimizations
3. **PCIe transfer:** Minimal if we batch process

**Strategy:** Process multiple images in parallel (mini-batches) if GPU memory allows

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up CUDA toolchain (NVCC compiler, cuDNN, cuBLAS)
- [ ] Create `cpp_cuda/` directory structure
- [ ] Implement Tensor class with GPU memory management
- [ ] Add CUDA error checking macros
- [ ] Create simple Conv2D kernel (or cuDNN wrapper)

### Phase 2: Core Layers (Week 2-3)
- [ ] Implement BatchNorm2D kernel (fused with Conv if possible)
- [ ] Implement ReLU, MaxPool, AdaptiveAvgPool kernels
- [ ] Implement Linear layer with cuBLAS
- [ ] Test individual layers against CPU reference

### Phase 3: Integration (Week 3-4)
- [ ] Create GPU BasicBlock (orchestrate layer sequence)
- [ ] Integrate all layers into main.cu
- [ ] Add CLI argument handling (same as CPU version)
- [ ] Validate outputs against PyTorch reference

### Phase 4: Optimization (Week 4-5)
- [ ] Profile with `nvprof` or `nsys`
- [ ] Identify remaining bottlenecks
- [ ] Implement kernel fusion if needed
- [ ] Batch multiple images if memory allows

### Phase 5: Benchmarking & Validation (Week 5-6)
- [ ] Create GPU binary (`cpp_cuda/build/resnet18_cuda`)
- [ ] Update `benchmark_parallel_vs_sequential.py` to include GPU
- [ ] Run full comparison: Sequential vs. OpenMP vs. CUDA
- [ ] Generate performance report

---

## 6. Common Pitfalls & Solutions

### Pitfall 1: Memory Inefficiency
**Problem:** Transferring data between CPU and GPU repeatedly
**Solution:** Batch process images on GPU, keep weights persistent

### Pitfall 2: Unoptimized Kernels
**Problem:** Writing naive CUDA kernels that underutilize GPU
**Solution:** Use cuDNN/cuBLAS where possible, profile custom kernels

### Pitfall 3: Numerical Precision
**Problem:** float32 GPU results differ from float64 CPU
**Solution:** Use float32 consistently on both CPU and GPU, allow tolerance ~1e-4

### Pitfall 4: Synchronization Overhead
**Problem:** Copying intermediate results for validation hurts performance
**Solution:** Validate only final output, keep intermediate buffers on GPU

### Pitfall 5: Small Batch Size
**Problem:** Single image (batch=1) doesn't fully utilize GPU
**Solution:** Process 4-8 images per batch if memory allows, or increase batch size in CLI

---

## 7. Validation Strategy

### Layer-by-Layer Validation

```cpp
// In main.cu, after each layer:
conv1_output.to_cpu();  // Transfer GPU result to CPU
float max_diff = compare_with_reference(conv1_output, reference_conv1);
if (max_diff > 1e-4) {
    std::cerr << "Conv1 validation failed: diff=" << max_diff << std::endl;
}
```

### Final Output Validation

```bash
# Compare GPU predictions vs. PyTorch
python validate.py --cpp-binary cpp_cuda/build/resnet18_cuda --tolerance 1e-4
```

---

## 8. Dependencies & Tools

### Required Libraries
- **CUDA Toolkit 11.0+** (compiler, runtime, profiler)
- **cuDNN 8.0+** (optimized convolution, batch norm)
- **cuBLAS** (included in CUDA Toolkit)
- **CMake 3.17+** (CUDA project support)

### Optional Tools
- **NVIDIA Visual Profiler (nvvp)** or **Nsight Systems** for profiling
- **NVIDIA Nsight Compute** for kernel analysis
- **Thrust library** (GPU algorithms, included in CUDA)

### CMakeLists.txt Template
```cmake
cmake_minimum_required(VERSION 3.17)
project(resnet18_cuda CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Adjust for your GPU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

find_package(CUDA REQUIRED)
find_package(cuDNN REQUIRED)

add_executable(resnet18_cuda
    src/main.cu
    src/conv2d.cu
    src/batchnorm.cu
    # ... other .cu files
)

target_link_libraries(resnet18_cuda
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    cuDNN::cuDNN
)
```

---

## 9. Benchmarking & Reporting

### Metrics to Collect

For each implementation (Sequential, OpenMP, CUDA):
- Total inference time (ms)
- Per-layer timing breakdown
- Memory usage (peak, average)
- Power consumption (if GPU supports it)
- Numerical accuracy (max error vs. PyTorch)

### Expected Report Table

| Implementation | Inference (ms) | Speedup | Memory (MB) | Accuracy (MAE) |
|---|---|---|---|---|
| Sequential CPU | 4647 | 1x | 150 | 1.2e-5 |
| OpenMP (4 cores) | 2400 | 1.9x | 150 | 1.2e-5 |
| CUDA (V100) | 450 | **10.3x** | 200 | 1.5e-4 |

---

## 10. Next Steps

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   ```
   Note: GPU model, compute capability, memory

2. **Install CUDA toolkit:**
   ```bash
   # Linux
   wget https://developer.nvidia.com/cuda-downloads
   # Follow NVIDIA instructions
   ```

3. **Verify installation:**
   ```bash
   nvcc --version
   ```

4. **Start Phase 1:** Set up project structure and Tensor class

---

## References

- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- cuDNN User Guide: https://docs.nvidia.com/deeplearning/cudnn/user-guide/
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- Best Practices for CUDA: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

## Appendix: Minimal CUDA Example

### Simple Conv2D using cuDNN

```cpp
// Example: Forward pass with cuDNN
#include <cudnn.h>
#include <cuda_runtime.h>

void convolution_forward_cudnn(float* input_gpu, float* weights_gpu,
                               float* output_gpu,
                               cudnnHandle_t cudnn_handle,
                               cudnnTensorDescriptor_t input_desc,
                               cudnnTensorDescriptor_t output_desc,
                               cudnnFilterDescriptor_t filter_desc,
                               cudnnConvolutionDescriptor_t conv_desc) {
    
    float alpha = 1.0f, beta = 0.0f;
    
    cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc, input_gpu,
        filter_desc, weights_gpu,
        conv_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        nullptr, 0,  // workspace
        &beta,
        output_desc, output_gpu
    );
}
```

This document provides a solid foundation for CUDA implementation. Start with Phase 1, validate rigorously, and measure performance at each step.
