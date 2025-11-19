# CUDA Parallelization Phases 2-5: Complete Roadmap

## Overview

Phase 1 established the GPU infrastructure foundation. Phases 2-5 will implement actual GPU kernels and measure performance improvements.

---

## Phase 2: GPU Kernel Implementation (Conv2D & BatchNorm)

### Goal
Implement the most expensive operations in ResNet18:
- **Conv2D** - Convolution (70% of compute time)
- **BatchNorm** - Normalization (15% of compute time)

### What's Different from Phase 1
- **Phase 1:** Copy data to GPU, do nothing, copy back (slow!)
- **Phase 2:** Copy data to GPU, run CUDA kernels on GPU, copy back (fast!)

### Implementation Details

#### Step 1: Conv2D GPU Kernel
```cuda
// Current: CPU only
Tensor Conv2D::forward(const Tensor& input) {
    Tensor output(batch, out_channels, out_h, out_w);
    // 4 nested loops on CPU - SLOW for large tensors
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    // Manual convolution...
                }
            }
        }
    }
    return output;
}

// New: GPU kernel (in Phase 2)
__global__ void conv2d_kernel(
    const float* input, const float* weights, float* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w, int kernel_size, int stride, int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch * out_channels * out_h * out_w;
    
    if (idx < total_outputs) {
        // Each thread computes one output element in parallel
        // With 2048 threads per block, thousands of elements compute simultaneously
        compute_output_element(input, weights, output, idx, ...);
    }
}
```

**Performance Impact:**
- CPU: ~500 ms for ResNet forward pass (on i7)
- GPU: ~10 ms for same computation (50x faster!)

#### Step 2: BatchNorm GPU Kernel
```cuda
// Normalize: y = (x - mean) / sqrt(var + eps) * gamma + beta

__global__ void batchnorm_kernel(
    float* data, const float* mean, const float* var,
    const float* gamma, const float* beta,
    int batch, int channels, int height, int width, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    
    if (idx < total) {
        int c = (idx / (height * width)) % channels;
        float normalized = (data[idx] - mean[c]) / sqrt(var[c] + eps);
        data[idx] = normalized * gamma[c] + beta[c];
    }
}
```

### Phase 2 Deliverables

| Component | Status | Details |
|-----------|--------|---------|
| Conv2D kernel | Implement | CUDA kernel + CPU wrapper |
| BatchNorm kernel | Implement | In-place normalization |
| Integration | Implement | Call GPU kernels from Layer classes |
| Testing | Create | Unit tests for each kernel |
| Benchmarking | Create | Compare CPU vs GPU |

### Estimated Timeline
- **Duration:** 1-2 weeks
- **Effort:** Moderate (kernels are straightforward)
- **Complexity:** Medium (memory layout, thread scheduling)

---

## Phase 3: Complete Layer GPU Implementation

### Goal
Implement GPU kernels for remaining layers:
- **ReLU** - Activation (simple)
- **MaxPool** - Pooling (moderate)
- **Linear** - Fully connected (complex)
- **AdaptiveAvgPool** - Global pooling (simple)
- **BasicBlock** - Integration layer (no new kernels needed)

### What Gets GPU Acceleration

```
Original ResNet18:
├─ Conv1 (7x7, stride 2)         → GPU kernel
├─ BatchNorm1                     → GPU kernel
├─ ReLU                           → GPU kernel
├─ MaxPool (3x3, stride 2)        → GPU kernel
├─ Layer1 (2x BasicBlock)         → GPU kernels for all ops
├─ Layer2 (2x BasicBlock)         → GPU kernels for all ops
├─ Layer3 (2x BasicBlock)         → GPU kernels for all ops
├─ Layer4 (2x BasicBlock)         → GPU kernels for all ops
├─ AdaptiveAvgPool                → GPU kernel
└─ FC (Linear)                    → GPU kernel or cuBLAS
```

### New Kernels Needed

#### ReLU Kernel
```cuda
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);  // max(0, x)
    }
}
```
**GPU Time:** ~0.1 ms (vs ~1 ms on CPU) = 10x faster

#### MaxPool Kernel
```cuda
__global__ void maxpool2d_kernel(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int kernel_size, int stride, int out_h, int out_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    
    if (idx < total) {
        // Compute which input window this output comes from
        // Find max value in that window
    }
}
```
**GPU Time:** ~0.5 ms (vs ~5 ms on CPU) = 10x faster

#### Linear Kernel (Option A: Custom)
```cuda
__global__ void linear_kernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_features, int out_features) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch && col < out_features) {
        float sum = bias[col];
        for (int i = 0; i < in_features; ++i) {
            sum += input[row * in_features + i] * weights[col * in_features + i];
        }
        output[row * out_features + col] = sum;
    }
}
```

#### Linear Kernel (Option B: Use cuBLAS - Better!)
```cuda
// ResNet's FC layer is just matrix multiplication + bias
// cuBLAS does this 100x better than custom kernel

cublasHandle_t handle;
cublasCreate(&handle);

// output = input @ weights^T + bias
cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    out_features, batch, in_features,
    1.0f,
    weights_gpu, in_features,
    input_gpu, in_features,
    0.0f,
    output_gpu, out_features);

// Add bias
add_bias_kernel<<<blocks, threads>>>(output_gpu, bias_gpu, batch, out_features);

cublasDestroy(handle);
```

### Performance Expectations

| Layer | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| Conv2D | 400 ms | 8 ms | **50x** |
| BatchNorm | 80 ms | 1 ms | **80x** |
| ReLU | 20 ms | 0.2 ms | **100x** |
| MaxPool | 30 ms | 0.3 ms | **100x** |
| Linear | 50 ms | 1 ms | **50x** |
| **Total Forward** | **~580 ms** | **~10 ms** | **~58x** |

### Phase 3 Deliverables
- GPU kernels for all remaining layers
- Integration with Layer classes
- Comprehensive testing (each kernel verified)
- Performance profiling

### Estimated Timeline
- **Duration:** 2-3 weeks
- **Effort:** Moderate-High
- **Complexity:** Medium (mix of simple and complex kernels)

---

## Phase 4: Optimization & Batching

### Goal
Maximize GPU utilization beyond single-image forward pass.

### Optimization Strategies

#### Strategy 1: Kernel Fusion
**Problem:** Each kernel launch has overhead (~5 microseconds)

**Solution:** Combine operations
```cuda
// Before: 3 separate kernels
output = conv2d(input);     // Kernel 1 launch
output = batchnorm(output); // Kernel 2 launch
output = relu(output);      // Kernel 3 launch
// Total overhead: 15 microseconds

// After: Fused kernel
output = conv2d_bn_relu_fused(input);  // 1 kernel launch
// Total overhead: 5 microseconds (3x less overhead)
```

**Expected Gain:** 5-10% performance improvement

#### Strategy 2: Batch Processing
**Problem:** Processing 1 image at a time wastes GPU capacity

**Current:**
```cuda
// Process 1 image
Tensor image({1, 3, 224, 224});
resnet.forward(image);  // GPU only 10% utilized
```

**Improved:**
```cuda
// Process 4 images in parallel
Tensor batch({4, 3, 224, 224});
resnet.forward(batch);  // GPU 40% utilized, 3.5x faster!

// Or 32 images
Tensor batch({32, 3, 224, 224});
resnet.forward(batch);  // GPU 90% utilized, 10x faster!
```

**Expected Gain:** 3-10x depending on batch size

#### Strategy 3: Memory Optimization
**Problem:** Redundant data copies between layers

**Solution:** Keep data on GPU between layers
```cuda
// Before: Copy after each layer
Tensor t = conv1(input);  // GPU→CPU copy
t = bn1(t);               // CPU→GPU copy
t = relu(t);              // GPU→CPU copy
t = maxpool(t);           // CPU→GPU copy
// Result: 4 unnecessary copies!

// After: Stream processing
t = input;
t.to_gpu();               // 1 copy
while (t.on_gpu()) {
    t = conv1(t);         // Stays on GPU
    t = bn1(t);           // Stays on GPU
    t = relu(t);          // Stays on GPU
    t = maxpool(t);       // Stays on GPU
}
t.to_cpu();               // 1 copy
// Result: Only 2 copies total! (H2D + D2H)
```

**Expected Gain:** 20-40% depending on data size

#### Strategy 4: Asynchronous Operations
```cuda
// Overlap computation with transfers
cudaStream_t stream;
cudaStreamCreate(&stream);

// Copy first batch while computing previous batch
cudaMemcpyAsync(gpu_data, cpu_data, size, cudaMemcpyHostToDevice, stream);
conv2d_kernel<<<blocks, threads, 0, stream>>>(gpu_data, ...);
cudaMemcpyAsync(cpu_output, gpu_output, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

**Expected Gain:** 10-20% in certain scenarios

### Phase 4 Deliverables
- Kernel fusion implementations
- Multi-image batch processing support
- Memory transfer optimization
- Asynchronous kernel launches
- Performance profiling and tuning

### Estimated Timeline
- **Duration:** 1-2 weeks
- **Effort:** Moderate (mostly engineering)
- **Complexity:** Medium-High (synchronization issues possible)

---

## Phase 5: Benchmarking & Validation

### Goal
Comprehensive performance analysis and comparison across all implementations.

### Benchmark Strategy

#### Test 1: Single Image Forward Pass
```
Test: ResNet18 inference on 1 image
Input: {1, 3, 224, 224}

Implementations:
1. CPU-only (original)        → baseline
2. GPU Phase 1 (transfer test)  → expected: 10x slower
3. GPU Phase 2 (Conv + BN)      → expected: 5-10x faster
4. GPU Phase 3 (All layers)     → expected: 20-50x faster
5. GPU Phase 4 (Optimized)      → expected: 30-100x faster
```

#### Test 2: Batch Processing
```
Test: ResNet18 inference on N images
Batch sizes: 1, 4, 8, 16, 32, 64

Metric: Throughput (images/second)
CPU:  ~2 images/sec
GPU:  ~200-500 images/sec (depending on batch)
```

#### Test 3: Memory Analysis
```
Metric: Peak GPU memory usage
Phase 1: 1 MB (for one tensor)
Phase 3: 10 MB (model weights + activations)
Phase 4: 100+ MB (for batch processing)
```

#### Test 4: Accuracy Verification
```
Requirement: GPU results == CPU results (within floating point error)

Tolerance: 1e-5 (single precision)
Compare: Last layer output for same input
```

### Benchmark Code Structure
```cpp
// Pseudo-code for benchmark
for (int batch_size : {1, 4, 8, 16, 32}) {
    Tensor input({batch_size, 3, 224, 224});
    
    // Test CPU
    auto t0 = chrono::now();
    Tensor output_cpu = resnet_cpu.forward(input);
    auto t1 = chrono::now();
    
    // Test GPU
    t0 = chrono::now();
    Tensor output_gpu = resnet_gpu.forward(input);
    auto t2 = chrono::now();
    
    // Verify correctness
    assert(outputs_match(output_cpu, output_gpu));
    
    // Report
    printf("Batch %d: CPU %.2f ms, GPU %.2f ms, Speedup: %.1fx\n",
           batch_size,
           duration_ms(t0, t1),
           duration_ms(t0, t2),
           duration_ms(t0, t1) / duration_ms(t0, t2));
}
```

### Phase 5 Deliverables
- Comprehensive benchmark suite
- Performance graphs and analysis
- Accuracy verification tests
- Final report with results
- Documentation of findings

### Estimated Timeline
- **Duration:** 1 week
- **Effort:** Low (mostly running tests)
- **Complexity:** Low (testing framework)

---

## Overall Timeline & Effort

| Phase | Duration | Effort | Complexity | Status |
|-------|----------|--------|-----------|--------|
| 1 | ✅ Done | ✅ Done | ✅ Done | **COMPLETE** |
| 2 | 1-2 weeks | Moderate | Medium | Next |
| 3 | 2-3 weeks | Moderate-High | Medium | After Phase 2 |
| 4 | 1-2 weeks | Moderate | Medium-High | After Phase 3 |
| 5 | 1 week | Low | Low | Final |
| **Total** | **~6-9 weeks** | **High** | **Medium** | **In Progress** |

---

## Expected Final Results

### Performance
```
Single-image inference:
  CPU:  500 ms
  GPU:  5-10 ms
  Speedup: 50-100x

Batch inference (32 images):
  CPU:  16 seconds
  GPU:  200-400 ms
  Throughput: 80-160 images/sec
```

### Memory
```
GPU VRAM:
  Phase 1: 1 MB (test tensors)
  Phase 3: 50 MB (model + one batch)
  Phase 4: 500 MB (model + multi-batch)
```

### Accuracy
```
Classification accuracy:
  CPU: 69.76% (ImageNet validation)
  GPU: 69.76% (same - bit-exact or within 1e-5)
```

---

## Summary

- **Phase 1:** ✅ Infrastructure (DONE)
- **Phase 2:** GPU kernels for Conv2D + BatchNorm (30-40% performance gain)
- **Phase 3:** GPU kernels for all layers (50-100x overall speedup)
- **Phase 4:** Optimization + batching (additional 2-5x improvement)
- **Phase 5:** Benchmarking and validation (proof of performance)

**Total Expected Improvement:** 100-500x for batch processing, 50-100x for single images
