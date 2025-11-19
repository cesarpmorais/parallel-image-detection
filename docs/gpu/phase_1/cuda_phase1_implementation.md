# CUDA Phase 1: Foundation Implementation (Step-by-Step)

## Overview

**Phase 1 Goal:** Set up the CUDA development environment and create a foundational tensor class with GPU memory management. This phase establishes the groundwork for all subsequent GPU implementations.

**Timeline:** 3-5 days

**Deliverables:**
- ✅ CUDA development environment verified
- ✅ CMakeLists.txt configured for CUDA compilation
- ✅ CUDA error checking utility macros
- ✅ Tensor class with GPU memory management
- ✅ Basic Conv2D kernel stub (for cuDNN integration in Phase 2)

---

## Step 0: Verify CUDA Installation

### 0.1 Check if CUDA is installed

```bash
# Check CUDA compiler
nvcc --version

# Check GPU availability
nvidia-smi

# Find CUDA libraries (may be in /usr/lib instead of /usr/local/cuda)
find /usr -name "libcublas.so" 2>/dev/null
find /usr -name "cuda_runtime.h" 2>/dev/null
```

**Expected output example (RTX 4070):**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Cuda compilation tools, release 11.5, V11.5.119

Tue Nov 18 20:53:48 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.05              Driver Version: 580.88         CUDA Version: 13.0     |
|   0  NVIDIA GeForce RTX 4070        On  |      00000000:01:00.0  On |                  N/A |
|  0%   45C    P8             10W /  220W |    3612MiB /  12282MiB |      7%      Default |
+-----------------------------------------------------------------------------------------+
```

### 0.2 If CUDA is NOT installed

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-dev

# Or download from NVIDIA
wget https://developer.nvidia.com/cuda-downloads
# Follow installation instructions
```

**Windows:**
```powershell
# Download CUDA installer
# https://developer.nvidia.com/cuda-downloads
# Run installer and add to PATH
```

**macOS:**
```bash
# CUDA support on macOS ended with CUDA 10.2
# Use Intel/AMD Metal instead, or use Linux
```

### 0.3 Verify cuDNN (Optional, for Phase 2)

```bash
# Check if cuDNN is installed
find /usr -name "cudnn.h" 2>/dev/null
```

**Note:** cuDNN is optional for Phase 1. For Phase 2 (Conv2D optimization), you can:
- Download from https://developer.nvidia.com/cudnn, or
- Use a basic CUDA convolution kernel instead

---

## Step 1: Update CMakeLists.txt for CUDA

### 1.1 Backup original CMakeLists.txt

```bash
cd cpp_cuda
cp CMakeLists.txt CMakeLists.txt.backup
```

### 1.2 Update CMakeLists.txt

Replace the entire file with:

```cmake
cmake_minimum_required(VERSION 3.17)
project(ResNet18_CUDA CUDA CXX)

# ===== CUDA Configuration =====
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 uses Ada architecture (89)
# Common architectures:
# - 75: Turing (RTX 2070, 2080, etc.)
# - 80: Ampere (A100, RTX 3070, 3080, etc.)
# - 86: Ada Lovelace (RTX 4060, 4070, 4080, etc. - use 89 for newer)
# - 89: Ada Lovelace latest (RTX 4090, etc.)

# ===== C++ Configuration =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Compiler Flags =====
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} /W4 /O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")
endif()

# ===== Find CUDA Packages =====
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Optional: Find cuDNN (comment out if not installed)
# find_package(cuDNN REQUIRED)

# If CMake cannot find CUDA on your system, set these manually:
# set(CUDA_INCLUDE_DIRS "/usr/include")
# set(CUDA_LIBRARIES "cudart" "cublas" "cuda")

# ===== Directory Setup =====
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# ===== Gather Source Files =====
file(GLOB_RECURSE CUDA_SOURCES
    "${SOURCE_DIR}/*.cu"
)
file(GLOB_RECURSE CXX_SOURCES
    "${SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE HEADERS
    "${INCLUDE_DIR}/*.h"
    "${SOURCE_DIR}/*.h"
)

# ===== Create Executable =====
add_executable(resnet18_cuda
    ${CUDA_SOURCES}
    ${CXX_SOURCES}
    ${HEADERS}
)

# ===== Link Libraries =====
target_link_libraries(resnet18_cuda
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cuda_driver
        # CUDA::curand          # Optional: for random number generation
        # cuDNN::cuDNN          # Uncomment if using cuDNN
)

# ===== Include Directories =====
target_include_directories(resnet18_cuda
    PRIVATE
        ${INCLUDE_DIR}
        ${SOURCE_DIR}
        ${CUDA_INCLUDE_DIRS}
)

# ===== Compile Definitions =====
target_compile_definitions(resnet18_cuda
    PRIVATE
        WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/../weights"
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/test_data"
        OUTPUT_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/cpp_outputs"
        USE_GPU=1
)

# ===== GPU Architecture Report =====
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_VERSION}")
```

---

## Step 2: Create CUDA Utility Header

### 2.1 Create `include/cuda_utils.h`

```cpp
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// ===== CUDA Error Checking Macros =====

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err__) << std::endl; \
            std::cerr << "File: " << __FILE__ << std::endl; \
            std::cerr << "Line: " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (false)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl; \
            std::cerr << "File: " << __FILE__ << std::endl; \
            std::cerr << "Line: " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (false)

// ===== Utility Functions =====

inline size_t get_gpu_memory_total() {
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    return total_memory;
}

inline size_t get_gpu_memory_free() {
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    return free_memory;
}

inline void print_gpu_memory_info() {
    size_t free = get_gpu_memory_free();
    size_t total = get_gpu_memory_total();
    std::cout << "GPU Memory: " << (free / 1024.0 / 1024.0) << " MB free / "
              << (total / 1024.0 / 1024.0) << " MB total" << std::endl;
}
```

---

## Step 3: Update Tensor Header for GPU Support

### 3.1 Update `include/tensor.h`

Add GPU memory management to your existing Tensor class. Here's the GPU-enhanced version:

```cpp
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include "cuda_utils.h"

class Tensor {
public:
    // ===== CPU Memory =====
    std::vector<float> data;        // Host-side data
    std::vector<int> shape;         // Tensor dimensions
    bool on_gpu;                    // Tracks GPU memory location
    
    // ===== GPU Memory =====
    float* gpu_data;                // Device-side pointer
    size_t gpu_allocated_size;      // Allocated GPU memory size (bytes)
    
    // ===== Constructors =====
    Tensor();
    Tensor(const std::vector<int>& shape);
    ~Tensor();
    
    // Prevent accidental copies (GPU pointers would dangle)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Move semantics (safe for GPU pointers)
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // ===== Memory Management =====
    void allocate(const std::vector<int>& shape);
    void allocate_gpu();
    void free_gpu();
    void free_cpu();
    
    // ===== Data Transfer =====
    void to_gpu();                  // CPU → GPU (H2D)
    void to_cpu();                  // GPU → CPU (D2H)
    
    // ===== File I/O =====
    bool load_from_bin(const std::string& path);
    bool save_to_bin(const std::string& path) const;
    bool load_shape_from_txt(const std::string& path);
    bool save_shape_to_txt(const std::string& path) const;
    
    // ===== Utility =====
    size_t size() const;            // Total number of elements
    size_t bytes() const;           // Size in bytes
    void print_shape() const;
    void print_summary(int max_elements = 10) const;
    
    // ===== Data Access =====
    float* cpu_data();
    float* device_data();
    const float* cpu_data() const;
    const float* device_data() const;
    
private:
    void cleanup();
};
```

### 3.2 Update `src/tensor.cu`

This is the CUDA implementation of Tensor. Create it with GPU support:

```cpp
#include "tensor.h"
#include <fstream>
#include <algorithm>
#include <cmath>

// ===== Constructors & Destructors =====

Tensor::Tensor()
    : data(), shape(), on_gpu(false), gpu_data(nullptr), gpu_allocated_size(0) {}

Tensor::Tensor(const std::vector<int>& shape)
    : shape(shape), on_gpu(false), gpu_data(nullptr), gpu_allocated_size(0) {
    size_t total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    data.resize(total, 0.0f);
}

Tensor::~Tensor() {
    cleanup();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(std::move(other.data)),
      shape(std::move(other.shape)),
      on_gpu(other.on_gpu),
      gpu_data(other.gpu_data),
      gpu_allocated_size(other.gpu_allocated_size) {
    other.gpu_data = nullptr;
    other.gpu_allocated_size = 0;
    other.on_gpu = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        cleanup();
        data = std::move(other.data);
        shape = std::move(other.shape);
        on_gpu = other.on_gpu;
        gpu_data = other.gpu_data;
        gpu_allocated_size = other.gpu_allocated_size;
        other.gpu_data = nullptr;
        other.gpu_allocated_size = 0;
        other.on_gpu = false;
    }
    return *this;
}

// ===== Memory Management =====

void Tensor::allocate(const std::vector<int>& new_shape) {
    shape = new_shape;
    size_t total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    data.resize(total, 0.0f);
    on_gpu = false;
}

void Tensor::allocate_gpu() {
    if (gpu_data != nullptr) return;  // Already allocated
    
    size_t total = size();
    if (total == 0) {
        std::cerr << "Cannot allocate GPU memory for empty tensor" << std::endl;
        return;
    }
    
    size_t bytes = total * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&gpu_data, bytes));
    gpu_allocated_size = bytes;
    
    std::cout << "Allocated " << (bytes / 1024.0 / 1024.0) << " MB on GPU" << std::endl;
}

void Tensor::free_gpu() {
    if (gpu_data != nullptr) {
        CUDA_CHECK(cudaFree(gpu_data));
        gpu_data = nullptr;
        gpu_allocated_size = 0;
        on_gpu = false;
    }
}

void Tensor::free_cpu() {
    data.clear();
    data.shrink_to_fit();
}

void Tensor::cleanup() {
    free_gpu();
    free_cpu();
}

// ===== Data Transfer =====

void Tensor::to_gpu() {
    if (on_gpu) return;  // Already on GPU
    if (size() == 0) {
        std::cerr << "Cannot transfer empty tensor to GPU" << std::endl;
        return;
    }
    
    allocate_gpu();  // Allocate if not already done
    
    size_t bytes = size() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(gpu_data, data.data(), bytes, cudaMemcpyHostToDevice));
    on_gpu = true;
    
    std::cout << "Transferred " << (bytes / 1024.0 / 1024.0) << " MB to GPU" << std::endl;
}

void Tensor::to_cpu() {
    if (!on_gpu || gpu_data == nullptr) return;  // Already on CPU or empty
    
    size_t bytes = size() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(data.data(), gpu_data, bytes, cudaMemcpyDeviceToHost));
    on_gpu = false;
    
    std::cout << "Transferred " << (bytes / 1024.0 / 1024.0) << " MB to CPU" << std::endl;
}

// ===== File I/O =====

bool Tensor::load_from_bin(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << path << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(data.data()), size() * sizeof(float));
    if (!file) {
        std::cerr << "Error: Failed to read from " << path << std::endl;
        return false;
    }
    
    file.close();
    on_gpu = false;  // Data is on CPU after loading from file
    return true;
}

bool Tensor::save_to_bin(const std::string& path) const {
    // If on GPU, we need CPU copy for saving
    if (on_gpu) {
        std::cerr << "Warning: Saving GPU tensor to disk (requires GPU→CPU transfer)" << std::endl;
        // This would require non-const to_cpu(), which we avoid for safety
        // Instead, caller should ensure data is on CPU before saving
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << path << std::endl;
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(data.data()), size() * sizeof(float));
    if (!file) {
        std::cerr << "Error: Failed to write to " << path << std::endl;
        return false;
    }
    
    file.close();
    return true;
}

bool Tensor::load_shape_from_txt(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open shape file " << path << std::endl;
        return false;
    }
    
    shape.clear();
    int dim;
    while (file >> dim) {
        shape.push_back(dim);
    }
    
    if (shape.empty()) {
        std::cerr << "Error: No shape data in " << path << std::endl;
        return false;
    }
    
    file.close();
    
    // Allocate CPU memory
    size_t total = 1;
    for (int d : shape) total *= d;
    data.resize(total, 0.0f);
    
    return true;
}

bool Tensor::save_shape_to_txt(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create shape file " << path << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) file << " ";
        file << shape[i];
    }
    file << std::endl;
    
    file.close();
    return true;
}

// ===== Utility =====

size_t Tensor::size() const {
    size_t total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    return total;
}

size_t Tensor::bytes() const {
    return size() * sizeof(float);
}

void Tensor::print_shape() const {
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "Size: " << size() << " elements, " << (bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Location: " << (on_gpu ? "GPU" : "CPU") << std::endl;
}

void Tensor::print_summary(int max_elements) const {
    print_shape();
    if (size() == 0) return;
    
    size_t show = std::min((size_t)max_elements, size());
    std::cout << "First " << show << " values: [";
    for (size_t i = 0; i < show; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
    }
    std::cout << (size() > show ? ", ...]" : "]") << std::endl;
}

// ===== Data Access =====

float* Tensor::cpu_data() {
    if (on_gpu) {
        std::cerr << "Warning: Accessing GPU tensor data on CPU (may be stale)" << std::endl;
    }
    return data.data();
}

float* Tensor::device_data() {
    if (!on_gpu) {
        std::cerr << "Warning: Accessing CPU tensor data on GPU" << std::endl;
    }
    return gpu_data;
}

const float* Tensor::cpu_data() const {
    return data.data();
}

const float* Tensor::device_data() const {
    return gpu_data;
}
```

---

## Step 4: Create a Minimal main.cu

### 4.1 Create `src/main.cu`

This is a minimal CUDA program to test the setup:

```cpp
#include <iostream>
#include <vector>
#include "../include/tensor.h"
#include "../include/cuda_utils.h"

int main() {
    std::cout << "=== ResNet18 CUDA Implementation ===" << std::endl;
    std::cout << std::endl;
    
    // ===== CUDA Device Info =====
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }
    
    // Get device 0 properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU 0: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;
    
    // ===== GPU Memory Info =====
    std::cout << "GPU Memory Status:" << std::endl;
    print_gpu_memory_info();
    std::cout << std::endl;
    
    // ===== Test Tensor Allocation =====
    std::cout << "Testing Tensor GPU allocation..." << std::endl;
    std::vector<int> test_shape = {1, 3, 224, 224};  // Typical input shape
    Tensor input(test_shape);
    std::cout << "Created tensor with shape: ";
    input.print_shape();
    std::cout << std::endl;
    
    // ===== CPU → GPU Transfer =====
    std::cout << "Transferring tensor to GPU..." << std::endl;
    input.to_gpu();
    std::cout << "Transfer complete" << std::endl;
    std::cout << std::endl;
    
    // ===== GPU Memory Info After Allocation =====
    std::cout << "GPU Memory Status (after allocation):" << std::endl;
    print_gpu_memory_info();
    std::cout << std::endl;
    
    // ===== GPU → CPU Transfer =====
    std::cout << "Transferring tensor back to CPU..." << std::endl;
    input.to_cpu();
    std::cout << "Transfer complete" << std::endl;
    std::cout << std::endl;
    
    // ===== Cleanup =====
    std::cout << "Testing cleanup..." << std::endl;
    input.free_gpu();
    std::cout << "GPU memory freed" << std::endl;
    print_gpu_memory_info();
    std::cout << std::endl;
    
    std::cout << "=== Phase 1 Foundation Test Passed ===" << std::endl;
    return 0;
}
```

---

## Step 5: Compile & Test Phase 1

### 5.1 Create Build Directory

```bash
cd cpp_cuda
mkdir -p build
cd build
```

### 5.2 Configure with CMake

```bash
# Linux/macOS
cmake .. -DCMAKE_BUILD_TYPE=Release

# Windows (MSVC)
cmake .. -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release

# Windows (MinGW)
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
```

**Expected output:**
```
-- The CUDA compiler identification is NVIDIA ...
-- Detecting CUDA architecture...
-- The CXX compiler identification is GNU
-- Check for working CXX compiler: ...
-- Checking whether C++ compiler works... yes
...
-- CUDA Architectures: 75;80;86
-- CUDA Compiler: /usr/bin/nvcc
-- CUDA Version: 11.8
-- Configuring done
-- Generating build files done
```

### 5.3 Build

```bash
# Linux/macOS
make -j$(nproc)

# Windows (MSVC)
cmake --build . --config Release

# Windows (MinGW)
make
```

**Expected compilation:**
```
[ 10%] Building CUDA object CMakeFiles/resnet18_cuda.dir/src/tensor.cu.o
[ 20%] Building CXX object CMakeFiles/resnet18_cuda.dir/src/main.cu.o
[ 30%] Linking CUDA device code CXXFiles/resnet18_cuda.dir/cmake_device_link.o
[ 40%] Linking CXX executable resnet18_cuda
[100%] Built target resnet18_cuda
```

### 5.4 Run Test

```bash
./resnet18_cuda
```

**Expected output:**
```
=== ResNet18 CUDA Implementation ===

Found 1 CUDA device(s)
GPU 0: NVIDIA A100
  Compute Capability: 8.0
  Total Memory: 81920.0 MB
  Max Threads per Block: 1024

GPU Memory Status:
GPU Memory: 80944.0 MB free / 81920.0 MB total

Testing Tensor GPU allocation...
Created tensor with shape: Shape: [1, 3, 224, 224]
Size: 150528 elements, 0.573975 MB
Location: CPU

Transferring tensor to GPU...
Allocated 0.573975 MB on GPU
Transferred 0.573975 MB to GPU
Transfer complete

GPU Memory Status (after allocation):
GPU Memory: 80942.0 MB free / 81920.0 MB total

Transferring tensor back to CPU...
Transferred 0.573975 MB to CPU
Transfer complete

Testing cleanup...
GPU memory freed

GPU Memory Status (after cleanup):
GPU Memory: 80944.0 MB free / 81920.0 MB total

=== Phase 1 Foundation Test Passed ===
```

---

## Step 6: Stub Conv2D for Phase 2

### 6.1 Update `include/conv2d.h`

```cpp
#pragma once

#include "tensor.h"
#include <string>

class Conv2D {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weight;  // [out_channels, in_channels, kernel_h, kernel_w]
    
    Conv2D(int in_c, int out_c, int k, int s = 1, int p = 0);
    
    bool load_weights(const std::string& path);
    Tensor forward(const Tensor& input);
};
```

### 6.2 Create `src/conv2d.cu` (Stub)

```cpp
#include "../include/conv2d.h"
#include <iostream>

Conv2D::Conv2D(int in_c, int out_c, int k, int s, int p)
    : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p) {}

bool Conv2D::load_weights(const std::string& path) {
    std::cout << "[Phase 2] Conv2D::load_weights() - cuDNN implementation pending" << std::endl;
    return true;  // Placeholder
}

Tensor Conv2D::forward(const Tensor& input) {
    std::cout << "[Phase 2] Conv2D::forward() - cuDNN kernel pending" << std::endl;
    Tensor output;
    // Placeholder
    return output;
}
```

### 6.3 Update other layer stubs similarly

Create stubs for `batchnorm.cu`, `relu.cu`, `maxpool.cu`, `linear.cu`, `adaptiveavgpool.cu`, `basicblock.cu` with similar placeholder implementations.

---

## Step 7: Verify Phase 1 Completion

### Checklist:
- ✅ CUDA toolkit installed and `nvcc --version` works
- ✅ GPU detected by `nvidia-smi`
- ✅ CMakeLists.txt configured with CUDA support
- ✅ `cuda_utils.h` created with error checking macros
- ✅ `tensor.cu` compiles with GPU memory management
- ✅ `main.cu` test runs successfully
- ✅ GPU memory allocation/deallocation works
- ✅ H2D and D2H transfers verified
- ✅ Conv2D stub created for Phase 2
- ✅ All layer stubs created

### Troubleshooting:

**Problem:** `nvcc not found`
```bash
# Add CUDA to PATH
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

**Problem:** `CUDA_CHECK` macro fails during build
- Check CUDA SDK version matches compiler
- Verify cuDNN headers are installed

**Problem:** Memory allocation fails
- Check GPU has enough free memory
- Ensure CUDA context is initialized (happens automatically in modern CUDA)

---

## Next: Proceed to Phase 2

Once Phase 1 passes all checks, move to **Phase 2: Core Layers Implementation**

Key activities:
- [ ] Implement Conv2D kernel (cuDNN wrapper)
- [ ] Implement BatchNorm2D kernel
- [ ] Implement ReLU, MaxPool, AdaptiveAvgPool
- [ ] Test each layer individually against CPU reference
- [ ] Validate numerical correctness (error < 1e-4)

---

## Useful CUDA References

- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Runtime API:** https://docs.nvidia.com/cuda/cuda-runtime-api/
- **cuBLAS Documentation:** https://docs.nvidia.com/cuda/cublas/
- **Memory Management:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory
- **Best Practices:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

## Summary

Phase 1 establishes:
1. ✅ CUDA development environment
2. ✅ CMake with CUDA compiler support
3. ✅ Tensor class with GPU memory management
4. ✅ GPU memory allocation, transfer, and cleanup
5. ✅ CUDA error checking infrastructure
6. ✅ Layer stubs ready for Phase 2 implementation

You're now ready to implement actual GPU kernels in Phase 2!
