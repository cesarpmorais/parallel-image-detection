# CUDA Implementation v1 Guide

This document outlines the key differences and new implementations introduced in the CUDA-based version of the ResNet-18 model (`cpp_cuda/`) compared to the CPU-based implementation (`cpp/`).

## Overview
The CUDA implementation (`cpp_cuda/`) accelerates ResNet-18 inference by leveraging GPU parallelism. Key changes include:
- GPU-specific kernels for computational layers.
- A GPU-aware `Tensor` class for efficient memory management.
- Modifications to the build system to support CUDA.

## Key Changes

### 1. GPU Kernels
- **Conv2D**: A naive CUDA kernel was implemented to perform convolution operations in parallel on the GPU.
- **BatchNorm**: A CUDA kernel was added to compute batch normalization efficiently on the GPU.
- These kernels are invoked from the respective layer classes (`Conv2D`, `BatchNorm2D`) in their `forward` methods.

### 2. GPU-Aware Tensor Class
- The `Tensor` class was extended to support GPU operations:
  - **Memory Management**: Added methods to allocate, copy, and free GPU memory.
  - **Data Transfers**: Added methods to transfer data between host (CPU) and device (GPU).
  - **Compatibility Loaders**: Implemented `load_from_bin_compat` to handle both raw float arrays and headered tensor files.

### 3. Build System
- **CMake Configuration**:
  - Added CUDA support with `find_package(CUDA REQUIRED)`.
  - Set the target architecture to Ada (86) for compatibility with modern GPUs.
  - Excluded test files (e.g., `main.cu`) to avoid duplicate `main` definitions.

### 4. Main Driver Modifications
- The `main.cpp` file was updated to:
  - Use the compatibility loader for input files.
  - Remove unnecessary output file writes for intermediate tensors.

### 5. Benchmarking Script
- The benchmarking script (`benchmark_parallel_vs_sequential.py`) was updated to:
  - Include the CUDA implementation in the comparison.
  - Save outputs in separate directories for CPU, OpenMP, and CUDA runs.
  - Compute per-input correctness metrics (e.g., MAE, max-diff).

## Summary
The CUDA implementation introduces GPU acceleration for key layers of ResNet-18, significantly improving inference speed. While the implementation is functional, further optimization of the CUDA kernels is possible to achieve even greater performance gains.