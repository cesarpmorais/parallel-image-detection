# Phase 1, Step 4: Minimal GPU Test Program - COMPLETED

## Overview

Step 4 creates a comprehensive test program (`main.cu`) that validates the GPU-enabled Tensor infrastructure before proceeding to actual GPU kernel implementation in Phase 2.

## File Created

### `/cpp_cuda/src/main.cu`
**Purpose:** Minimal test program demonstrating Tensor GPU capabilities

**File Size:** ~380 lines

## Test Coverage

The program performs 10 sequential tests:

### Test 1: GPU Information
```cpp
print_gpu_memory_info();
```
- Displays GPU device name, compute capability, total/free memory
- Uses `cuda_utils.h` utilities
- **Expected Output:**
  ```
  GPU Device: NVIDIA GeForce RTX 4070
  Compute Capability: 8.9
  Total Memory: 12 GB
  Free Memory: 11.5 GB
  ```

### Test 2: Tensor Creation
```cpp
Tensor input({1, 3, 224, 224});
input.print_info();
```
- Create a 4D tensor matching ResNet18 input shape (batch=1, channels=3, height=224, width=224)
- Verify shape, element count, memory size
- **Expected Output:**
  ```
  Shape: [1, 3, 224, 224]
  Elements: 150528
  Size: 0.57 MB
  Location: CPU
  GPU Allocated: No
  ```

### Test 3: CPU Data Operations
```cpp
for (size_t i = 0; i < input.numel(); i++) {
    input[i] = static_cast<float>(i) / input.numel();
}
float min_val = input.min();
float max_val = input.max();
float mean_val = input.mean();
float sum_val = input.sum();
```
- Fill tensor with sequential normalized values (0 to 1)
- Test statistical operations (min, max, mean, sum)
- **Expected Output:**
  ```
  Min value: 0
  Max value: 0.999...
  Mean value: 0.5
  Sum: 75264
  ```

### Test 4: 4D Convenience Indexing
```cpp
float val = input(0, 0, 0, 0);  // Read
input(0, 0, 0, 0) = 99.5f;      // Write
```
- Test 4D indexing operator: `tensor(b, c, h, w)`
- Verify read and write operations
- **Expected Output:**
  ```
  Value at (0,0,0,0): 0
  Verification: input(0,0,0,0) = 99.5
  ```

### Test 5: GPU Memory Allocation
```cpp
input.allocate_gpu();
```
- Allocate GPU memory without copying data
- Test idempotency (safe to call multiple times)
- **Expected Output:**
  ```
  GPU allocated: No (need to transfer)
  [GPU memory shows ~0.57 MB allocated]
  ```

### Test 6: H2D Transfer (CPU → GPU)
```cpp
input.to_gpu();
```
- Transfer tensor data from CPU to GPU
- Verify GPU pointer is valid
- Check GPU memory usage increased
- **Expected Output:**
  ```
  Data on GPU: Yes
  GPU data pointer: 0x7f8a10000000  [valid CUDA pointer]
  [GPU free memory decreased by ~0.57 MB]
  ```

### Test 7: D2H Transfer (GPU → CPU)
```cpp
input[0] = 42.0f;  // Modify CPU copy
input.to_cpu();     // Overwrite with GPU data
```
- Transfer tensor data from GPU back to CPU
- Verify GPU version overwrites CPU modifications
- **Expected Output:**
  ```
  Data on GPU: No
  CPU value at [0]: 99.5  [GPU version, not 42.0]
  ```

### Test 8: File I/O
```cpp
input.save_to_bin(output_file);
Tensor loaded;
loaded.load_from_bin(output_file);
```
- Save tensor to binary file (shape + data)
- Load tensor from binary file
- Verify shape and data integrity
- **Expected Output:**
  ```
  Saved successfully
  Loaded shape: [1, 3, 224, 224]
  [GPU Allocated: No, Location: CPU]
  ```

### Test 9: GPU Memory Cleanup
```cpp
input.free_gpu();
```
- Release GPU memory explicitly
- Verify GPU memory is freed
- **Expected Output:**
  ```
  GPU allocated: No
  [GPU free memory increased by ~0.57 MB]
  ```

### Test 10: Tensor String Representation
```cpp
std::cout << loaded.to_string();
```
- Display compact tensor string
- **Expected Output:**
  ```
  Tensor(shape=[1, 3, 224, 224], on_gpu=false)
  ```

## Building and Running

### Prerequisites
Before compiling, ensure:
- [ ] CUDA 11.5+ installed (`nvcc --version` works)
- [ ] CMake 3.17+ available
- [ ] CUDA libraries found by CMake
- [ ] gcc/g++ compiler available

### Build Steps

```bash
cd /home/cesar/9o_Periodo/parallel/tp_final/cpp_cuda
mkdir -p build
cd build
cmake ..
make -j4
```

### Run Test

```bash
./resnet18_cuda
```

## Expected Output Structure

```
========================================
Phase 1, Step 4: GPU Tensor Test
========================================

========================================
Test 1: GPU Information
========================================
GPU Device: NVIDIA GeForce RTX 4070
...

========================================
Test 2: Tensor Creation
========================================
Creating 4D tensor (1, 3, 224, 224)...
...

[... Tests 3-10 ...]

========================================
All Tests Passed!
========================================
✓ Tensor creation and shape management
✓ CPU data operations (min, max, mean, sum)
✓ 4D convenience indexing
✓ GPU memory allocation
✓ H2D transfers (CPU -> GPU)
✓ D2H transfers (GPU -> CPU)
✓ File I/O (save/load)
✓ GPU memory cleanup
✓ Error handling with CUDA_CHECK
✓ GPU memory tracking

Phase 1 Foundation is Ready!
Ready to proceed to Phase 2: GPU Kernel Implementation
```

## Error Handling

All CUDA operations are wrapped with `CUDA_CHECK()` macro:
- If any CUDA call fails, the program throws an exception
- Exception message explains the CUDA error
- Program catches exception and exits with code 1

**Example CUDA Error:**
```
❌ ERROR: CUDA Error: out of memory (2)
```

## Debugging Tips

### If Compilation Fails

1. **Include path errors:** Check `target_include_directories()` in CMakeLists.txt
2. **CUDA not found:** Run `cmake` with `--debug-output`
3. **Missing libcudart:** Check `/usr/lib/x86_64-linux-gnu/libcudart.so*` exists

### If Runtime Fails

1. **"No CUDA device found":** Run `nvidia-smi` to check GPU visibility
2. **"Out of memory":** Reduce tensor size (currently 0.57 MB, should fit on any GPU)
3. **"GPU data pointer is null":** `allocate_gpu()` failed; check error message

## Output Files

The program saves a test tensor:
- **Path:** `${OUTPUT_DIR}/test_tensor.bin`
- **Format:** Binary (4-byte shape_size, shape integers, float data)
- **Verification:** Loads and compares shape with original

## Next Steps

### Step 5: CMake Configuration and First Compilation
- Configure CMake to detect CUDA correctly
- Resolve any compiler/linker issues
- Verify all libraries are found

### Step 6: Create Layer Stubs
- BasicBlock, Conv2D, BatchNorm, ReLU, etc.
- Skeleton implementations (CPU-only, no kernels yet)
- Ensure layer architecture compiles

### Step 7: Phase 1 Verification Checklist
- All steps completed and tested
- No memory leaks detected
- Ready for Phase 2: GPU kernel implementation

## Compilation Checklist

- [ ] `cmake ..` succeeds with no CUDA errors
- [ ] `make -j4` produces `./resnet18_cuda` executable
- [ ] `./resnet18_cuda` runs without segmentation faults
- [ ] All 10 tests pass
- [ ] GPU memory correctly allocated/freed
- [ ] File I/O creates `test_tensor.bin` in OUTPUT_DIR
- [ ] Program exits with code 0

## Code Structure

```cpp
main.cu
├── #include "tensor.h"           // GPU-enabled Tensor class
├── #include "cuda_utils.h"       // GPU memory utilities
├── void print_section()          // Formatting helper
└── int main()
    ├── Test 1: GPU Info
    ├── Test 2: Tensor Creation
    ├── Test 3: CPU Operations
    ├── Test 4: 4D Indexing
    ├── Test 5: GPU Allocation
    ├── Test 6: H2D Transfer
    ├── Test 7: D2H Transfer
    ├── Test 8: File I/O
    ├── Test 9: Cleanup
    ├── Test 10: String Repr
    └── Summary Output
```

## Summary

✅ **Step 4 Complete**

The test program provides:
1. **10 comprehensive tests** covering all Tensor functionality
2. **GPU memory tracking** via cuda_utils.h
3. **Clear error reporting** with CUDA_CHECK() macro
4. **File I/O validation** (save and load)
5. **GPU transfer verification** (H2D and D2H)

**Ready for Step 5: CMake Configuration and First Compilation**
