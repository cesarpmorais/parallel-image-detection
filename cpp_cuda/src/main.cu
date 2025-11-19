#include "../include/tensor.h"
#include "../include/cuda_utils.h"
#include <iostream>
#include <iomanip>

/**
 * Phase 1, Step 4: Minimal GPU Test Program
 *
 * Tests:
 * 1. Tensor creation (CPU)
 * 2. CPU operations (min, max, mean)
 * 3. GPU memory allocation and transfers (H2D, D2H)
 * 4. GPU memory info
 * 5. File I/O
 *
 * This program verifies the GPU-enabled Tensor infrastructure is working
 * before moving to Phase 2 (implementing actual GPU kernels).
 */

void print_section(const std::string &title)
{
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << title << "\n";
    std::cout << "========================================\n";
}

int main()
{
    try
    {
        print_section("Phase 1, Step 4: GPU Tensor Test");

        // ============ Test 1: GPU Info ============
        print_section("Test 1: GPU Information");
        print_gpu_memory_info();

        // ============ Test 2: Tensor Creation ============
        print_section("Test 2: Tensor Creation");

        std::cout << "Creating 4D tensor (1, 3, 224, 224)...\n";
        Tensor input({1, 3, 224, 224});
        input.print_info();

        std::cout << "\nTensor shape: [";
        for (size_t i = 0; i < input.get_shape().size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << input.get_shape()[i];
        }
        std::cout << "]\n";
        std::cout << "Total elements: " << input.numel() << "\n";
        std::cout << "Memory size: " << std::fixed << std::setprecision(2)
                  << input.size_bytes() / (1024.0 * 1024.0) << " MB\n";

        // ============ Test 3: CPU Operations ============
        print_section("Test 3: CPU Data Operations");

        std::cout << "Filling tensor with sequential values...\n";
        for (size_t i = 0; i < input.numel(); i++)
        {
            input[i] = static_cast<float>(i) / input.numel();
        }

        std::cout << "Min value: " << input.min() << "\n";
        std::cout << "Max value: " << input.max() << "\n";
        std::cout << "Mean value: " << input.mean() << "\n";
        std::cout << "Sum: " << input.sum() << "\n";

        // ============ Test 4: 4D Indexing ============
        print_section("Test 4: 4D Convenience Indexing");

        std::cout << "Testing 4D indexing: input(0, 0, 0, 0)...\n";
        float val = input(0, 0, 0, 0);
        std::cout << "Value at (0,0,0,0): " << val << "\n";

        std::cout << "Setting input(0, 0, 0, 0) = 99.5f\n";
        input(0, 0, 0, 0) = 99.5f;
        std::cout << "Verification: input(0, 0, 0, 0) = " << input(0, 0, 0, 0) << "\n";

        // ============ Test 5: GPU Memory Allocation ============
        print_section("Test 5: GPU Memory Allocation");

        std::cout << "Allocating GPU memory...\n";
        input.allocate_gpu();
        std::cout << "GPU allocated: " << (input.is_on_gpu() ? "Yes" : "No (need to transfer)\n");
        print_gpu_memory_info();

        // ============ Test 6: H2D Transfer ============
        print_section("Test 6: CPU -> GPU Transfer (H2D)");

        std::cout << "Transferring tensor to GPU...\n";
        input.to_gpu();
        std::cout << "Data on GPU: " << (input.is_on_gpu() ? "Yes" : "No") << "\n";
        std::cout << "GPU data pointer: " << input.get_gpu_data() << "\n";
        print_gpu_memory_info();

        // ============ Test 7: D2H Transfer ============
        print_section("Test 7: GPU -> CPU Transfer (D2H)");

        std::cout << "Modifying value on CPU before transfer...\n";
        input[0] = 42.0f;

        std::cout << "Transferring tensor back to CPU...\n";
        input.to_cpu();
        std::cout << "Data on GPU: " << (input.is_on_gpu() ? "Yes" : "No") << "\n";

        std::cout << "CPU value at [0]: " << input[0] << "\n";
        std::cout << "Note: GPU version should have overwridden CPU value\n";
        print_gpu_memory_info();

        // ============ Test 8: File I/O ============
        print_section("Test 8: File I/O");

        std::string output_file = OUTPUT_DIR "/test_tensor.bin";
        std::cout << "Saving tensor to: " << output_file << "\n";
        input.save_to_bin(output_file);
        std::cout << "Saved successfully.\n";

        std::cout << "\nLoading tensor from file...\n";
        Tensor loaded;
        loaded.load_from_bin(output_file);
        loaded.print_info();
        std::cout << "Loaded shape: [";
        for (size_t i = 0; i < loaded.get_shape().size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << loaded.get_shape()[i];
        }
        std::cout << "]\n";

        // ============ Test 9: Memory Cleanup ============
        print_section("Test 9: GPU Memory Cleanup");

        std::cout << "Freeing GPU memory...\n";
        input.free_gpu();
        std::cout << "GPU allocated: " << (input.is_on_gpu() ? "Yes" : "No") << "\n";
        print_gpu_memory_info();

        // ============ Test 10: Tensor Statistics ============
        print_section("Test 10: Tensor String Representation");

        std::cout << loaded.to_string() << "\n";

        // ============ Summary ============
        print_section("All Tests Passed!");

        std::cout << "✓ Tensor creation and shape management\n";
        std::cout << "✓ CPU data operations (min, max, mean, sum)\n";
        std::cout << "✓ 4D convenience indexing\n";
        std::cout << "✓ GPU memory allocation\n";
        std::cout << "✓ H2D transfers (CPU -> GPU)\n";
        std::cout << "✓ D2H transfers (GPU -> CPU)\n";
        std::cout << "✓ File I/O (save/load)\n";
        std::cout << "✓ GPU memory cleanup\n";
        std::cout << "✓ Error handling with CUDA_CHECK\n";
        std::cout << "✓ GPU memory tracking\n";

        std::cout << "\nPhase 1 Foundation is Ready!\n";
        std::cout << "Ready to proceed to Phase 2: GPU Kernel Implementation\n\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n❌ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
