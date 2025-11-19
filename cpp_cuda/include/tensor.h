#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <iostream>

/**
 * Generic GPU-enabled Tensor class supporting any number of dimensions.
 *
 * Supports both CPU and GPU memory, with explicit memory location tracking
 * and transfer operations (H2D, D2H).
 *
 * Usage:
 *   // Create a 4D tensor (batch, channels, height, width)
 *   Tensor t({1, 3, 224, 224});
 *
 *   // Fill with data
 *   for (size_t i = 0; i < t.numel(); i++) {
 *       t.data[i] = some_value;
 *   }
 *
 *   // Transfer to GPU
 *   t.to_gpu();
 *
 *   // Run GPU kernel (handles gpu_data pointer)
 *   gpu_kernel(t);
 *
 *   // Transfer back to CPU
 *   t.to_cpu();
 */
class Tensor
{
public:
    // ============ Constructors & Destructors ============

    /**
     * Create a tensor with given shape.
     * @param shape Vector of dimensions. E.g., {1, 3, 224, 224} for 4D.
     */
    explicit Tensor(const std::vector<int> &shape);

    /**
     * Create a tensor with a single element.
     */
    Tensor();

    /**
     * Create a tensor with given batch, channels, height, width (4D convenience).
     * Legacy API for compatibility.
     */
    Tensor(int batch, int channels, int height, int width);

    /**
     * Copy constructor: deep copy of data and GPU state.
     */
    Tensor(const Tensor &other);

    /**
     * Move constructor: steal data and GPU pointers.
     */
    Tensor(Tensor &&other) noexcept;

    /**
     * Destructor: free GPU memory if allocated.
     */
    ~Tensor();

    /**
     * Copy assignment operator.
     */
    Tensor &operator=(const Tensor &other);

    /**
     * Move assignment operator.
     */
    Tensor &operator=(Tensor &&other) noexcept;

    // ============ Memory Management ============

    /**
     * Allocate GPU memory without copying data.
     * Safe to call multiple times; only allocates if not already allocated.
     */
    void allocate_gpu();

    /**
     * Transfer data from CPU to GPU (H2D).
     * Allocates GPU memory if needed.
     */
    void to_gpu();

    /**
     * Transfer data from GPU to CPU (D2H).
     * Does nothing if not on GPU.
     */
    void to_cpu();

    /**
     * Free GPU memory.
     * Safe to call multiple times; does nothing if not allocated.
     */
    void free_gpu();

    /**
     * Check if tensor data is currently on GPU.
     */
    bool is_on_gpu() const { return on_gpu; }

    /**
     * Get GPU data pointer. Returns nullptr if not on GPU.
     */
    float *get_gpu_data() { return gpu_data; }
    const float *get_gpu_data() const { return gpu_data; }

    // ============ Indexing & Access ============

    /**
     * Linear index access to CPU data.
     * Elements are stored in row-major (C) order.
     *
     * For 4D tensor of shape {B,C,H,W}, linear index is:
     *   idx = b*C*H*W + c*H*W + h*W + w
     */
    float &operator[](size_t index);
    const float &operator[](size_t index) const;

    /**
     * 4D convenience indexing: tensor(b, c, h, w).
     * Assumes shape is {B, C, H, W}. Throws if shape != 4D.
     */
    float &operator()(int b, int c, int h, int w);
    const float &operator()(int b, int c, int h, int w) const;

    /**
     * Get linear index for given multi-dimensional indices.
     * E.g., for shape {B,C,H,W}: compute_index({b,c,h,w})
     */
    size_t compute_index(const std::vector<int> &indices) const;

    /**
     * Direct pointer to CPU data.
     */
    float *data_ptr() { return data.data(); }
    const float *data_ptr() const { return data.data(); }

    // ============ Shape & Size ============

    /**
     * Get tensor shape (vector of dimensions).
     */
    const std::vector<int> &get_shape() const { return shape; }

    /**
     * Legacy 4D API convenience methods.
     * Only valid for 4D tensors; throws if tensor is not 4D.
     */
    int batch() const
    {
        if (ndim() != 4)
            throw std::runtime_error("batch() requires 4D tensor");
        return shape[0];
    }
    int channels() const
    {
        if (ndim() != 4)
            throw std::runtime_error("channels() requires 4D tensor");
        return shape[1];
    }
    int height() const
    {
        if (ndim() != 4)
            throw std::runtime_error("height() requires 4D tensor");
        return shape[2];
    }
    int width() const
    {
        if (ndim() != 4)
            throw std::runtime_error("width() requires 4D tensor");
        return shape[3];
    }

    /**
     * Get size along a specific dimension.
     * @param dim Dimension (0-indexed). Negative indices count from end.
     */
    int size(int dim) const;

    /**
     * Legacy API: get total element count (same as numel()).
     */
    int size() const { return numel(); }

    /**
     * Get total number of elements.
     */
    size_t numel() const { return num_elements; }

    /**
     * Get number of dimensions.
     */
    int ndim() const { return shape.size(); }

    /**
     * Reshape tensor (must have same number of elements).
     */
    void reshape(const std::vector<int> &new_shape);

    /**
     * Get size in bytes of CPU data.
     */
    size_t size_bytes() const { return num_elements * sizeof(float); }

    /**
     * Fill tensor with zeros.
     */
    void zeros()
    {
        std::fill(data.begin(), data.end(), 0.0f);
        on_gpu = false;
        free_gpu();
    }

    /**
     * Fill tensor with ones.
     */
    void ones()
    {
        std::fill(data.begin(), data.end(), 1.0f);
        on_gpu = false;
        free_gpu();
    }

    /**
     * Fill tensor with a specific value.
     */
    void fill(float value)
    {
        std::fill(data.begin(), data.end(), value);
        on_gpu = false;
        free_gpu();
    }

    // ============ File I/O ============

    /**
     * Load tensor from binary file.
     * Expects file format: [shape_size:int32][shape...][data...].
     * Data is loaded to CPU.
     */
    void load_from_bin(const std::string &filename);

    /**
     * Load tensor from binary file (returns bool for legacy compatibility).
     */
    bool load_from_bin_compat(const std::string &filename)
    {
        try
        {
            load_from_bin(filename);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    /**
     * Save tensor to binary file.
     * Saves CPU data. If on GPU, automatically transfers to CPU first.
     * Format: [shape_size:int32][shape...][data...].
     */
    void save_to_bin(const std::string &filename) const;

    /**
     * Save tensor shape to a text file.
     * Format: one dimension per line.
     */
    bool save_shape_to_txt(const std::string &filepath) const;

    // ============ Statistics ============

    /**
     * Compute minimum value (CPU data).
     */
    float min() const;

    /**
     * Compute maximum value (CPU data).
     */
    float max() const;

    /**
     * Compute mean value (CPU data).
     */
    float mean() const;

    /**
     * Compute sum of all elements (CPU data).
     */
    float sum() const;

    // ============ Utilities ============

    /**
     * Print tensor shape and memory status.
     */
    void print_info() const;

    /**
     * Get detailed string representation.
     */
    std::string to_string() const;

private:
    std::vector<float> data; // CPU memory
    std::vector<int> shape;  // Tensor shape (e.g., {batch, channels, height, width})
    size_t num_elements;     // Total number of elements (product of shape)

    float *gpu_data;    // GPU memory pointer (nullptr if not allocated)
    bool on_gpu;        // True if data is currently on GPU
    bool gpu_allocated; // True if GPU memory was allocated

    /**
     * Recompute num_elements from shape.
     */
    void compute_num_elements();
};

#endif // TENSOR_H
