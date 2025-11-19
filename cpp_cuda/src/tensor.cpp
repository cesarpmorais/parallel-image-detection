#include "../include/tensor.h"
#include "../include/cuda_utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

// ============ Constructors & Destructors ============

Tensor::Tensor()
    : shape({1}), num_elements(1), gpu_data(nullptr), on_gpu(false), gpu_allocated(false)
{
    data.resize(1, 0.0f);
}

Tensor::Tensor(const std::vector<int> &shape_)
    : shape(shape_), gpu_data(nullptr), on_gpu(false), gpu_allocated(false)
{
    compute_num_elements();
    data.resize(num_elements, 0.0f);
}

Tensor::Tensor(int batch, int channels, int height, int width)
    : shape({batch, channels, height, width}),
      gpu_data(nullptr), on_gpu(false), gpu_allocated(false)
{
    compute_num_elements();
    data.resize(num_elements, 0.0f);
}

Tensor::Tensor(const Tensor &other)
    : data(other.data), shape(other.shape), num_elements(other.num_elements),
      gpu_data(nullptr), on_gpu(false), gpu_allocated(false)
{
    // Don't copy GPU state; new tensor starts on CPU only
}

Tensor::Tensor(Tensor &&other) noexcept
    : data(std::move(other.data)), shape(std::move(other.shape)),
      num_elements(other.num_elements), gpu_data(other.gpu_data),
      on_gpu(other.on_gpu), gpu_allocated(other.gpu_allocated)
{
    // Steal GPU pointer; clear the source
    other.gpu_data = nullptr;
    other.on_gpu = false;
    other.gpu_allocated = false;
}

Tensor::~Tensor()
{
    free_gpu();
}

Tensor &Tensor::operator=(const Tensor &other)
{
    if (this == &other)
        return *this;

    // Free existing GPU memory
    free_gpu();

    // Copy CPU data
    data = other.data;
    shape = other.shape;
    num_elements = other.num_elements;

    // Don't copy GPU state
    gpu_data = nullptr;
    on_gpu = false;
    gpu_allocated = false;

    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this == &other)
        return *this;

    // Free existing GPU memory
    free_gpu();

    // Steal resources
    data = std::move(other.data);
    shape = std::move(other.shape);
    num_elements = other.num_elements;
    gpu_data = other.gpu_data;
    on_gpu = other.on_gpu;
    gpu_allocated = other.gpu_allocated;

    // Clear source
    other.gpu_data = nullptr;
    other.on_gpu = false;
    other.gpu_allocated = false;

    return *this;
}

// ============ Memory Management ============

void Tensor::compute_num_elements()
{
    num_elements = 1;
    for (int s : shape)
    {
        if (s <= 0)
            throw std::invalid_argument("Tensor shape dimensions must be positive");
        num_elements *= s;
    }
}

void Tensor::allocate_gpu()
{
    if (gpu_allocated)
        return; // Already allocated

    size_t bytes = num_elements * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_data, bytes));
    gpu_allocated = true;
    on_gpu = false; // Not yet populated with data

    // Debug output
    // std::cout << "Allocated " << bytes / (1024.0 * 1024.0) << " MB on GPU\n";
}

void Tensor::to_gpu()
{
    if (on_gpu)
        return; // Already on GPU

    allocate_gpu(); // Ensure GPU memory is allocated

    size_t bytes = num_elements * sizeof(float);
    CUDA_CHECK(cudaMemcpy(gpu_data, data.data(), bytes, cudaMemcpyHostToDevice));
    on_gpu = true;

    // Debug output
    // std::cout << "Copied " << bytes / (1024.0 * 1024.0) << " MB to GPU (H2D)\n";
}

void Tensor::to_cpu()
{
    if (!on_gpu)
        return; // Not on GPU, nothing to do

    size_t bytes = num_elements * sizeof(float);
    CUDA_CHECK(cudaMemcpy(data.data(), gpu_data, bytes, cudaMemcpyDeviceToHost));
    on_gpu = false;

    // Debug output
    // std::cout << "Copied " << bytes / (1024.0 * 1024.0) << " MB to CPU (D2H)\n";
}

void Tensor::free_gpu()
{
    if (!gpu_allocated)
        return;

    if (gpu_data != nullptr)
    {
        CUDA_CHECK(cudaFree(gpu_data));
        gpu_data = nullptr;
    }
    gpu_allocated = false;
    on_gpu = false;
}

// ============ Indexing & Access ============

float &Tensor::operator[](size_t index)
{
    if (index >= num_elements)
        throw std::out_of_range("Tensor index out of range");
    return data[index];
}

const float &Tensor::operator[](size_t index) const
{
    if (index >= num_elements)
        throw std::out_of_range("Tensor index out of range");
    return data[index];
}

float &Tensor::operator()(int b, int c, int h, int w)
{
    if (ndim() != 4)
        throw std::runtime_error("4D indexing requires 4D tensor");
    int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
    size_t idx = ((size_t)b * C + c) * H * W + (size_t)h * W + w;
    return data[idx];
}

const float &Tensor::operator()(int b, int c, int h, int w) const
{
    if (ndim() != 4)
        throw std::runtime_error("4D indexing requires 4D tensor");
    int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
    size_t idx = ((size_t)b * C + c) * H * W + (size_t)h * W + w;
    return data[idx];
}

size_t Tensor::compute_index(const std::vector<int> &indices) const
{
    if (indices.size() != shape.size())
    {
        throw std::invalid_argument("Index dimensions don't match tensor dimensions");
    }

    size_t idx = 0;
    size_t multiplier = 1;

    // Compute index in row-major (C) order
    for (int i = (int)shape.size() - 1; i >= 0; i--)
    {
        if (indices[i] < 0 || indices[i] >= shape[i])
        {
            throw std::out_of_range("Index out of bounds");
        }
        idx += indices[i] * multiplier;
        multiplier *= shape[i];
    }

    return idx;
}

// ============ Shape & Size ============

int Tensor::size(int dim) const
{
    int actual_dim = dim;
    if (dim < 0)
        actual_dim = shape.size() + dim; // Negative indexing
    if (actual_dim < 0 || actual_dim >= (int)shape.size())
    {
        throw std::out_of_range("Dimension out of range");
    }
    return shape[actual_dim];
}

void Tensor::reshape(const std::vector<int> &new_shape)
{
    size_t new_elements = 1;
    for (int s : new_shape)
    {
        if (s <= 0)
            throw std::invalid_argument("New shape dimensions must be positive");
        new_elements *= s;
    }

    if (new_elements != num_elements)
    {
        throw std::invalid_argument("Reshape dimensions must match total element count");
    }

    shape = new_shape;
    // num_elements stays the same
    on_gpu = false; // Invalidate GPU copy after reshape
    free_gpu();
}

// ============ File I/O ============

void Tensor::load_from_bin(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Read shape size
    int shape_size;
    file.read(reinterpret_cast<char *>(&shape_size), sizeof(int));
    if (!file)
        throw std::runtime_error("Failed to read shape size");

    // Read shape
    shape.resize(shape_size);
    file.read(reinterpret_cast<char *>(shape.data()), shape_size * sizeof(int));
    if (!file)
        throw std::runtime_error("Failed to read shape");

    compute_num_elements();
    data.resize(num_elements);

    // Read data
    file.read(reinterpret_cast<char *>(data.data()), num_elements * sizeof(float));
    if (!file)
        throw std::runtime_error("Failed to read tensor data");

    on_gpu = false;
    free_gpu();

    file.close();
}

void Tensor::save_to_bin(const std::string &filename) const
{
    // If on GPU, copy to CPU first
    if (on_gpu)
    {
        const_cast<Tensor *>(this)->to_cpu();
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Write shape size
    int shape_size = shape.size();
    file.write(reinterpret_cast<const char *>(&shape_size), sizeof(int));

    // Write shape
    file.write(reinterpret_cast<const char *>(shape.data()), shape_size * sizeof(int));

    // Write data
    file.write(reinterpret_cast<const char *>(data.data()), num_elements * sizeof(float));

    if (!file)
        throw std::runtime_error("Failed to write tensor to file");

    file.close();
}

bool Tensor::save_shape_to_txt(const std::string &filepath) const
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        return false;
    }

    for (const auto &dim : shape)
    {
        file << dim << "\n";
    }

    file.close();
    return true;
}

// ============ Statistics ============

float Tensor::min() const
{
    if (num_elements == 0)
        throw std::runtime_error("Empty tensor");
    return *std::min_element(data.begin(), data.end());
}

float Tensor::max() const
{
    if (num_elements == 0)
        throw std::runtime_error("Empty tensor");
    return *std::max_element(data.begin(), data.end());
}

float Tensor::mean() const
{
    if (num_elements == 0)
        throw std::runtime_error("Empty tensor");
    return sum() / num_elements;
}

float Tensor::sum() const
{
    if (num_elements == 0)
        return 0.0f;
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

// ============ Utilities ============

void Tensor::print_info() const
{
    std::cout << "Tensor Info:\n";
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]\n";
    std::cout << "  Elements: " << num_elements << "\n";
    std::cout << "  Size: " << num_elements * sizeof(float) / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Location: " << (on_gpu ? "GPU" : "CPU") << "\n";
    std::cout << "  GPU Allocated: " << (gpu_allocated ? "Yes" : "No") << "\n";
    if (num_elements > 0)
    {
        std::cout << "  Range: [" << min() << ", " << max() << "]\n";
        std::cout << "  Mean: " << mean() << "\n";
    }
}

std::string Tensor::to_string() const
{
    std::string result = "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++)
    {
        if (i > 0)
            result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "], on_gpu=" + std::string(on_gpu ? "true" : "false") + ")";
    return result;
}
