#include "../include/conv2d.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include "../include/gpu/conv2d_gpu.h"

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size,
               int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      weights_(out_channels, in_channels, kernel_size, kernel_size)
{
}

bool Conv2D::load_weights(const std::string &weight_file)
{
    // Load as raw float data (like CPU version)
    std::ifstream file(weight_file, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expected_size = weights_.numel() * sizeof(float);
    if (file_size != expected_size)
    {
        file.close();
        return false;
    }

    // Read raw data directly into tensor
    file.read(reinterpret_cast<char *>(weights_.data_ptr()), file_size);
    file.close();

    return true;
}

Tensor Conv2D::forward(const Tensor &input) const
{
    // Compute output shape
    int batch = input.batch();
    int in_h = input.height();
    int in_w = input.width();

    int out_h = output_size(in_h);
    int out_w = output_size(in_w);

    Tensor output(batch, out_channels_, out_h, out_w);

    // GPU path: copy inputs/weights to GPU, run kernel, copy result back
    Tensor input_gpu = input;      // shallow copy of metadata + CPU data
    Tensor weights_gpu = weights_; // copy weights

    // Ensure GPU copies
    input_gpu.to_gpu();
    weights_gpu.to_gpu();

    // Allocate output on GPU
    output.allocate_gpu();
    output.to_gpu();

    // Execute GPU convolution
    conv2d_gpu_forward(input_gpu, weights_gpu, output, stride_, padding_);

    // Move result back to CPU for compatibility with existing codepaths
    output.to_cpu();

    // Free temporary GPU memory
    input_gpu.free_gpu();
    weights_gpu.free_gpu();

    return output;
}
