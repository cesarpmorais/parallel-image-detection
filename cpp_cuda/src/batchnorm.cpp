#include "../include/batchnorm.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include "../include/gpu/batchnorm_gpu.h"

// Helper function to load raw float data
static bool load_tensor_raw(Tensor &tensor, const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expected_size = tensor.numel() * sizeof(float);
    if (file_size != expected_size)
    {
        file.close();
        return false;
    }

    // Read raw data
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), file_size);
    file.close();

    return true;
}

BatchNorm2D::BatchNorm2D(int num_channels, float eps)
    : num_channels_(num_channels), eps_(eps),
      weight_(1, num_channels, 1, 1), // gamma
      bias_(1, num_channels, 1, 1),   // beta
      running_mean_(1, num_channels, 1, 1),
      running_var_(1, num_channels, 1, 1)
{
}

bool BatchNorm2D::load_weights(const std::string &weight_file)
{
    // weight_file contém gamma (scale)
    return load_tensor_raw(weight_, weight_file);
}

bool BatchNorm2D::load_bias(const std::string &bias_file)
{
    // bias_file contém beta (shift)
    return load_tensor_raw(bias_, bias_file);
}

bool BatchNorm2D::load_running_mean(const std::string &mean_file)
{
    return load_tensor_raw(running_mean_, mean_file);
}

bool BatchNorm2D::load_running_var(const std::string &var_file)
{
    return load_tensor_raw(running_var_, var_file);
}

Tensor BatchNorm2D::forward(const Tensor &input) const
{
    int batch = input.batch();
    int channels = input.channels();
    int height = input.height();
    int width = input.width();

    // Output tem mesmo shape que input
    Tensor output(batch, channels, height, width);

    // GPU path: copy input and per-channel params to GPU, run kernel, copy back
    Tensor input_gpu = input;
    Tensor gamma_gpu = weight_;
    Tensor beta_gpu = bias_;
    Tensor mean_gpu = running_mean_;
    Tensor var_gpu = running_var_;

    // Ensure params and input are on GPU
    input_gpu.to_gpu();
    gamma_gpu.to_gpu();
    beta_gpu.to_gpu();
    mean_gpu.to_gpu();
    var_gpu.to_gpu();

    // Allocate output on GPU
    output.allocate_gpu();
    output.to_gpu();

    // Execute GPU batchnorm
    batchnorm_gpu_forward(input_gpu, gamma_gpu, beta_gpu, mean_gpu, var_gpu, output, eps_);

    // Move result back to CPU
    output.to_cpu();

    // Free temp GPU buffers
    input_gpu.free_gpu();
    gamma_gpu.free_gpu();
    beta_gpu.free_gpu();
    mean_gpu.free_gpu();
    var_gpu.free_gpu();

    return output;
}
