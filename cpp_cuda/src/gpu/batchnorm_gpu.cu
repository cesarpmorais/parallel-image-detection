#include "../include/gpu/batchnorm_gpu.h"
#include "../include/cuda_utils.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void batchnorm_kernel(const float *input, const float *gamma, const float *beta,
                                 const float *running_mean, const float *running_var,
                                 float *output, int B, int C, int H, int W, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total)
        return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp = tmp / H;
    int c = tmp % C;
    int b = tmp / C;

    // Per-channel params
    float g = gamma[c];
    float be = beta[c];
    float mean = running_mean[c];
    float var = running_var[c];

    float inv_std = 1.0f / sqrtf(var + eps);

    // Compute flat index corresponding to (b,c,h,w)
    size_t flat = ((size_t)b * C + c) * H * W + (size_t)h * W + w;
    float x = input[flat];
    float y = g * (x - mean) * inv_std + be;
    output[flat] = y;
}

void batchnorm_gpu_forward(const Tensor &input,
                           const Tensor &gamma,
                           const Tensor &beta,
                           const Tensor &running_mean,
                           const Tensor &running_var,
                           Tensor &output,
                           float eps)
{
    auto in_shape = input.get_shape();
    int B = in_shape[0];
    int C = in_shape[1];
    int H = in_shape[2];
    int W = in_shape[3];

    if (!input.is_on_gpu() || !gamma.is_on_gpu() || !beta.is_on_gpu() || !running_mean.is_on_gpu() || !running_var.is_on_gpu())
    {
        throw std::runtime_error("input and batchnorm params must be on GPU before calling batchnorm_gpu_forward");
    }

    if (!output.is_on_gpu())
    {
        output.allocate_gpu();
    }

    const float *d_input = input.get_gpu_data();
    const float *d_gamma = gamma.get_gpu_data();
    const float *d_beta = beta.get_gpu_data();
    const float *d_mean = running_mean.get_gpu_data();
    const float *d_var = running_var.get_gpu_data();
    float *d_output = output.get_gpu_data();

    int total = B * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    batchnorm_kernel<<<blocks, threads>>>(d_input, d_gamma, d_beta, d_mean, d_var, d_output, B, C, H, W, eps);
    CUDA_CHECK_KERNEL();
}
