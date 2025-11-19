#include "../include/gpu/conv2d_gpu.h"
#include "../include/cuda_utils.h"
#include <cuda_runtime.h>
#include <cmath>

// Naive per-output-element convolution kernel.
__global__ void conv2d_kernel(const float *input, const float *weights, float *output,
                              int B, int C_in, int H_in, int W_in,
                              int K, int stride, int pad,
                              int C_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * H_out * W_out;
    if (idx >= total)
        return;

    int w = idx % W_out;
    int tmp = idx / W_out;
    int h = tmp % H_out;
    tmp = tmp / H_out;
    int oc = tmp % C_out;
    int b = tmp / C_out;

    float acc = 0.0f;

    for (int ic = 0; ic < C_in; ++ic)
    {
        for (int kh = 0; kh < K; ++kh)
        {
            for (int kw = 0; kw < K; ++kw)
            {
                int in_h = h * stride - pad + kh;
                int in_w = w * stride - pad + kw;
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in)
                    continue;

                // input index: ((b * C_in + ic) * H_in + in_h) * W_in + in_w
                size_t in_idx = ((size_t)b * C_in + ic) * H_in * W_in + (size_t)in_h * W_in + in_w;
                // weight index: ((oc * C_in + ic) * K + kh) * K + kw
                size_t w_idx = ((size_t)oc * C_in + ic) * K * K + kh * K + kw;

                acc += input[in_idx] * weights[w_idx];
            }
        }
    }

    // output index: ((b * C_out + oc) * H_out + h) * W_out + w
    size_t out_idx = ((size_t)b * C_out + oc) * H_out * W_out + (size_t)h * W_out + w;
    output[out_idx] = acc;
}

void conv2d_gpu_forward(const Tensor &input, const Tensor &weights, Tensor &output,
                        int stride, int padding)
{
    // Expect shapes:
    // input: {B, C_in, H_in, W_in}
    // weights: {C_out, C_in, K, K}
    // output: {B, C_out, H_out, W_out} (gpu memory allocated)

    const std::vector<int> in_shape = input.get_shape();
    const std::vector<int> w_shape = weights.get_shape();
    const std::vector<int> out_shape = output.get_shape();

    int B = in_shape[0];
    int C_in = in_shape[1];
    int H_in = in_shape[2];
    int W_in = in_shape[3];

    int C_out = w_shape[0];
    int K = w_shape[2];

    int H_out = out_shape[2];
    int W_out = out_shape[3];

    // Ensure GPU memory is allocated
    if (!input.is_on_gpu() || !weights.is_on_gpu())
    {
        throw std::runtime_error("input and weights must be on GPU before calling conv2d_gpu_forward");
    }
    if (!output.is_on_gpu())
    {
        output.allocate_gpu();
    }

    const float *d_input = input.get_gpu_data();
    const float *d_weights = weights.get_gpu_data();
    float *d_output = output.get_gpu_data();

    int total = B * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(d_input, d_weights, d_output,
                                       B, C_in, H_in, W_in,
                                       K, stride, padding,
                                       C_out, H_out, W_out);
    CUDA_CHECK_KERNEL();
}
