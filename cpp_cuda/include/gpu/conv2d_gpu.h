#ifndef CONV2D_GPU_H
#define CONV2D_GPU_H

#include "../tensor.h"

// Host wrapper: runs convolution on GPU. Assumes input/weights are on GPU.
// output will be allocated on GPU (output.allocate_gpu() must be called by caller)
void conv2d_gpu_forward(const Tensor &input, const Tensor &weights, Tensor &output,
                        int stride, int padding);

#endif // CONV2D_GPU_H
