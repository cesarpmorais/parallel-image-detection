#ifndef BATCHNORM_GPU_H
#define BATCHNORM_GPU_H

#include "../tensor.h"

// Host wrapper: runs batch normalization on GPU. Assumes input and params are on GPU.
// output will be allocated on GPU (caller should ensure GPU allocation or call will allocate).
void batchnorm_gpu_forward(const Tensor &input,
                           const Tensor &gamma,
                           const Tensor &beta,
                           const Tensor &running_mean,
                           const Tensor &running_var,
                           Tensor &output,
                           float eps);

#endif // BATCHNORM_GPU_H
