#include "../include/maxpool.h"
#include <algorithm>
#include <limits>

MaxPool2D::MaxPool2D(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
}

Tensor MaxPool2D::forward(const Tensor& input) const {
    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();
    
    int out_h = output_size(in_h);
    int out_w = output_size(in_w);
    
    Tensor output(batch, channels, out_h, out_w);
    
    // Para cada item do batch
    for (int b = 0; b < batch; ++b) {
        // Para cada canal
        for (int c = 0; c < channels; ++c) {
            // Para cada posição de saída
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = std::numeric_limits<float>::lowest();
                    
                    // Para cada posição do kernel
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            // Calcular posição na entrada (com padding)
                            int ih = oh * stride_ + kh - padding_;
                            int iw = ow * stride_ + kw - padding_;
                            
                            // Aplicar padding (zero-padding)
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                float val = input(b, c, ih, iw);
                                max_val = std::max(max_val, val);
                            }
                        }
                    }
                    
                    output(b, c, oh, ow) = max_val;
                }
            }
        }
    }
    
    return output;
}

