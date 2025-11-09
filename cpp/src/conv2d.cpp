#include "../include/conv2d.h"
#include <iostream>
#include <cstring>

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, 
               int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      weights_(out_channels, in_channels, kernel_size, kernel_size) {
}

bool Conv2D::load_weights(const std::string& weight_file) {
    return weights_.load_from_bin(weight_file);
}

Tensor Conv2D::forward(const Tensor& input) const {
    int batch = input.batch();
    int in_h = input.height();
    int in_w = input.width();
    
    int out_h = output_size(in_h);
    int out_w = output_size(in_w);
    
    Tensor output(batch, out_channels_, out_h, out_w);
    output.zeros();
    
    // Para cada item do batch
    for (int b = 0; b < batch; ++b) {
        // Para cada canal de saída
        for (int oc = 0; oc < out_channels_; ++oc) {
            // Para cada posição de saída
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    
                    // Para cada canal de entrada
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        // Para cada posição do kernel
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                // Calcular posição na entrada (com padding)
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                
                                // Aplicar padding (zero-padding)
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    float input_val = input(b, ic, ih, iw);
                                    float weight_val = weights_(oc, ic, kh, kw);
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                    
                    output(b, oc, oh, ow) = sum;
                }
            }
        }
    }
    
    return output;
}

