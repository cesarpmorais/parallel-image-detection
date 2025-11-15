#include "../include/adaptiveavgpool.h"
#include <cmath>

AdaptiveAvgPool2D::AdaptiveAvgPool2D(int output_size)
    : output_size_(output_size) {
}

Tensor AdaptiveAvgPool2D::forward(const Tensor& input) const {
    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();
    
    // Output sempre será (batch, channels, 1, 1) para AdaptiveAvgPool
    Tensor output(batch, channels, output_size_, output_size_);
    
    // Para cada item do batch e canal
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            int count = in_h * in_w;
            
            // Calcular média de todas as posições espaciais
            for (int h = 0; h < in_h; ++h) {
                for (int w = 0; w < in_w; ++w) {
                    sum += input(b, c, h, w);
                }
            }
            
            float avg = sum / count;
            
            // Preencher saída (1x1)
            output(b, c, 0, 0) = avg;
        }
    }
    
    return output;
}

