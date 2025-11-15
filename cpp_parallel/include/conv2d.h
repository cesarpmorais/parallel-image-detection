#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"
#include <string>

/**
 * Camada Convolucional 2D
 * Implementa convolução com padding e stride
 */
class Conv2D {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0);
    
    // Carregar pesos de arquivo binário
    bool load_weights(const std::string& weight_file);
    
    // Forward pass
    Tensor forward(const Tensor& input) const;
    
    // Getters
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int kernel_size() const { return kernel_size_; }
    int stride() const { return stride_; }
    int padding() const { return padding_; }
    
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    
    // Pesos: [out_channels, in_channels, kernel_h, kernel_w]
    Tensor weights_;
    
    // Helper para calcular dimensões de saída
    int output_size(int input_size) const {
        return (input_size + 2 * padding_ - kernel_size_) / stride_ + 1;
    }
};

#endif // CONV2D_H

