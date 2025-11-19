#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "tensor.h"

/**
 * Camada MaxPool2D
 * Pooling máximo 2D que reduz dimensões espaciais
 * Toma o valor máximo em cada janela
 */
class MaxPool2D {
public:
    MaxPool2D(int kernel_size, int stride, int padding = 0);
    
    // Forward pass
    Tensor forward(const Tensor& input) const;
    
    // Getters
    int kernel_size() const { return kernel_size_; }
    int stride() const { return stride_; }
    int padding() const { return padding_; }
    
private:
    int kernel_size_;
    int stride_;
    int padding_;
    
    // Helper para calcular dimensões de saída
    int output_size(int input_size) const {
        return (input_size + 2 * padding_ - kernel_size_) / stride_ + 1;
    }
};

#endif // MAXPOOL_H

