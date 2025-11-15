#ifndef ADAPTIVEAVGPOOL_H
#define ADAPTIVEAVGPOOL_H

#include "tensor.h"

/**
 * AdaptiveAvgPool2D
 * Pooling adaptativo que reduz dimensões espaciais para tamanho fixo (1x1)
 * Calcula média de todas as dimensões espaciais
 */
class AdaptiveAvgPool2D {
public:
    AdaptiveAvgPool2D(int output_size = 1);
    
    // Forward pass
    Tensor forward(const Tensor& input) const;
    
private:
    int output_size_;
};

#endif // ADAPTIVEAVGPOOL_H

