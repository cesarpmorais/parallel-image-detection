#ifndef RELU_H
#define RELU_H

#include "tensor.h"

/**
 * Função de ativação ReLU
 * ReLU(x) = max(0, x)
 * Aplicada in-place para economizar memória
 */
class ReLU {
public:
    // Forward pass (modifica tensor in-place)
    void forward(Tensor& input) const;
    
    // Versão que retorna novo tensor (se necessário)
    Tensor forward_copy(const Tensor& input) const;
};

#endif // RELU_H

