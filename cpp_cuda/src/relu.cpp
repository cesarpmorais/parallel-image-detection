#include "../include/relu.h"
#include <algorithm>

void ReLU::forward(Tensor &input) const
{
    // Aplicar ReLU in-place: max(0, x)
    float *data = input.data_ptr();
    int size = input.size();

    for (int i = 0; i < size; ++i)
    {
        data[i] = std::max(0.0f, data[i]);
    }
}

Tensor ReLU::forward_copy(const Tensor &input) const
{
    // Versão que cria novo tensor (para casos onde não queremos modificar o original)
    Tensor output = input; // Copia
    forward(output);       // Aplica ReLU in-place na cópia
    return output;
}
