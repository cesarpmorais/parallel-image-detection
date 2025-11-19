#include "../include/linear.h"
#include <iostream>
#include <fstream>

// Helper function to load raw float data
static bool load_tensor_raw(Tensor &tensor, const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expected_size = tensor.numel() * sizeof(float);
    if (file_size != expected_size)
    {
        file.close();
        return false;
    }

    // Read raw data
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), file_size);
    file.close();

    return true;
}

Linear::Linear(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features),
      weight_(out_features, in_features, 1, 1), // [out_features, in_features]
      bias_(1, out_features, 1, 1)
{ // [out_features]
}

bool Linear::load_weights(const std::string &weight_file)
{
    return load_tensor_raw(weight_, weight_file);
}

bool Linear::load_bias(const std::string &bias_file)
{
    return load_tensor_raw(bias_, bias_file);
}

Tensor Linear::forward(const Tensor &input) const
{
    // Input deve ser [batch, in_features] ou [batch, in_features, 1, 1]
    // Vamos assumir que já foi flattenado: [batch, in_features]

    int batch = input.batch();
    int in_size = input.channels() * input.height() * input.width();

    if (in_size != in_features_)
    {
        std::cerr << "Error: Input size mismatch. Expected " << in_features_
                  << ", got " << in_size << std::endl;
        return Tensor(1, 1, 1, 1); // Retornar tensor vazio em caso de erro
    }

    // Output: [batch, out_features]
    Tensor output(batch, out_features_, 1, 1);
    output.zeros();

    // Para cada item do batch
    for (int b = 0; b < batch; ++b)
    {
        // Para cada feature de saída
        for (int of = 0; of < out_features_; ++of)
        {
            float sum = 0.0f;

            // Multiplicação de matrizes: output[b, of] = sum(input[b, if] * weight[of, if])
            for (int if_idx = 0; if_idx < in_features_; ++if_idx)
            {
                // Acessar input (pode estar em formato [batch, channels, h, w] ou já flattenado)
                float input_val;
                if (input.height() == 1 && input.width() == 1)
                {
                    // Já está no formato [batch, features, 1, 1]
                    input_val = input(b, if_idx, 0, 0);
                }
                else
                {
                    // Calcular índice linear
                    int c = if_idx / (input.height() * input.width());
                    int hw = if_idx % (input.height() * input.width());
                    int h = hw / input.width();
                    int w = hw % input.width();
                    input_val = input(b, c, h, w);
                }

                float weight_val = weight_(of, if_idx, 0, 0);
                sum += input_val * weight_val;
            }

            // Adicionar bias
            float bias_val = bias_(0, of, 0, 0);
            output(b, of, 0, 0) = sum + bias_val;
        }
    }

    return output;
}
