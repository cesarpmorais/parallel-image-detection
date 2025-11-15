#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "tensor.h"
#include <string>

/**
 * Camada BatchNorm2D
 * Normalização em lote para inferência
 * output = gamma * (input - mean) / sqrt(variance + eps) + beta
 */
class BatchNorm2D {
public:
    BatchNorm2D(int num_channels, float eps = 1e-5f);
    
    // Carregar parâmetros de arquivos binários
    bool load_weights(const std::string& weight_file);
    bool load_bias(const std::string& bias_file);
    bool load_running_mean(const std::string& mean_file);
    bool load_running_var(const std::string& var_file);
    
    // Forward pass (modo inferência - usa running statistics)
    Tensor forward(const Tensor& input) const;
    
    // Getters
    int num_channels() const { return num_channels_; }
    float eps() const { return eps_; }
    
private:
    int num_channels_;
    float eps_;
    
    // Parâmetros treináveis: [channels]
    Tensor weight_;   // gamma (scale)
    Tensor bias_;     // beta (shift)
    
    // Running statistics (usadas em inferência)
    Tensor running_mean_;
    Tensor running_var_;
};

#endif // BATCHNORM_H

