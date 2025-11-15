#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include <string>

/**
 * Camada Linear (Fully Connected)
 * Multiplicação de matrizes: output = input @ weight^T + bias
 */
class Linear {
public:
    Linear(int in_features, int out_features);
    
    // Carregar pesos
    bool load_weights(const std::string& weight_file);
    bool load_bias(const std::string& bias_file);
    
    // Forward pass
    Tensor forward(const Tensor& input) const;
    
    // Getters
    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }
    
private:
    int in_features_;
    int out_features_;
    
    // Pesos: [out_features, in_features]
    Tensor weight_;
    // Bias: [out_features]
    Tensor bias_;
};

#endif // LINEAR_H

