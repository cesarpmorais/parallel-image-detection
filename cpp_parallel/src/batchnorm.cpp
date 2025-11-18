#include "../include/batchnorm.h"
#include <iostream>
#include <cmath>

BatchNorm2D::BatchNorm2D(int num_channels, float eps)
    : num_channels_(num_channels), eps_(eps),
      weight_(1, num_channels, 1, 1),      // gamma
      bias_(1, num_channels, 1, 1),       // beta
      running_mean_(1, num_channels, 1, 1),
      running_var_(1, num_channels, 1, 1) {
}

bool BatchNorm2D::load_weights(const std::string& weight_file) {
    // weight_file contém gamma (scale)
    return weight_.load_from_bin(weight_file);
}

bool BatchNorm2D::load_bias(const std::string& bias_file) {
    // bias_file contém beta (shift)
    return bias_.load_from_bin(bias_file);
}

bool BatchNorm2D::load_running_mean(const std::string& mean_file) {
    return running_mean_.load_from_bin(mean_file);
}

bool BatchNorm2D::load_running_var(const std::string& var_file) {
    return running_var_.load_from_bin(var_file);
}

Tensor BatchNorm2D::forward(const Tensor& input) const {
    int batch = input.batch();
    int channels = input.channels();
    int height = input.height();
    int width = input.width();
    
    // Output tem mesmo shape que input
    Tensor output(batch, channels, height, width);
    
    // Para cada canal
    for (int c = 0; c < channels; ++c) {
        // Obter parâmetros do canal c
        float gamma = weight_(0, c, 0, 0);
        float beta = bias_(0, c, 0, 0);
        float mean = running_mean_(0, c, 0, 0);
        float var = running_var_(0, c, 0, 0);
        
        // Calcular std = sqrt(var + eps)
        float std = std::sqrt(var + eps_);
        
        // Normalizar: (x - mean) / std
        float inv_std = 1.0f / std;
        
        // Aplicar para todos os elementos do canal
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float x = input(b, c, h, w);
                    // output = gamma * (x - mean) / std + beta
                    output(b, c, h, w) = gamma * (x - mean) * inv_std + beta;
                }
            }
        }
    }
    
    return output;
}

