#include "../include/basicblock.h"
#include <iostream>
#include <sstream>

BasicBlock::BasicBlock(int in_channels, int out_channels, int stride, bool downsample)
    : in_channels_(in_channels), out_channels_(out_channels), 
      stride_(stride), downsample_(downsample),
      conv1_(in_channels, out_channels, 3, stride, 1),  // 3x3 conv, padding=1
      bn1_(out_channels),
      conv2_(out_channels, out_channels, 3, 1, 1),       // 3x3 conv, stride=1, padding=1
      bn2_(out_channels),
      downsample_conv_(in_channels, out_channels, 1, stride, 0),  // 1x1 conv para downsampling
      downsample_bn_(out_channels),
      has_downsample_(downsample || (in_channels != out_channels) || (stride != 1)) {
}

bool BasicBlock::load_weights(const std::string& weights_dir, const std::string& prefix) {
    // Carregar pesos do caminho principal
    std::string conv1_weight = weights_dir + "/" + prefix + "_conv1_weight.bin";
    std::string bn1_weight = weights_dir + "/" + prefix + "_bn1_weight.bin";
    std::string bn1_bias = weights_dir + "/" + prefix + "_bn1_bias.bin";
    std::string bn1_mean = weights_dir + "/" + prefix + "_bn1_running_mean.bin";
    std::string bn1_var = weights_dir + "/" + prefix + "_bn1_running_var.bin";
    
    std::string conv2_weight = weights_dir + "/" + prefix + "_conv2_weight.bin";
    std::string bn2_weight = weights_dir + "/" + prefix + "_bn2_weight.bin";
    std::string bn2_bias = weights_dir + "/" + prefix + "_bn2_bias.bin";
    std::string bn2_mean = weights_dir + "/" + prefix + "_bn2_running_mean.bin";
    std::string bn2_var = weights_dir + "/" + prefix + "_bn2_running_var.bin";
    
    // Carregar conv1
    if (!conv1_.load_weights(conv1_weight)) {
        std::cerr << "Error: Failed to load " << conv1_weight << std::endl;
        return false;
    }
    
    // Carregar bn1
    if (!bn1_.load_weights(bn1_weight) || !bn1_.load_bias(bn1_bias) ||
        !bn1_.load_running_mean(bn1_mean) || !bn1_.load_running_var(bn1_var)) {
        std::cerr << "Error: Failed to load bn1 for " << prefix << std::endl;
        return false;
    }
    
    // Carregar conv2
    if (!conv2_.load_weights(conv2_weight)) {
        std::cerr << "Error: Failed to load " << conv2_weight << std::endl;
        return false;
    }
    
    // Carregar bn2
    if (!bn2_.load_weights(bn2_weight) || !bn2_.load_bias(bn2_bias) ||
        !bn2_.load_running_mean(bn2_mean) || !bn2_.load_running_var(bn2_var)) {
        std::cerr << "Error: Failed to load bn2 for " << prefix << std::endl;
        return false;
    }
    
    // Carregar downsampling se necessário
    if (has_downsample_) {
        std::string ds_conv_weight = weights_dir + "/" + prefix + "_downsample_0_weight.bin";
        std::string ds_bn_weight = weights_dir + "/" + prefix + "_downsample_1_weight.bin";
        std::string ds_bn_bias = weights_dir + "/" + prefix + "_downsample_1_bias.bin";
        std::string ds_bn_mean = weights_dir + "/" + prefix + "_downsample_1_running_mean.bin";
        std::string ds_bn_var = weights_dir + "/" + prefix + "_downsample_1_running_var.bin";
        
        if (!downsample_conv_.load_weights(ds_conv_weight)) {
            std::cerr << "Error: Failed to load downsample conv for " << prefix << std::endl;
            return false;
        }
        
        if (!downsample_bn_.load_weights(ds_bn_weight) || !downsample_bn_.load_bias(ds_bn_bias) ||
            !downsample_bn_.load_running_mean(ds_bn_mean) || !downsample_bn_.load_running_var(ds_bn_var)) {
            std::cerr << "Error: Failed to load downsample bn for " << prefix << std::endl;
            return false;
        }
    }
    
    return true;
}

Tensor BasicBlock::forward(const Tensor& input) const {
    // Caminho principal: conv1 → bn1 → relu → conv2 → bn2
    Tensor out = conv1_.forward(input);
    out = bn1_.forward(out);
    relu_.forward(out);  // In-place
    
    out = conv2_.forward(out);
    out = bn2_.forward(out);
    
    // Skip connection
    Tensor identity;
    if (has_downsample_) {
        // Downsampling: conv 1x1 + BatchNorm
        identity = downsample_conv_.forward(input);
        identity = downsample_bn_.forward(identity);
    } else {
        // Identity: cópia direta
        identity = input;
    }
    
    // Adicionar skip connection
    int batch = out.batch();
    int channels = out.channels();
    int height = out.height();
    int width = out.width();
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    out(b, c, h, w) += identity(b, c, h, w);
                }
            }
        }
    }
    
    // ReLU final
    relu_.forward(out);  // In-place
    
    return out;
}

