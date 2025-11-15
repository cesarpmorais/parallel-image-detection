#ifndef BASICBLOCK_H
#define BASICBLOCK_H

#include "tensor.h"
#include "conv2d.h"
#include "batchnorm.h"
#include "relu.h"
#include <string>

/**
 * BasicBlock - Bloco residual do ResNet
 * Estrutura: conv1 → bn1 → relu → conv2 → bn2 → (+skip) → relu
 * 
 * Skip connection pode ser:
 * - Identity: se in_channels == out_channels e stride == 1
 * - Downsampling: conv 1x1 + BatchNorm se precisa ajustar dimensões
 */
class BasicBlock {
public:
    BasicBlock(int in_channels, int out_channels, int stride = 1, 
               bool downsample = false);
    
    // Carregar pesos
    bool load_weights(const std::string& weights_dir, const std::string& prefix);
    
    // Forward pass
    Tensor forward(const Tensor& input) const;
    
    // Getters
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int stride() const { return stride_; }
    
private:
    int in_channels_;
    int out_channels_;
    int stride_;
    bool downsample_;
    
    // Caminho principal
    Conv2D conv1_;
    BatchNorm2D bn1_;
    ReLU relu_;
    Conv2D conv2_;
    BatchNorm2D bn2_;
    
    // Caminho de skip (downsampling se necessário)
    Conv2D downsample_conv_;
    BatchNorm2D downsample_bn_;
    bool has_downsample_;
};

#endif // BASICBLOCK_H

