#include "../include/tensor.h"
#include "../include/conv2d.h"
#include "../include/batchnorm.h"
#include "../include/relu.h"
#include "../include/maxpool.h"
#include "../include/basicblock.h"
#include "../include/adaptiveavgpool.h"
#include "../include/linear.h"
#include <iostream>
#include <string>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define access _access
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

// Diretórios definidos no CMakeLists.txt
#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "../weights"
#endif

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "../src/validate_results/test_data"
#endif

#ifndef OUTPUT_DIR
#define OUTPUT_DIR "../src/validate_results/cpp_outputs"
#endif

void create_output_dir() {
    std::string dir = OUTPUT_DIR;
    
    // Criar diretório se não existir
    #ifdef _WIN32
    if (_access(dir.c_str(), 0) != 0) {
        _mkdir(dir.c_str());
    }
    #else
    if (access(dir.c_str(), F_OK) != 0) {
        mkdir(dir.c_str(), 0755);
    }
    #endif
}

int main() {
    std::cout << "=== ResNet18 C++ Implementation ===" << std::endl;
    std::cout << "Starting with Conv2D layer..." << std::endl;
    
    // Criar diretório de saída
    create_output_dir();
    
    // 1. Carregar entrada de teste
    std::cout << "\n[1] Loading test input..." << std::endl;
    Tensor input;
    
    std::string input_shape_file = std::string(TEST_DATA_DIR) + "/input_shape.txt";
    std::string input_bin_file = std::string(TEST_DATA_DIR) + "/input.bin";
    
    if (!input.load_shape_from_txt(input_shape_file)) {
        std::cerr << "Error: Failed to load input shape" << std::endl;
        return 1;
    }
    
    if (!input.load_from_bin(input_bin_file)) {
        std::cerr << "Error: Failed to load input data" << std::endl;
        return 1;
    }
    
    input.print_info("Input");
    
    // Salvar input para validação
    std::string output_input = std::string(OUTPUT_DIR) + "/input.bin";
    input.save_to_bin(output_input);
    input.save_shape_to_txt(std::string(OUTPUT_DIR) + "/input_shape.txt");
    
    // 2. Criar e carregar Conv2D layer (conv1)
    std::cout << "\n[2] Creating Conv2D layer (conv1)..." << std::endl;
    // conv1: 3->64, 7x7, stride=2, padding=3
    Conv2D conv1(3, 64, 7, 2, 3);
    
    std::string weight_file = std::string(WEIGHTS_DIR) + "/conv1_weight.bin";
    if (!conv1.load_weights(weight_file)) {
        std::cerr << "Error: Failed to load conv1 weights" << std::endl;
        return 1;
    }
    
    std::cout << "Conv1 loaded: " << conv1.in_channels() << " -> " 
              << conv1.out_channels() << ", kernel=" << conv1.kernel_size() 
              << ", stride=" << conv1.stride() << ", padding=" << conv1.padding() << std::endl;
    
    // 3. Forward pass - Conv2D
    std::cout << "\n[3] Running Conv2D forward pass..." << std::endl;
    Tensor after_conv1 = conv1.forward(input);
    after_conv1.print_info("After conv1");
    
    // Salvar after_conv1
    std::string conv1_output = std::string(OUTPUT_DIR) + "/after_conv1.bin";
    std::string conv1_shape = std::string(OUTPUT_DIR) + "/after_conv1_shape.txt";
    after_conv1.save_to_bin(conv1_output);
    after_conv1.save_shape_to_txt(conv1_shape);
    
    // 4. BatchNorm1
    std::cout << "\n[4] Creating BatchNorm1 layer..." << std::endl;
    BatchNorm2D bn1(64);  // 64 canais
    
    std::string bn1_weight_file = std::string(WEIGHTS_DIR) + "/bn1_weight.bin";
    std::string bn1_bias_file = std::string(WEIGHTS_DIR) + "/bn1_bias.bin";
    std::string bn1_mean_file = std::string(WEIGHTS_DIR) + "/bn1_running_mean.bin";
    std::string bn1_var_file = std::string(WEIGHTS_DIR) + "/bn1_running_var.bin";
    
    if (!bn1.load_weights(bn1_weight_file)) {
        std::cerr << "Error: Failed to load bn1 weights" << std::endl;
        return 1;
    }
    
    if (!bn1.load_bias(bn1_bias_file)) {
        std::cerr << "Error: Failed to load bn1 bias" << std::endl;
        return 1;
    }
    
    if (!bn1.load_running_mean(bn1_mean_file)) {
        std::cerr << "Error: Failed to load bn1 running_mean" << std::endl;
        return 1;
    }
    
    if (!bn1.load_running_var(bn1_var_file)) {
        std::cerr << "Error: Failed to load bn1 running_var" << std::endl;
        return 1;
    }
    
    std::cout << "BatchNorm1 loaded: " << bn1.num_channels() << " channels" << std::endl;
    
    // Forward pass - BatchNorm
    std::cout << "\n[5] Running BatchNorm forward pass..." << std::endl;
    Tensor after_bn1 = bn1.forward(after_conv1);
    after_bn1.print_info("After bn1");
    
    // Salvar after_bn1
    std::string bn1_output = std::string(OUTPUT_DIR) + "/after_bn1.bin";
    std::string bn1_shape = std::string(OUTPUT_DIR) + "/after_bn1_shape.txt";
    after_bn1.save_to_bin(bn1_output);
    after_bn1.save_shape_to_txt(bn1_shape);
    
    // 6. ReLU
    std::cout << "\n[6] Running ReLU..." << std::endl;
    ReLU relu;
    relu.forward(after_bn1);  // In-place
    after_bn1.print_info("After relu1");
    
    // Salvar after_relu1
    std::string relu1_output = std::string(OUTPUT_DIR) + "/after_relu1.bin";
    std::string relu1_shape = std::string(OUTPUT_DIR) + "/after_relu1_shape.txt";
    after_bn1.save_to_bin(relu1_output);
    after_bn1.save_shape_to_txt(relu1_shape);
    
    // 7. MaxPool2D
    std::cout << "\n[7] Creating MaxPool2D layer..." << std::endl;
    // MaxPool: 3x3, stride=2, padding=1 (para manter dimensões corretas)
    MaxPool2D maxpool(3, 2, 1);
    
    std::cout << "MaxPool loaded: kernel=" << maxpool.kernel_size() 
              << ", stride=" << maxpool.stride() << ", padding=" << maxpool.padding() << std::endl;
    
    // Forward pass - MaxPool
    std::cout << "\n[8] Running MaxPool forward pass..." << std::endl;
    Tensor after_maxpool = maxpool.forward(after_bn1);
    after_maxpool.print_info("After maxpool");
    
    // Salvar after_maxpool
    std::string maxpool_output = std::string(OUTPUT_DIR) + "/after_maxpool.bin";
    std::string maxpool_shape = std::string(OUTPUT_DIR) + "/after_maxpool_shape.txt";
    after_maxpool.save_to_bin(maxpool_output);
    after_maxpool.save_shape_to_txt(maxpool_shape);
    
    // 9. Layer1: 2x BasicBlock (64→64, sem downsampling)
    std::cout << "\n[9] Creating Layer1 (2x BasicBlock)..." << std::endl;
    
    // BasicBlock 0: 64→64, stride=1, sem downsampling
    BasicBlock layer1_block0(64, 64, 1, false);
    if (!layer1_block0.load_weights(std::string(WEIGHTS_DIR), "layer1_0")) {
        std::cerr << "Error: Failed to load layer1_0 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer1 Block0 loaded" << std::endl;
    
    // BasicBlock 1: 64→64, stride=1, sem downsampling
    BasicBlock layer1_block1(64, 64, 1, false);
    if (!layer1_block1.load_weights(std::string(WEIGHTS_DIR), "layer1_1")) {
        std::cerr << "Error: Failed to load layer1_1 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer1 Block1 loaded" << std::endl;
    
    // Forward pass - Layer1
    std::cout << "\n[10] Running Layer1 forward pass..." << std::endl;
    Tensor after_layer1 = layer1_block0.forward(after_maxpool);
    after_layer1 = layer1_block1.forward(after_layer1);
    after_layer1.print_info("After layer1");
    
    // Salvar after_layer1
    std::string layer1_output = std::string(OUTPUT_DIR) + "/after_layer1.bin";
    std::string layer1_shape = std::string(OUTPUT_DIR) + "/after_layer1_shape.txt";
    after_layer1.save_to_bin(layer1_output);
    after_layer1.save_shape_to_txt(layer1_shape);
    
    // 11. Layer2: 2x BasicBlock (64→128, com downsampling no primeiro bloco)
    std::cout << "\n[11] Creating Layer2 (2x BasicBlock, 64→128)..." << std::endl;
    
    BasicBlock layer2_block0(64, 128, 2, true);  // stride=2, com downsampling
    if (!layer2_block0.load_weights(std::string(WEIGHTS_DIR), "layer2_0")) {
        std::cerr << "Error: Failed to load layer2_0 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer2 Block0 loaded" << std::endl;
    
    BasicBlock layer2_block1(128, 128, 1, false);  // stride=1, sem downsampling
    if (!layer2_block1.load_weights(std::string(WEIGHTS_DIR), "layer2_1")) {
        std::cerr << "Error: Failed to load layer2_1 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer2 Block1 loaded" << std::endl;
    
    Tensor after_layer2 = layer2_block0.forward(after_layer1);
    after_layer2 = layer2_block1.forward(after_layer2);
    after_layer2.print_info("After layer2");
    
    std::string layer2_output = std::string(OUTPUT_DIR) + "/after_layer2.bin";
    std::string layer2_shape = std::string(OUTPUT_DIR) + "/after_layer2_shape.txt";
    after_layer2.save_to_bin(layer2_output);
    after_layer2.save_shape_to_txt(layer2_shape);
    
    // 12. Layer3: 2x BasicBlock (128→256, com downsampling no primeiro bloco)
    std::cout << "\n[12] Creating Layer3 (2x BasicBlock, 128→256)..." << std::endl;
    
    BasicBlock layer3_block0(128, 256, 2, true);
    if (!layer3_block0.load_weights(std::string(WEIGHTS_DIR), "layer3_0")) {
        std::cerr << "Error: Failed to load layer3_0 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer3 Block0 loaded" << std::endl;
    
    BasicBlock layer3_block1(256, 256, 1, false);
    if (!layer3_block1.load_weights(std::string(WEIGHTS_DIR), "layer3_1")) {
        std::cerr << "Error: Failed to load layer3_1 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer3 Block1 loaded" << std::endl;
    
    Tensor after_layer3 = layer3_block0.forward(after_layer2);
    after_layer3 = layer3_block1.forward(after_layer3);
    after_layer3.print_info("After layer3");
    
    std::string layer3_output = std::string(OUTPUT_DIR) + "/after_layer3.bin";
    std::string layer3_shape = std::string(OUTPUT_DIR) + "/after_layer3_shape.txt";
    after_layer3.save_to_bin(layer3_output);
    after_layer3.save_shape_to_txt(layer3_shape);
    
    // 13. Layer4: 2x BasicBlock (256→512, com downsampling no primeiro bloco)
    std::cout << "\n[13] Creating Layer4 (2x BasicBlock, 256→512)..." << std::endl;
    
    BasicBlock layer4_block0(256, 512, 2, true);
    if (!layer4_block0.load_weights(std::string(WEIGHTS_DIR), "layer4_0")) {
        std::cerr << "Error: Failed to load layer4_0 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer4 Block0 loaded" << std::endl;
    
    BasicBlock layer4_block1(512, 512, 1, false);
    if (!layer4_block1.load_weights(std::string(WEIGHTS_DIR), "layer4_1")) {
        std::cerr << "Error: Failed to load layer4_1 weights" << std::endl;
        return 1;
    }
    std::cout << "Layer4 Block1 loaded" << std::endl;
    
    Tensor after_layer4 = layer4_block0.forward(after_layer3);
    after_layer4 = layer4_block1.forward(after_layer4);
    after_layer4.print_info("After layer4");
    
    std::string layer4_output = std::string(OUTPUT_DIR) + "/after_layer4.bin";
    std::string layer4_shape = std::string(OUTPUT_DIR) + "/after_layer4_shape.txt";
    after_layer4.save_to_bin(layer4_output);
    after_layer4.save_shape_to_txt(layer4_shape);
    
    // 14. AdaptiveAvgPool2D
    std::cout << "\n[14] Creating AdaptiveAvgPool2D..." << std::endl;
    AdaptiveAvgPool2D avgpool(1);
    
    Tensor after_avgpool = avgpool.forward(after_layer4);
    after_avgpool.print_info("After avgpool");
    
    std::string avgpool_output = std::string(OUTPUT_DIR) + "/after_avgpool.bin";
    std::string avgpool_shape = std::string(OUTPUT_DIR) + "/after_avgpool_shape.txt";
    after_avgpool.save_to_bin(avgpool_output);
    after_avgpool.save_shape_to_txt(avgpool_shape);
    
    // 15. Flatten (preparar para Linear)
    std::cout << "\n[15] Flattening..." << std::endl;
    // Flatten: [batch, channels, 1, 1] -> [batch, channels]
    // Já está no formato correto, apenas precisamos ajustar para Linear
    
    // 16. Linear (FC)
    std::cout << "\n[16] Creating Linear layer (FC)..." << std::endl;
    Linear fc(512, 1000);
    
    std::string fc_weight_file = std::string(WEIGHTS_DIR) + "/fc_weight.bin";
    std::string fc_bias_file = std::string(WEIGHTS_DIR) + "/fc_bias.bin";
    
    if (!fc.load_weights(fc_weight_file)) {
        std::cerr << "Error: Failed to load fc weights" << std::endl;
        return 1;
    }
    
    if (!fc.load_bias(fc_bias_file)) {
        std::cerr << "Error: Failed to load fc bias" << std::endl;
        return 1;
    }
    
    std::cout << "FC loaded: " << fc.in_features() << " -> " << fc.out_features() << std::endl;
    
    Tensor final_output = fc.forward(after_avgpool);
    final_output.print_info("Final output");
    
    // Salvar after_flatten (mesmo que after_avgpool, mas em formato diferente para validação)
    std::string flatten_output = std::string(OUTPUT_DIR) + "/after_flatten.bin";
    std::string flatten_shape = std::string(OUTPUT_DIR) + "/after_flatten_shape.txt";
    after_avgpool.save_to_bin(flatten_output);
    after_avgpool.save_shape_to_txt(flatten_shape);
    
    // Salvar final_output
    std::string final_output_file = std::string(OUTPUT_DIR) + "/final_output.bin";
    std::string final_output_shape = std::string(OUTPUT_DIR) + "/final_output_shape.txt";
    final_output.save_to_bin(final_output_file);
    final_output.save_shape_to_txt(final_output_shape);
    
    std::cout << "\n[OK] ResNet18 COMPLETE! All layers implemented!" << std::endl;
    std::cout << "Next: Run validation with:" << std::endl;
    std::cout << "  python validate.py  # Validate all layers" << std::endl;
    
    return 0;
}

