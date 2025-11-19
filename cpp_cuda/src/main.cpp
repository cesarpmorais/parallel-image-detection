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
// Timing
#include <chrono>
#include <vector>
#include <iomanip>
#include <dirent.h>
#include <sys/types.h>
#include <algorithm>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define access _access
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "../weights"
#endif

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "../src/validate_results/test_data"
#endif

#ifndef OUTPUT_DIR
#define OUTPUT_DIR "../src/validate_results/cpp_outputs"
#endif

void create_output_dir()
{
    std::string dir = OUTPUT_DIR;
#ifdef _WIN32
    if (_access(dir.c_str(), 0) != 0)
    {
        _mkdir(dir.c_str());
    }
#else
    if (access(dir.c_str(), F_OK) != 0)
    {
        mkdir(dir.c_str(), 0755);
    }
#endif
}

int main(int argc, char **argv)
{
    std::cout << "=== ResNet18 C++ Implementation ===" << std::endl;
    std::cout << "Starting with Conv2D layer..." << std::endl;

    using Clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    std::string images_dir = "";
    int max_images = 0;
    int repeat = 1;
    for (int i = 1; i < argc; ++i)
    {
        std::string a(argv[i]);
        if (a == "--images-dir" && i + 1 < argc)
        {
            images_dir = argv[++i];
        }
        else if ((a == "--max-images" || a == "-n" || a == "--num-images") && i + 1 < argc)
        {
            max_images = std::stoi(argv[++i]);
        }
        else if ((a == "--repeat" || a == "--runs") && i + 1 < argc)
        {
            repeat = std::stoi(argv[++i]);
            if (repeat < 1)
                repeat = 1;
        }
    }

    create_output_dir();

    std::vector<std::string> inputs;
    if (!images_dir.empty())
    {
        DIR *d = opendir(images_dir.c_str());
        if (!d)
        {
            std::cerr << "Error: images-dir does not exist or cannot be opened: " << images_dir << std::endl;
            return 1;
        }
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL)
        {
            if (ent->d_type != DT_REG && ent->d_type != DT_UNKNOWN)
                continue;
            std::string name(ent->d_name);
            auto pos = name.rfind('.');
            if (pos == std::string::npos)
                continue;
            std::string ext = name.substr(pos);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".bin")
            {
                if (name.find("final_output") != std::string::npos)
                    continue;
                if (name.rfind("after_", 0) == 0)
                    continue;
                inputs.push_back(images_dir + std::string("/") + name);
            }
        }
        closedir(d);
        std::sort(inputs.begin(), inputs.end());
        if (max_images > 0 && (int)inputs.size() > max_images)
            inputs.resize(max_images);
    }
    else
    {
        inputs.push_back(std::string(TEST_DATA_DIR) + "/input.bin");
    }

    if (inputs.empty())
    {
        std::cerr << "No input .bin files found to process." << std::endl;
        return 1;
    }

    std::cout << "Will process " << inputs.size() << " input(s)." << std::endl;

    std::cout << "\n[2] Creating and loading layers..." << std::endl;
    Conv2D conv1(3, 64, 7, 2, 3);
    std::string conv1_path = std::string(WEIGHTS_DIR) + "/conv1_weight.bin";
    if (!conv1.load_weights(conv1_path))
    {
        std::cerr << "Error: Failed to load conv1 weights" << std::endl;
        return 1;
    }
    BatchNorm2D bn1(64);
    if (!bn1.load_weights(std::string(WEIGHTS_DIR) + "/bn1_weight.bin") ||
        !bn1.load_bias(std::string(WEIGHTS_DIR) + "/bn1_bias.bin") ||
        !bn1.load_running_mean(std::string(WEIGHTS_DIR) + "/bn1_running_mean.bin") ||
        !bn1.load_running_var(std::string(WEIGHTS_DIR) + "/bn1_running_var.bin"))
    {
        std::cerr << "Error: Failed to load bn1 parameters" << std::endl;
        return 1;
    }

    ReLU relu;
    MaxPool2D maxpool(3, 2, 1);

    BasicBlock layer1_block0(64, 64, 1, false);
    layer1_block0.load_weights(std::string(WEIGHTS_DIR), "layer1_0");
    BasicBlock layer1_block1(64, 64, 1, false);
    layer1_block1.load_weights(std::string(WEIGHTS_DIR), "layer1_1");

    BasicBlock layer2_block0(64, 128, 2, true);
    layer2_block0.load_weights(std::string(WEIGHTS_DIR), "layer2_0");
    BasicBlock layer2_block1(128, 128, 1, false);
    layer2_block1.load_weights(std::string(WEIGHTS_DIR), "layer2_1");

    BasicBlock layer3_block0(128, 256, 2, true);
    layer3_block0.load_weights(std::string(WEIGHTS_DIR), "layer3_0");
    BasicBlock layer3_block1(256, 256, 1, false);
    layer3_block1.load_weights(std::string(WEIGHTS_DIR), "layer3_1");

    BasicBlock layer4_block0(256, 512, 2, true);
    layer4_block0.load_weights(std::string(WEIGHTS_DIR), "layer4_0");
    BasicBlock layer4_block1(512, 512, 1, false);
    layer4_block1.load_weights(std::string(WEIGHTS_DIR), "layer4_1");

    AdaptiveAvgPool2D avgpool(1);
    Linear fc(512, 1000);
    fc.load_weights(std::string(WEIGHTS_DIR) + "/fc_weight.bin");
    fc.load_bias(std::string(WEIGHTS_DIR) + "/fc_bias.bin");

    std::string global_csv = std::string(OUTPUT_DIR) + "/timings.csv";
    bool need_header = (access(global_csv.c_str(), F_OK) != 0);

    for (size_t idx = 0; idx < inputs.size(); ++idx)
    {
        const std::string bin_path = inputs[idx];
        std::string fname = bin_path.substr(bin_path.find_last_of('/') + 1);
        std::string stem = fname.substr(0, fname.find_last_of('.'));

        std::cout << "Processing [" << (idx + 1) << "/" << inputs.size() << "] " << fname << std::endl;

        for (int r = 0; r < repeat; ++r)
        {
            // Create input tensor with expected shape: (1, 3, 224, 224)
            Tensor input(1, 3, 224, 224);

            // Load raw float data
            std::ifstream input_file(bin_path, std::ios::binary);
            if (!input_file.is_open())
            {
                std::cerr << "Error: Failed to open input file: " << bin_path << std::endl;
                return 1;
            }

            // Check file size
            input_file.seekg(0, std::ios::end);
            size_t file_size = input_file.tellg();
            input_file.seekg(0, std::ios::beg);

            size_t expected_size = input.numel() * sizeof(float);
            if (file_size != expected_size)
            {
                std::cerr << "Error: Input file size mismatch. Expected " << expected_size
                          << " bytes, got " << file_size << " bytes" << std::endl;
                return 1;
            }

            // Read raw float data
            input_file.read(reinterpret_cast<char *>(input.data_ptr()), file_size);
            input_file.close();

            std::vector<std::pair<std::string, double>> img_timings;

            // Conv1
            auto t_conv0 = Clock::now();
            Tensor after_conv1 = conv1.forward(input);
            auto t_conv1 = Clock::now();
            img_timings.emplace_back("conv1", ms(t_conv1 - t_conv0).count());
            after_conv1.save_to_bin(std::string(OUTPUT_DIR) + "/after_conv1.bin");
            after_conv1.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_conv1_shape.txt");

            // BN1
            auto t_bn0 = Clock::now();
            Tensor after_bn1 = bn1.forward(after_conv1);
            auto t_bn1 = Clock::now();
            img_timings.emplace_back("bn1", ms(t_bn1 - t_bn0).count());
            after_bn1.save_to_bin(std::string(OUTPUT_DIR) + "/after_bn1.bin");
            after_bn1.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_bn1_shape.txt");

            // ReLU
            auto t_relu0 = Clock::now();
            relu.forward(after_bn1);
            auto t_relu1 = Clock::now();
            img_timings.emplace_back("relu1", ms(t_relu1 - t_relu0).count());
            after_bn1.save_to_bin(std::string(OUTPUT_DIR) + "/after_relu1.bin");
            after_bn1.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_relu1_shape.txt");

            // MaxPool
            auto t_mp0 = Clock::now();
            Tensor after_maxpool = maxpool.forward(after_bn1);
            auto t_mp1 = Clock::now();
            img_timings.emplace_back("maxpool", ms(t_mp1 - t_mp0).count());
            after_maxpool.save_to_bin(std::string(OUTPUT_DIR) + "/after_maxpool.bin");
            after_maxpool.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_maxpool_shape.txt");

            // Layers
            auto t_l10 = Clock::now();
            Tensor after_layer1 = layer1_block0.forward(after_maxpool);
            after_layer1 = layer1_block1.forward(after_layer1);
            auto t_l11 = Clock::now();
            img_timings.emplace_back("layer1", ms(t_l11 - t_l10).count());
            after_layer1.save_to_bin(std::string(OUTPUT_DIR) + "/after_layer1.bin");
            after_layer1.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_layer1_shape.txt");

            auto t_l20 = Clock::now();
            Tensor after_layer2 = layer2_block0.forward(after_layer1);
            after_layer2 = layer2_block1.forward(after_layer2);
            auto t_l21 = Clock::now();
            img_timings.emplace_back("layer2", ms(t_l21 - t_l20).count());
            after_layer2.save_to_bin(std::string(OUTPUT_DIR) + "/after_layer2.bin");
            after_layer2.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_layer2_shape.txt");

            auto t_l30 = Clock::now();
            Tensor after_layer3 = layer3_block0.forward(after_layer2);
            after_layer3 = layer3_block1.forward(after_layer3);
            auto t_l31 = Clock::now();
            img_timings.emplace_back("layer3", ms(t_l31 - t_l30).count());
            after_layer3.save_to_bin(std::string(OUTPUT_DIR) + "/after_layer3.bin");
            after_layer3.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_layer3_shape.txt");

            auto t_l40 = Clock::now();
            Tensor after_layer4 = layer4_block0.forward(after_layer3);
            after_layer4 = layer4_block1.forward(after_layer4);
            auto t_l41 = Clock::now();
            img_timings.emplace_back("layer4", ms(t_l41 - t_l40).count());
            after_layer4.save_to_bin(std::string(OUTPUT_DIR) + "/after_layer4.bin");
            after_layer4.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_layer4_shape.txt");

            // AvgPool
            auto t_avg0 = Clock::now();
            Tensor after_avgpool = avgpool.forward(after_layer4);
            auto t_avg1 = Clock::now();
            img_timings.emplace_back("avgpool", ms(t_avg1 - t_avg0).count());
            after_avgpool.save_to_bin(std::string(OUTPUT_DIR) + "/after_avgpool.bin");
            after_avgpool.save_shape_to_txt(std::string(OUTPUT_DIR) + "/after_avgpool_shape.txt");

            // FC
            auto t_fc0 = Clock::now();
            Tensor final_output = fc.forward(after_avgpool);
            auto t_fc1 = Clock::now();
            img_timings.emplace_back("fc", ms(t_fc1 - t_fc0).count());

            // Save final output for validation
            final_output.save_to_bin(std::string(OUTPUT_DIR) + "/final_output_" + stem + ".bin");

            double total = 0.0;
            for (const auto &p : img_timings)
                total += p.second;

            std::ofstream g(global_csv);
            if (g.is_open())
            {
                g << "layer,time_ms\n";
                for (const auto &p : img_timings)
                    g << p.first << "," << p.second << "\n";
                g << "total," << total << "\n";
                g.close();
            }

            std::cout << "Run " << (r + 1) << ": total=" << total << " ms" << std::endl;
        }
    }

    std::cout << "Processing complete." << std::endl;
    return 0;
}
