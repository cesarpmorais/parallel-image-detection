#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

/**
 * Classe Tensor para representar tensores 4D (batch, channels, height, width)
 * Layout de memória: row-major (C-style)
 */
class Tensor {
public:
    // Construtor vazio
    Tensor();
    
    // Construtor com shape
    Tensor(int batch, int channels, int height, int width);
    
    // Construtor copiando dados
    Tensor(int batch, int channels, int height, int width, const float* data);
    
    // Destrutor
    ~Tensor() = default;
    
    // Copy constructor e assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Acesso aos dados
    float& operator()(int b, int c, int h, int w);
    const float& operator()(int b, int c, int h, int w) const;
    
    // Acesso linear (para operações eficientes)
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    // Getters
    int batch() const { return batch_; }
    int channels() const { return channels_; }
    int height() const { return height_; }
    int width() const { return width_; }
    int size() const { return data_.size(); }
    std::vector<int> shape() const { return {batch_, channels_, height_, width_}; }
    
    // Operações
    void fill(float value);
    void zeros() { fill(0.0f); }
    void ones() { fill(1.0f); }
    
    // Estatísticas
    float min() const;
    float max() const;
    float mean() const;
    
    // I/O
    bool load_from_bin(const std::string& filepath);
    bool save_to_bin(const std::string& filepath) const;
    bool load_shape_from_txt(const std::string& filepath);
    bool save_shape_to_txt(const std::string& filepath) const;
    
    // Debug
    void print_info(const std::string& name = "Tensor") const;
    
private:
    int batch_;
    int channels_;
    int height_;
    int width_;
    std::vector<float> data_;
    
    // Helper para calcular índice linear
    int index(int b, int c, int h, int w) const {
        return ((b * channels_ + c) * height_ + h) * width_ + w;
    }
};

#endif // TENSOR_H

