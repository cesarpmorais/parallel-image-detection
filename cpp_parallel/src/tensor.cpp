#include "../include/tensor.h"
#include <algorithm>
#include <numeric>

Tensor::Tensor() 
    : batch_(0), channels_(0), height_(0), width_(0) {
}

Tensor::Tensor(int batch, int channels, int height, int width)
    : batch_(batch), channels_(channels), height_(height), width_(width) {
    data_.resize(batch * channels * height * width, 0.0f);
}

Tensor::Tensor(int batch, int channels, int height, int width, const float* data)
    : batch_(batch), channels_(channels), height_(height), width_(width) {
    int size = batch * channels * height * width;
    data_.resize(size);
    std::copy(data, data + size, data_.begin());
}

Tensor::Tensor(const Tensor& other)
    : batch_(other.batch_), channels_(other.channels_), 
      height_(other.height_), width_(other.width_), data_(other.data_) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        batch_ = other.batch_;
        channels_ = other.channels_;
        height_ = other.height_;
        width_ = other.width_;
        data_ = other.data_;
    }
    return *this;
}

float& Tensor::operator()(int b, int c, int h, int w) {
    return data_[index(b, c, h, w)];
}

const float& Tensor::operator()(int b, int c, int h, int w) const {
    return data_[index(b, c, h, w)];
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

float Tensor::min() const {
    if (data_.empty()) return 0.0f;
    return *std::min_element(data_.begin(), data_.end());
}

float Tensor::max() const {
    if (data_.empty()) return 0.0f;
    return *std::max_element(data_.begin(), data_.end());
}

float Tensor::mean() const {
    if (data_.empty()) return 0.0f;
    float sum = std::accumulate(data_.begin(), data_.end(), 0.0f);
    return sum / data_.size();
}

bool Tensor::load_from_bin(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return false;
    }
    
    // Ler tamanho do arquivo
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Verificar se o tamanho corresponde
    size_t expected_size = batch_ * channels_ * height_ * width_ * sizeof(float);
    if (file_size != expected_size) {
        std::cerr << "Error: File size mismatch. Expected " << expected_size 
                  << " bytes, got " << file_size << std::endl;
        return false;
    }
    
    // Ler dados
    file.read(reinterpret_cast<char*>(data_.data()), file_size);
    file.close();
    
    return true;
}

bool Tensor::save_to_bin(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filepath << std::endl;
        return false;
    }
    
    size_t size = data_.size() * sizeof(float);
    file.write(reinterpret_cast<const char*>(data_.data()), size);
    file.close();
    
    return true;
}

bool Tensor::load_shape_from_txt(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open shape file " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    
    int b, c, h, w;
    if (!(iss >> b >> c >> h >> w)) {
        std::cerr << "Error: Invalid shape format" << std::endl;
        return false;
    }
    
    batch_ = b;
    channels_ = c;
    height_ = h;
    width_ = w;
    data_.resize(batch_ * channels_ * height_ * width_, 0.0f);
    
    return true;
}

bool Tensor::save_shape_to_txt(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create shape file " << filepath << std::endl;
        return false;
    }
    
    file << batch_ << " " << channels_ << " " << height_ << " " << width_ << std::endl;
    file.close();
    
    return true;
}

void Tensor::print_info(const std::string& name) const {
    std::cout << name << ": shape=(" << batch_ << ", " << channels_ 
              << ", " << height_ << ", " << width_ << "), "
              << "size=" << size() << ", "
              << "min=" << min() << ", max=" << max() << ", mean=" << mean() << std::endl;
}

