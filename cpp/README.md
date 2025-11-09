# ResNet18 C++ Implementation

Implementação C++ do ResNet18 com paralelização customizada.

## Estrutura

```
cpp/
├── CMakeLists.txt      # Sistema de build
├── include/            # Headers
│   ├── tensor.h
│   └── conv2d.h
├── src/               # Implementação
│   ├── tensor.cpp
│   ├── conv2d.cpp
│   └── main.cpp
└── README.md
```

# Como Compilar o Projeto C++

## Pré-requisitos

- **CMake** (versão 3.10 ou superior)
- **Compilador C++17**:
  - Windows: Visual Studio 2019+ ou MinGW
  - Linux: GCC 7+ ou Clang 5+
  - macOS: Xcode Command Line Tools

## Instalação do CMake

### Windows:
1. Baixe de: https://cmake.org/download/
2. Ou use Chocolatey: `choco install cmake`
3. Ou use Visual Studio Installer (inclui CMake)

### Linux:
```bash
sudo apt-get install cmake  # Ubuntu/Debian
sudo yum install cmake      # CentOS/RHEL
```

### macOS:
```bash
brew install cmake
```

## Compilação

### Windows (Visual Studio):

```powershell
cd cpp
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Windows (MinGW):

```powershell
cd cpp
mkdir build
cd build
cmake .. 
cmake --build .
```

### Linux/macOS:

```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

## Executar

Após compilar, execute:

```bash
./build/resnet18
```

## Estrutura de Saída

O programa cria automaticamente:
- `src/validate_results/cpp_outputs/` - Diretório de saídas
- `after_conv1.bin` - Saída da primeira convolução
- `after_conv1_shape.txt` - Shape da saída

## Validação

Após executar, valide:

```bash
cd src/validate_results
python validate.py --layer after_conv1
```