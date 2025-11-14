cmake_minimum_required(VERSION 3.10)
project(ResNet18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Opções de compilação
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Diretórios
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Arquivos fonte
file(GLOB_RECURSE SOURCES 
    "${SOURCE_DIR}/*.cpp"
)

# Arquivos header
file(GLOB_RECURSE HEADERS 
    "${INCLUDE_DIR}/*.h"
    "${SOURCE_DIR}/*.h"
)

# Executável
add_executable(resnet18_openmp main.cpp ...)

# Incluir diretórios
target_include_directories(resnet18 PRIVATE
    ${INCLUDE_DIR}
    ${SOURCE_DIR}
)

# Para Windows: definir caminhos relativos
if(WIN32)
    target_compile_definitions(resnet18 PRIVATE
        WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/../weights"
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/test_data"
        OUTPUT_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/cpp_outputs"
    )
else()
    target_compile_definitions(resnet18 PRIVATE
        WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/../weights"
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/test_data"
        OUTPUT_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/cpp_outputs"
    )
endif()

