cmake_minimum_required(VERSION 3.10)
project(ResNet18_Parallel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Flags de compilação ===
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# === Diretórios ===
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# === Arquivos fonte ===
file(GLOB_RECURSE SOURCES
    "${SOURCE_DIR}/*.cpp"
)

# === Executável ===
add_executable(resnet18 ${SOURCES})

# === OpenMP ===
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(resnet18 PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(resnet18 PRIVATE PARALLEL_ENABLED)
    message(STATUS "OpenMP habilitado para build paralelo.")
endif()

# === Diretórios de include ===
target_include_directories(resnet18 PRIVATE
    ${INCLUDE_DIR}
    ${SOURCE_DIR}
)

# === Diretórios e definições para pesos e saídas ===
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
