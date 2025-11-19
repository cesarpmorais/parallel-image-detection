cmake_minimum_required(VERSION 3.17)
project(ResNet18_CUDA CUDA CXX)

# ===== CUDA Configuration =====
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 4070 uses Ada architecture (86 for CUDA 11.5 compatibility)
# Common architectures:
# - 75: Turing (RTX 2070, 2080, etc.)
# - 80: Ampere (A100, RTX 3070, 3080, etc.)
# - 86: Ada Lovelace (RTX 4060, 4070, 4080, etc.)
# - 89: Ada Lovelace latest (RTX 4090, etc.)

# ===== C++ Configuration =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Compiler Flags =====
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} /W4 /O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")
endif()

# ===== Find CUDA Packages =====
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Optional: Find cuDNN (comment out if not installed)
# find_package(cuDNN REQUIRED)

# If CMake cannot find CUDA on your system, set these manually:
# set(CUDA_INCLUDE_DIRS "/usr/include")
# set(CUDA_LIBRARIES "cudart" "cublas" "cuda")

# ===== Directory Setup =====
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# ===== Gather Source Files =====
file(GLOB_RECURSE CUDA_SOURCES
    "${SOURCE_DIR}/*.cu"
)
# Exclude any test or alternate mains to avoid duplicate `main` symbols.
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*/main\\.cu$")
file(GLOB_RECURSE CXX_SOURCES
    "${SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE HEADERS
    "${INCLUDE_DIR}/*.h"
    "${SOURCE_DIR}/*.h"
)

# ===== Create Executable =====
add_executable(resnet18_cuda
    ${CUDA_SOURCES}
    ${CXX_SOURCES}
    ${HEADERS}
)

# ===== Link Libraries =====
target_link_libraries(resnet18_cuda
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cuda_driver
        # CUDA::curand          # Optional: for random number generation
        # cuDNN::cuDNN          # Uncomment if using cuDNN
)

# ===== Include Directories =====
target_include_directories(resnet18_cuda
    PRIVATE
        ${INCLUDE_DIR}
        ${SOURCE_DIR}
        ${CUDA_INCLUDE_DIRS}
)

# ===== Compile Definitions =====
target_compile_definitions(resnet18_cuda
    PRIVATE
        WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/../weights"
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/test_data"
        OUTPUT_DIR="${CMAKE_SOURCE_DIR}/../src/validate_results/cpp_outputs"
        USE_GPU=1
)

# ===== GPU Architecture Report =====
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_VERSION}")
