# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
cmake_minimum_required(VERSION 3.10)
project(deep_gemm LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

option(DG_WITH_NCCL_GIN "Enable the SM120 NCCL GIN routed MoE runtime" OFF)
if(DG_WITH_NCCL_GIN)
    if(NOT DEFINED ENV{DG_NCCL_ROOT})
        message(FATAL_ERROR "DG_WITH_NCCL_GIN requires DG_NCCL_ROOT")
    endif()
    set(DG_NCCL_ROOT "$ENV{DG_NCCL_ROOT}")
    find_library(DG_NCCL_LIBRARY
        NAMES nccl libnccl.so.2
        PATHS ${DG_NCCL_ROOT}/lib ${DG_NCCL_ROOT}/lib64
        NO_DEFAULT_PATH REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories(deep_gemm/include third-party/cutlass/include third-party/cutlass/tools/util/include third-party/fmt/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/cccl ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# The main Python API entrance
pybind11_add_module(_C csrc/python_api.cpp)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python)
if(DG_WITH_NCCL_GIN)
    target_include_directories(_C PRIVATE ${DG_NCCL_ROOT}/include)
    target_compile_definitions(_C PRIVATE DG_WITH_NCCL_GIN NCCL_HOSTLIB_ONLY)
    target_link_libraries(_C PRIVATE ${DG_NCCL_LIBRARY})
endif()

# Enable kernel code indexing with CMake-based IDEs
cuda_add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)
