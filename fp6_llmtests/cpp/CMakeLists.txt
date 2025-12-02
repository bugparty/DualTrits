cmake_minimum_required(VERSION 3.18)
project(fp6_llm_tests LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set default GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../fp6_llm/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../fp6_llm/csrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../fp6_llm/csrc/utils
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ======================
# DualTrits Dequant Test
# ======================
add_executable(kernel_test_dualtrit
    kernel_test_dualtrit.cu
)

target_compile_definitions(kernel_test_dualtrit PRIVATE NO_PYTORCH)

target_compile_options(kernel_test_dualtrit PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
    >
)

target_link_libraries(kernel_test_dualtrit PRIVATE
    CUDA::cudart
)

set_target_properties(kernel_test_dualtrit PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ======================
# FP6/FPx Kernel Tests (require libfp6)
# ======================
# These tests require the fp6 library to be built first
# Build fp6 library: cd fp6_llm && cmake -B build_nopython -DWITH_PYTORCH=OFF && cmake --build build_nopython

find_library(FP6_LIBRARY fp6 
    PATHS 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../fp6_llm/build_nopython
)

if(FP6_LIBRARY)
    message(STATUS "Found libfp6: ${FP6_LIBRARY}")
    
    # FP6 GEMM test
    add_executable(kernel_test_fp6 kernel_test_fp6.cu)
    target_compile_definitions(kernel_test_fp6 PRIVATE NO_PYTORCH)
    target_link_libraries(kernel_test_fp6 PRIVATE ${FP6_LIBRARY} CUDA::cudart CUDA::cublas)
    
    # FPx GEMM test
    add_executable(kernel_test_fpx kernel_test_fpx.cu)
    target_compile_definitions(kernel_test_fpx PRIVATE NO_PYTORCH)
    target_link_libraries(kernel_test_fpx PRIVATE ${FP6_LIBRARY} CUDA::cudart CUDA::cublas)
    
    # DualTrits GEMM test
    add_executable(kernel_test_dualtrits kernel_test_dualtrits.cu)
    target_compile_definitions(kernel_test_dualtrits PRIVATE NO_PYTORCH)
    target_compile_options(kernel_test_dualtrits PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
        >
    )
    target_link_libraries(kernel_test_dualtrits PRIVATE ${FP6_LIBRARY} CUDA::cudart CUDA::cublas)
else()
    message(WARNING "libfp6 not found. FP6/FPx/DualTrits GEMM tests will not be built.")
    message(STATUS "To build libfp6: cd fp6_llm && cmake -B build_nopython -DWITH_PYTORCH=OFF && cmake --build build_nopython")
endif()
