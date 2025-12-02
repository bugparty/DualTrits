cmake_minimum_required(VERSION 3.18)
project(fp6_llm LANGUAGES CXX CUDA)

# Option to enable PyTorch support
option(WITH_PYTORCH "Build with PyTorch support" ON)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
# Find Python (needed for PyTorch)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

# Find PyTorch if enabled
if(WITH_PYTORCH)
    # Use Python to find PyTorch cmake path
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        RESULT_VARIABLE TORCH_PREFIX_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
    endif()
    
    find_package(Torch REQUIRED)
    message(STATUS "Found PyTorch: ${TORCH_INSTALL_PREFIX}")
    message(STATUS "PyTorch version: ${Torch_VERSION}")
endif()

# Set default GPU architecture (SM 80 for Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# Source files
set(SOURCES
    csrc/fp6_linear.cu
)

# Add pybind source if building with PyTorch
if(WITH_PYTORCH)
    list(APPEND SOURCES csrc/pybind.cpp)
endif()

# Header files
set(HEADERS
    csrc/fp6_linear.cuh
    csrc/include/configs.h
    csrc/include/kernel_matmul.cuh
    csrc/include/kernel_reduction.cuh
    csrc/include/ptx_cp.async.cuh
    csrc/include/ptx_mma.cuh
    csrc/include/utils_core.cuh
    csrc/include/utils_gmem.cuh
    csrc/include/utils_parallel_dequant.cuh
    csrc/utils/common.h
    csrc/utils/weight_prepacking.h
    csrc/utils/weight_quant.h
    csrc/utils/weight_dequant.h
        csrc/utils/common.h
)

# Create shared library
add_library(fp6 SHARED ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(fp6 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Compile definitions and linking based on PyTorch support
if(WITH_PYTORCH)
    target_include_directories(fp6 PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(fp6 PRIVATE ${TORCH_LIBRARIES} Python::Python CUDA::cudart)
    # Use PyTorch's C++ flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    target_compile_definitions(fp6 PRIVATE NO_PYTORCH)
    target_link_libraries(fp6 PRIVATE CUDA::cudart)
endif()

# CUDA compile options
target_compile_options(fp6 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -maxrregcount=255
        --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
    >
)

# Position independent code for shared library
set_target_properties(fp6 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "fp6"
)

# Link CUDA runtime
target_link_libraries(fp6 PRIVATE CUDA::cudart)
# Position independent code for shared library