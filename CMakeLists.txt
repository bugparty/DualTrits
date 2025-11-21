cmake_minimum_required(VERSION 3.26)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})

project(project_float LANGUAGES CXX)

# Optional CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA support enabled")
else()
    set(ENABLE_CUDA OFF)
    message(STATUS "CUDA not found, building CPU-only version")
endif()
set(CMAKE_CXX_STANDARD 20)

# Option to enable/disable MPFR/GMP support. When OFF, project will skip
# finding GMP/MPFR/MPFRCPP and will not define `USE_MPFR`.
option(USE_MPFR "Enable MPFR/GMP support" ON)
if(USE_MPFR)
        message(STATUS "Building with GMP/MPFR/MPFRCPP support (USE_MPFR=ON)")
        find_package(GMP REQUIRED)
        find_package(MPFR 2.3.0 REQUIRED)
        find_package(MPFRCPP 2.7.0 REQUIRED)
        add_compile_definitions(USE_MPFR)
else()
        message(STATUS "Building without GMP/MPFR/MPFRCPP support (USE_MPFR=OFF)")
        # Ensure the variables referenced later exist as empty values so
        # existing target_link_libraries calls remain valid.
        set(GMP_LIBRARIES "")
        set(MPFR_LIBRARIES "")
        set(MPFRCPP_LIBRARIES "")
        set(GMP_INCLUDES "")
        set(MPFR_INCLUDES "")
        set(MPFRCPP_INCLUDES "")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
if(USE_MPFR)
        include_directories(${GMP_INCLUDES} ${MPFR_INCLUDES} ${MPFRCPP_INCLUDES})
endif()

if(ENABLE_CUDA)
    include_directories(${CMAKE_SOURCE_DIR}/include/cuda)
    # Configure a header file to pass CUDA availability to source code
    add_compile_definitions(HAVE_CUDA)
endif()

add_executable(project_float 
        src/main.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(project_float PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

# =========================
# Testing with GoogleTest
# =========================
include(FetchContent)

# Fetch googletest (header-only gtest is built as targets GTest::gtest and GTest::gtest_main)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Fetch Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
FetchContent_MakeAvailable(googlebenchmark)

include(CTest)
enable_testing()
include(GoogleTest)

add_executable(packing_tests
        tests/test_packing.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(packing_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(packing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(packing_tests)

add_executable(arithmetic_tests
        tests/test_addition_subtraction.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(arithmetic_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(arithmetic_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(arithmetic_tests)

add_executable(multiplication_tests
        tests/test_multiplication.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(multiplication_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(multiplication_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(multiplication_tests)

add_executable(division_tests
        tests/test_division.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(division_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(division_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(division_tests)

add_executable(precision_tests
        tests/test_precision.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(precision_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(precision_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(precision_tests)

if(ENABLE_CUDA)
    add_executable(cuda_packing_tests
            tests/test_packing_cuda.cu
            src/common/DualTrits.cpp
            src/cpu/dual_trits_pack.cpp
            src/cuda/dual_trits_pack.cu)
    target_link_libraries(cuda_packing_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(cuda_packing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(cuda_packing_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(cuda_packing_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
    gtest_discover_tests(cuda_packing_tests)
endif()

# =========================
# Benchmarks with Google Benchmark
# =========================
add_executable(packing_benchmarks
        benchmarks/benchmark_cpu_packing.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(packing_benchmarks PRIVATE benchmark::benchmark ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(packing_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA Benchmarks
if(ENABLE_CUDA)
    add_executable(packing_cuda_benchmarks
            benchmarks/benchmark_packing_cuda.cu
            src/common/DualTrits.cpp
            src/cuda/dual_trits_pack.cu)
    target_link_libraries(packing_cuda_benchmarks PRIVATE benchmark::benchmark ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(packing_cuda_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(packing_cuda_benchmarks PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89")
    target_compile_options(packing_cuda_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
endif()
