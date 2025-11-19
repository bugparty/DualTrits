cmake_minimum_required(VERSION 3.28)
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

find_package(GMP REQUIRED)
find_package(MPFR 2.3.0 REQUIRED)
find_package(MPFRCPP 2.7.0 REQUIRED)
set(CMAKE_CXX_STANDARD 20)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
# include_directories(${CMAKE_SOURCE_DIR}/src/common)
# include_directories(${CMAKE_SOURCE_DIR}/src/cpu)
include_directories(${CMAKE_SOURCE_DIR}/include/common)
include_directories(${CMAKE_SOURCE_DIR}/include/cpu)

if(ENABLE_CUDA)
    include_directories(${CMAKE_SOURCE_DIR}/include/cuda)
endif()

# Configure a header file to pass CUDA availability to source code
if(ENABLE_CUDA)
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
        tests/test_arithmetic.cpp
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

# =========================
# Benchmarks with Google Benchmark
# =========================
add_executable(packing_benchmarks
        benchmarks/benchmark_cpu_packing.cpp
        src/common/DualTrits.cpp
        src/cpu/dual_trits_pack.cpp)
target_link_libraries(packing_benchmarks PRIVATE benchmark::benchmark ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(packing_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

