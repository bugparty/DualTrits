cmake_minimum_required(VERSION 3.28)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})

project(project_float)
find_package(GMP REQUIRED)
find_package(MPFR 2.3.0 REQUIRED)
find_package(MPFRCPP 2.7.0 REQUIRED)
set(CMAKE_CXX_STANDARD 20)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(project_float main.cpp
        DualTrits.cpp
        DualTrits.h
        dual_trits_pack.hpp
        dual_trits_pack.cpp)
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

include(CTest)
enable_testing()
include(GoogleTest)

add_executable(packing_tests
        tests/test_packing.cpp
        DualTrits.cpp
        dual_trits_pack.cpp)
target_link_libraries(packing_tests PRIVATE GTest::gtest GTest::gtest_main ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
target_include_directories(packing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(packing_tests)
