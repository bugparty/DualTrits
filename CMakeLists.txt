cmake_minimum_required(VERSION 3.31)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})

project(project_float)
find_package(GMP REQUIRED)
find_package(MPFR 2.3.0 REQUIRED)
find_package(MPFRCPP 2.7.0 REQUIRED)
set(CMAKE_CXX_STANDARD 20)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(project_float main.cpp
        DualTrits.cpp
        DualTrits.h)
target_link_libraries(project_float PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
