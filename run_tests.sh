#!/usr/bin/env bash
set -e
if [ ! -d "build" ]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DUSE_MPFR=ON
fi
cmake --build build -j 6
ctest --test-dir build --output-on-failure