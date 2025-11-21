#!/usr/bin/env bash
set -euo pipefail

# Ensure the build directory exists and is configured
if [[ ! -d "build" ]]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
fi

# Re-configure to make sure CUDA detection is up to date
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# Build only the CUDA packing tests to keep iterations fast
cmake --build build -j6 --target cuda_packing_tests

# Run just the CUDA packing test suite
ctest --test-dir build --output-on-failure -R CudaPack
