#!/usr/bin/env bash
set -e
# Script to build and run Google Benchmarks

set -e

echo "Building benchmarks in Release mode..."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target packing_benchmarks

echo ""
echo "Running benchmarks..."
./build/packing_benchmarks "$@"
