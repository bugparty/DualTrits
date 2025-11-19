#!/usr/bin/env bash
set -e
# Script to build and run Google Benchmarks

set -e

# Parse arguments
OUTPUT_JSON=false
BENCHMARK_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            OUTPUT_JSON=true
            shift
            ;;
        *)
            BENCHMARK_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Building cuda benchmarks in Release mode..."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target packing_cuda_benchmarks

echo ""
echo "Running cuda benchmarks..."

if [ "$OUTPUT_JSON" = true ]; then
    echo "Outputting results in JSON format..."
    ./build/packing_cuda_benchmarks --benchmark_format=json "${BENCHMARK_ARGS[@]}"
else
    ./build/packing_cuda_benchmarks "${BENCHMARK_ARGS[@]}"
fi