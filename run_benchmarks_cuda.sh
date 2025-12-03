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
if [ ${OUTPUT_JSON} = false ]; then
    echo "Building cuda benchmarks in Release mode..."  
    
fi  

cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_MPFR=OFF  > /dev/null
cmake --build build --target packing_cuda_benchmarks > /dev/null
if [ ${OUTPUT_JSON} = false ]; then
    echo ""
    echo "Running cuda benchmarks..."
fi

if [ "$OUTPUT_JSON" = true ]; then
    
    ./build/packing_cuda_benchmarks --benchmark_format=json "${BENCHMARK_ARGS[@]}"
else
    ./build/packing_cuda_benchmarks "${BENCHMARK_ARGS[@]}"
fi