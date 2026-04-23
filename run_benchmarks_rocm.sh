#!/usr/bin/env bash
set -euo pipefail

# Build and run benchmarks using the HIP/ROCm backend.
# Requires ROCm to be installed and hipcc available on PATH.
#
# Prerequisites:
#   - ROCm >= 5.0 installed (https://rocm.docs.amd.com/)
#   - hipcc in PATH (usually /opt/rocm/bin/hipcc)
#   - cmake >= 3.26
#
# Usage:
#   ./run_benchmarks_rocm.sh [--json] [extra benchmark args...]
#
# Environment variables (optional):
#   CMAKE_HIP_COMPILER   – override path to hipcc
#   ROCM_PATH            – ROCm installation prefix (default: /opt/rocm)

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# Make sure hipcc is reachable
if ! command -v hipcc &>/dev/null; then
    export PATH="${ROCM_PATH}/bin:${PATH}"
fi

if ! command -v hipcc &>/dev/null; then
    echo "ERROR: hipcc not found. Install ROCm and ensure it is on PATH." >&2
    exit 1
fi

echo "Using HIP compiler: $(command -v hipcc)"

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

if [[ "${OUTPUT_JSON}" == "false" ]]; then
    echo "Building HIP/ROCm benchmarks in Release mode..."
fi

cmake -B build_hip \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MPFR=OFF \
    -DDUALTRITS_GPU_BACKEND=HIP \
    > /dev/null

cmake --build build_hip --target packing_hip_benchmarks > /dev/null

if [[ "${OUTPUT_JSON}" == "false" ]]; then
    echo ""
    echo "Running HIP/ROCm benchmarks..."
fi

if [[ "${OUTPUT_JSON}" == "true" ]]; then
    ./build_hip/packing_hip_benchmarks --benchmark_format=json "${BENCHMARK_ARGS[@]}"
else
    ./build_hip/packing_hip_benchmarks "${BENCHMARK_ARGS[@]}"
fi
