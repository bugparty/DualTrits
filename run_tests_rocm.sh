#!/usr/bin/env bash
set -euo pipefail

# Build and run tests using the HIP/ROCm backend.
# Requires ROCm to be installed and hipcc available on PATH.
#
# Prerequisites:
#   - ROCm >= 5.0 installed (https://rocm.docs.amd.com/)
#   - hipcc in PATH (usually /opt/rocm/bin/hipcc)
#   - cmake >= 3.26
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

# Ensure the build directory exists and is configured for HIP
if [[ ! -d "build_hip" ]]; then
    cmake -S . -B build_hip \
        -DCMAKE_BUILD_TYPE=Debug \
        -DDUALTRITS_GPU_BACKEND=HIP
fi

# Re-configure to pick up any changes
cmake -S . -B build_hip \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDUALTRITS_GPU_BACKEND=HIP

# Build only the HIP packing tests
cmake --build build_hip -j"$(nproc)" --target hip_packing_tests

# Run the HIP test suite
ctest --test-dir build_hip --output-on-failure -R HipPack
