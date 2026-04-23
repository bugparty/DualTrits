//
// HIP/ROCm version of dual trits packing declarations.
// Mirrors include/cuda/dual_trits_pack.cuh with HIP-compatible API.
//

#ifndef PROJECT_FLOAT_HIP_PACKING_H
#define PROJECT_FLOAT_HIP_PACKING_H

#include <cstdint>
#include "common/DualTrits.hpp"

// Simple API: allocates device memory internally, no timing
template <std::size_t TritsPerPack, class UInt>
void pack_dual_trits_batch_hip(DualTrits const* h_input, UInt* h_output, int n);

template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_hip(UInt const* h_input, DualTrits* h_output, int n);

// Advanced API: uses pre-allocated device memory, optionally returns timing
template <std::size_t TritsPerPack, class UInt>
void pack_dual_trits_batch_hip_device(
    DualTrits const* d_input,   // device memory
    UInt* d_output,              // device memory
    int n,
    float* elapsed_ms = nullptr  // if not null, returns kernel execution time
);

template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_hip_device(
    UInt const* d_input,        // device memory
    DualTrits* d_output,         // device memory
    int n,
    float* elapsed_ms = nullptr  // if not null, returns kernel execution time
);

template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_stride_batch_hip_device(
    UInt const* d_input,        // device memory
    DualTrits* d_output,         // device memory
    int n,
    float* elapsed_ms = nullptr  // if not null, returns kernel execution time
);

#endif // PROJECT_FLOAT_HIP_PACKING_H
