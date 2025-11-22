//
// Created by bowman on 11/9/25.
// CUDA version of dual trits packing
//

#ifndef PROJECT_FLOAT_CUDA_KERNELS_H
#define PROJECT_FLOAT_CUDA_KERNELS_H

//
// Created by bowman on 11/9/25.
// CUDA implementation of dual trits packing
//

#include "dual_trits_pack.cuh"

// Device function: pack Count dual-trits into UInt
template <std::size_t Count, class UInt>
__device__ constexpr UInt pack_dual_trits_cuda(DualTrits const* dual_trits) {
    UInt packed = 0;
    
    // Encoding order: direction first, then exponent
    UInt exponent = 1;
    for (std::size_t i = 0; i < Count; ++i) {
        packed += exponent * dual_trits[Count - 1 - i].asRawPackedBits();
        exponent *= DualTrits::BASE * DualTrits::BASE;
    }
    return packed;
}

// Device function: unpack UInt into Count dual-trits
template <std::size_t Count, class UInt>
__device__ constexpr void unpack_dual_trits_cuda(UInt packed, DualTrits* out) noexcept {
    for (std::size_t i = 0; i < Count; ++i) {
        auto dir = static_cast<std::uint16_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;
        auto exp = static_cast<std::uint16_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;

        out[Count - 1 - i].setDirection(dir);
        out[Count - 1 - i].setExponent(exp);
    }
}

// Kernel: pack batch of dual-trits arrays
template <std::size_t Count, class UInt>
__global__ void pack_kernel(DualTrits const* d_input, UInt* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        d_output[idx] = pack_dual_trits_cuda<Count, UInt>(&d_input[idx * Count]);
    }
}

// Kernel: unpack batch of packed integers
template <std::size_t Count, class UInt>
__global__ void unpack_kernel(UInt const* d_input, DualTrits* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unpack_dual_trits_cuda<Count, UInt>(d_input[idx], &d_output[idx * Count]);
    }
}

#endif // PROJECT_FLOAT_CUDA_KERNELS_H
