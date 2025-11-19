//
// Created by bowman on 11/9/25.
// CUDA kernels for packing/unpacking dual trits
//

#ifndef PROJECT_FLOAT_PACK_KERNELS_CU
#define PROJECT_FLOAT_PACK_KERNELS_CU

#include "cuda/dual_trits_pack.cuh"

// Device function: pack Count dual-trits into UInt
template <std::size_t Count, class UInt>
__device__ constexpr UInt pack_dual_trits_cuda(DualTrits const* dual_trits) {
    UInt packed = 0;
    UInt multiplier = 1;

    constexpr auto pow_base = [](size_t exponent) constexpr {
        UInt result = 1;
        for (size_t loops = 0; loops < exponent; loops++) {
            result *= DualTrits::BASE;
        }
        return result;
    };
    
    // Encoding order: direction first, then exponent
    for (std::size_t i = 0; i < Count; ++i) {
        packed += pow_base(2 * i) * dual_trits[i].asRawPackedBits();
    }
    return packed;
}

// Device function: unpack UInt into Count dual-trits
template <std::size_t Count, class UInt>
__device__ constexpr void unpack_dual_trits_cuda(UInt packed, DualTrits* out) noexcept {

    constexpr auto pow_base = [](size_t exponent) constexpr {
        UInt result = 1;
        for (size_t loops = 0; loops < exponent; loops++) {
            result *= DualTrits::BASE;
        }
        return result;
    };

    for (std::size_t i = 0; i < Count; ++i) {
        UInt bits = packed / pow_base(2 * i);
        auto dir = static_cast<std::uint16_t>(bits % DualTrits::BASE);
        bits /= DualTrits::BASE;
        auto exp = static_cast<std::uint16_t>(bits % DualTrits::BASE);

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

#endif // PROJECT_FLOAT_PACK_KERNELS_CU
