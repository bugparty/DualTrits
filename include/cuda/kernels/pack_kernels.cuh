#ifndef PROJECT_FLOAT_CUDA_KERNELS_H
#define PROJECT_FLOAT_CUDA_KERNELS_H
 

#include "dual_trits_pack.cuh"

// Device function: pack TritsPerPack dual-trits into UInt
template <std::size_t TritsPerPack, class UInt>
__device__ constexpr UInt pack_dual_trits_cuda(DualTrits const* dual_trits) {
    UInt packed = 0;
    UInt multiplier = 1;
    
    // Encoding order: direction first, then exponent
    for (std::size_t i = 0; i < TritsPerPack; ++i) {
        const auto& t = dual_trits[i];
        
        packed += static_cast<UInt>(t.getDirection()) * multiplier;
        multiplier *= DualTrits::BASE;
        
        packed += static_cast<UInt>(t.getExponent()) * multiplier;
        multiplier *= DualTrits::BASE;
    }
    return packed;
}


// Device function: unpack UInt into TritsPerPack dual-trits
template <std::size_t TritsPerPack, class UInt>
__device__ constexpr void unpack_dual_trits_cuda(UInt packed, DualTrits* out) noexcept {
    #pragma unroll
    for (std::size_t i = 0; i < TritsPerPack; ++i) {
        auto dir = static_cast<std::uint8_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;
        auto exp = static_cast<std::uint8_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;
        out[i].setDirection(dir);
        out[i].setExponent(exp);
    }
}

// Kernel: pack batch of dual-trits arrays
template <std::size_t TritsPerPack, class UInt>
__global__ void pack_kernel(DualTrits const* d_input, UInt* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        d_output[idx] = pack_dual_trits_cuda<TritsPerPack, UInt>(&d_input[idx * TritsPerPack]);
    }
}

// Standard unpack kernel: one thread per packed integer
template <std::size_t TritsPerPack, class UInt>
__global__ void unpack_kernel(UInt const* d_input, DualTrits* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unpack_dual_trits_cuda<TritsPerPack, UInt>(d_input[idx], &d_output[idx * TritsPerPack]);
    }
}

// Helper function to get precomputed power of BASE (defined in dual_trits_pack.cu)
template <typename UInt>
__device__ __forceinline__ UInt pow_of_base(int exp);

// Optimized warp-cooperative unpack kernel declaration (defined in dual_trits_pack.cu)
template <std::size_t TritsPerPack, class UInt>
__global__ void unpack_kernel_warp(UInt const* d_input, DualTrits* d_output, int n);

#endif // PROJECT_FLOAT_CUDA_KERNELS_H
