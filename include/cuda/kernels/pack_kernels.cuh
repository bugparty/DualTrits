#ifndef PROJECT_FLOAT_CUDA_KERNELS_H
#define PROJECT_FLOAT_CUDA_KERNELS_H

#include "dual_trits_pack.cuh"
// Kernel: pack batch of dual-trits arrays
template <std::size_t TritsPerPack, class UInt>
__global__ void pack_kernel(DualTrits const*__restrict__ d_input, UInt* __restrict__ d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        DualTrits const * dual_trits = &d_input[idx * TritsPerPack];
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
        d_output[idx] = packed;
    }
}

// Kernel: unpack batch of packed integers
template <std::size_t TritsPerPack, class UInt>
__global__ void unpack_kernel_stride(UInt const* __restrict__ d_input, DualTrits* __restrict__ d_output,const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    #pragma unroll
    for (int idx = tid; idx < n; idx += stride) {
        auto packed = d_input[idx];
        DualTrits* out = &d_output[idx * TritsPerPack];
        for (std::size_t i = 0; i < TritsPerPack; ++i) {
            auto dir = static_cast<std::uint16_t>(packed % DualTrits::BASE);
            packed /= DualTrits::BASE;
            auto exp = static_cast<std::uint16_t>(packed % DualTrits::BASE);
            packed /= DualTrits::BASE;
            out[i].setDirection(dir);
            out[i].setExponent(exp);
        }
    }

}
template <std::size_t TritsPerPack, class UInt>
__global__ void unpack_kernel(UInt const* d_input, DualTrits* d_output, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        auto packed = d_input[tid];
        DualTrits* out = &d_output[tid * TritsPerPack];
        for (std::size_t i = 0; i < TritsPerPack; ++i) {
            auto dir = static_cast<std::uint16_t>(packed % DualTrits::BASE);
            packed /= DualTrits::BASE;
            auto exp = static_cast<std::uint16_t>(packed % DualTrits::BASE);
            packed /= DualTrits::BASE;
            out[i].setDirection(dir);
            out[i].setExponent(exp);
        }
    }

}


#endif // PROJECT_FLOAT_CUDA_KERNELS_H
