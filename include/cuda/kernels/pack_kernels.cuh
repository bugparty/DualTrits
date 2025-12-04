#ifndef PROJECT_FLOAT_CUDA_KERNELS_H
#define PROJECT_FLOAT_CUDA_KERNELS_H

#include "dual_trits_pack.cuh"

template<typename T>
struct div3_magic_traits;

// ------------ uint16_t ------------
// q = (x * 0x5556) >> 16
template<>
struct div3_magic_traits<uint16_t> {
    using T = uint16_t;
    static constexpr uint32_t magic = 0x5556u;

    __device__ __forceinline__ static T div(T x) {
        return (T)(((uint32_t)x * magic) >> 16);
    }

    __device__ __forceinline__ static T mod(T x) {
        T q = div(x);
        return x - q - q - q;   // x - q*3
    }
};

// ------------ uint32_t ------------
// q = (__umulhi(x, 0xAAAAAAABu)) >> 1
template<>
struct div3_magic_traits<uint32_t> {
    using T = uint32_t;
    static constexpr uint32_t magic = 0xAAAAAAABu;

    __device__ __forceinline__ static T div(T x) {
        return __umulhi(x, magic) >> 1;
    }

    __device__ __forceinline__ static T mod(T x) {
        T q = div(x);
        return x - q - q - q;
    }
};

// ------------ uint64_t ------------
// q = (__umul64hi(x, 0xAAAAAAAAAAAAAAABULL)) >> 1
template<>
struct div3_magic_traits<uint64_t> {
    using T = uint64_t;
    static constexpr uint64_t magic = 0xAAAAAAAAAAAAAAABULL;

    __device__ __forceinline__ static T div(T x) {
        return (__umul64hi(x, magic)) >> 1;
    }

    __device__ __forceinline__ static T mod(T x) {
        T q = div(x);
        return x - q - q - q;
    }
};

// ------------ 通用 wrapper 函数 ------------

template<typename T>
__device__ __forceinline__ T div3_magic(T x) {
    return div3_magic_traits<T>::div(x);
}

template<typename T>
__device__ __forceinline__ T mod3_magic(T x) {
    return div3_magic_traits<T>::mod(x);
}

// Kernel: pack batch of dual-trits arrays
template <std::size_t TritsPerPack, class UInt>
__global__ void pack_kernel(DualTrits const*__restrict__ d_input, UInt* __restrict__ d_output, int n) {
    int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int startInputIndex = outputIndex * TritsPerPack;

    if (outputIndex < n) {
        UInt packed = 0;
        UInt multiplier = 1;
    
        // Pack each trit with its multiplier and sum it all
        for (std::size_t dualTritIndex = 0; dualTritIndex < TritsPerPack; ++dualTritIndex) {
            int inputOffset = TritsPerPack - 1 - dualTritIndex;

            packed += multiplier * d_input[startInputIndex + inputOffset].asRawPackedBits();
            multiplier *= DualTrits::BASE * DualTrits::BASE;
        }
        d_output[outputIndex] = packed;
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
            out[TritsPerPack - 1 - i].setDirection(dir);
            out[TritsPerPack - 1 - i].setExponent(exp);
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
