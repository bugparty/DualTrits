//
// Created by bowman on 11/9/25.
// CUDA implementation of dual trits packing
//

#include "cuda/dual_trits_pack.cuh"
#include "cuda/kernels/pack_kernels.cuh"

// Note: Kernel templates are defined in pack_kernels.cuh and will be
// instantiated in each compilation unit that uses them. No explicit
// instantiation needed here to avoid multiple definition errors.

// Helper function definition for specialized kernel
__device__ constexpr int pow_of(int exp) {
    int result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= DualTrits::BASE;
    }
    return result;
}

// Helper for warp lane ID
__device__ __forceinline__ int lane_id() {
    return threadIdx.x & 31;
}

// Specialized kernel definition (must be in exactly one compilation unit)
template <>
__global__ void unpack_kernel<10, std::uint32_t>(std::uint32_t const* d_input, DualTrits* d_output, int n) {
    //blockIdx.x is the packed integer index
    //threadIdx.x / 2 is the packed dual-trits index inside the packed integer
    if (blockIdx.x < n) {
        uint32_t packed;
        //first thread in the warp to fetch the int32
        int laneid = lane_id();
        if (laneid == 0) {
            packed = d_input[blockIdx.x];
        }
         // Broadcast lane0's value within warp (mask covers all 32 lanes)
        unsigned mask = 0xffffffff;
        packed = __shfl_sync(mask, packed, 0);   // All lanes receive the same value
        if (laneid < 10) {
            packed = packed / pow_of(laneid * 2);
            auto dir = static_cast<std::uint8_t>(packed % DualTrits::BASE);
            packed /= DualTrits::BASE;
            auto exp = packed % DualTrits::BASE;
            d_output[blockIdx.x * 10 + laneid] = DualTrits(exp, dir);
        }
        //the rest of the warp lanes do nothing (wasted)
    }
}

// Host API implementations
template <std::size_t TritsPerPack, class UInt>
void pack_dual_trits_batch_cuda(DualTrits const* h_input, UInt* h_output, int n) {
    // Allocate device memory
    DualTrits* d_input;
    UInt* d_output;
    
    cudaMalloc(&d_input, n * TritsPerPack * sizeof(DualTrits));
    cudaMalloc(&d_output, n * sizeof(UInt));
    
    // Copy input to device
    cudaError_t err = cudaMemcpy(d_input, h_input, n * TritsPerPack * sizeof(DualTrits), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    pack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(UInt), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_cuda(UInt const* h_input, DualTrits* h_output, int n) {
    // Allocate device memory
    UInt* d_input;
    DualTrits* d_output;
    
    cudaMalloc(&d_input, n * sizeof(UInt));
    cudaMalloc(&d_output, n * TritsPerPack * sizeof(DualTrits));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(UInt), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    unpack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * TritsPerPack * sizeof(DualTrits), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <>
__host__ void unpack_dual_trits_batch_cuda<10, std::uint32_t>(std::uint32_t const* h_input, DualTrits* h_output, int n) {
    // Allocate device memory
    std::uint32_t* d_input;
    DualTrits* d_output;

    cudaMalloc(&d_input, n * sizeof(std::uint32_t));
    cudaMalloc(&d_output, n * 10 * sizeof(DualTrits));

    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    // Launch kernel
    int blockSize = 32;
    int gridSize = n;
    unpack_kernel<10, std::uint32_t><<<gridSize, blockSize>>>(d_input, d_output, n);
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * 10 * sizeof(DualTrits), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
// Explicit template instantiations for host API
template void pack_dual_trits_batch_cuda<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template void pack_dual_trits_batch_cuda<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template void pack_dual_trits_batch_cuda<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template void unpack_dual_trits_batch_cuda<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_cuda<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_cuda<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);
