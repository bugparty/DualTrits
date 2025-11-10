//
// Created by bowman on 11/9/25.
// CUDA implementation of dual trits packing
//

#include "dual_trits_pack.cuh"
#include "kernels/pack_kernels.cu"

// Explicit template instantiations for common types
template __global__ void pack_kernel<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template __global__ void pack_kernel<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template __global__ void pack_kernel<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template __global__ void unpack_kernel<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template __global__ void unpack_kernel<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template __global__ void unpack_kernel<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);

// Host API implementations
template <std::size_t Count, class UInt>
void pack_dual_trits_batch_cuda(DualTrits const* h_input, UInt* h_output, int n) {
    // Allocate device memory
    DualTrits* d_input;
    UInt* d_output;
    
    cudaMalloc(&d_input, n * Count * sizeof(DualTrits));
    cudaMalloc(&d_output, n * sizeof(UInt));
    
    // Copy input to device
    cudaError_t err = cudaMemcpy(d_input, h_input, n * Count * sizeof(DualTrits), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    pack_kernel<Count, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(UInt), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

template <std::size_t Count, class UInt>
void unpack_dual_trits_batch_cuda(UInt const* h_input, DualTrits* h_output, int n) {
    // Allocate device memory
    UInt* d_input;
    DualTrits* d_output;
    
    cudaMalloc(&d_input, n * sizeof(UInt));
    cudaMalloc(&d_output, n * Count * sizeof(DualTrits));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(UInt), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    unpack_kernel<Count, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * Count * sizeof(DualTrits), cudaMemcpyDeviceToHost);
    
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
