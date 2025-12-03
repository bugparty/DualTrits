//
// Created by bowman on 11/9/25.
// CUDA implementation of dual trits packing
//

#include "cuda/dual_trits_pack.cuh"
#include "cuda/kernels/pack_kernels.cuh"

// Explicit template instantiations for common types
template __global__ void pack_kernel<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template __global__ void pack_kernel<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template __global__ void pack_kernel<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template __global__ void unpack_kernel_stride<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template __global__ void unpack_kernel_stride<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template __global__ void unpack_kernel_stride<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);

// Advanced API: uses pre-allocated device memory, optionally returns timing
template <std::size_t TritsPerPack, class UInt>
void pack_dual_trits_batch_cuda_device(
    DualTrits const* d_input,
    UInt* d_output,
    int n,
    float* elapsed_ms
) {
    // Setup grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    if (elapsed_ms) {
        // Benchmark mode: use CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        pack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(elapsed_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // Normal mode: just launch kernel
        pack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
}

// Simple API: allocates device memory internally
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
    
    // Launch kernel using device API
    pack_dual_trits_batch_cuda_device<TritsPerPack, UInt>(d_input, d_output, n, nullptr);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(UInt), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Advanced API: uses pre-allocated device memory, optionally returns timing
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_stride_batch_cuda_device(
    UInt const* d_input,
    DualTrits* d_output,
    int n,
    float* elapsed_ms
) {
    // Setup grid and block dimensions
    // hardcoded for 5060ti
    int blockSize = 128;
    int gridSize = 216*2;
    
    if (elapsed_ms) {
        // Benchmark mode: use CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        unpack_kernel_stride<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(elapsed_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // Normal mode: just launch kernel
        unpack_kernel_stride<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
}

template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_cuda_device(
        UInt const* d_input,
        DualTrits* d_output,
        int n,
        float* elapsed_ms
) {
    // Setup grid and block dimensions
    // hardcoded for 5060ti
    int blockSize = 128;
    int gridSize = (n+ blockSize - 1) / blockSize;

    if (elapsed_ms) {
        // Benchmark mode: use CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        unpack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(elapsed_ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // Normal mode: just launch kernel
        unpack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
}

// Simple API: allocates device memory internally
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_cuda(UInt const* h_input, DualTrits* h_output, int n) {
    // Allocate device memory
    UInt* d_input;
    DualTrits* d_output;
    
    cudaMalloc(&d_input, n * sizeof(UInt));
    cudaMalloc(&d_output, n * TritsPerPack * sizeof(DualTrits));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(UInt), cudaMemcpyHostToDevice);
    
    // Launch kernel using device API
    unpack_dual_trits_stride_batch_cuda_device<TritsPerPack, UInt>(d_input, d_output, n, nullptr);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * TritsPerPack * sizeof(DualTrits), cudaMemcpyDeviceToHost);
    
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

// Explicit template instantiations for device API
template void pack_dual_trits_batch_cuda_device<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int, float*);
template void pack_dual_trits_batch_cuda_device<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int, float*);
template void pack_dual_trits_batch_cuda_device<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int, float*);

template void unpack_dual_trits_batch_cuda_device<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_batch_cuda_device<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_batch_cuda_device<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int, float*);

template void unpack_dual_trits_stride_batch_cuda_device<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_stride_batch_cuda_device<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_stride_batch_cuda_device<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int, float*);
