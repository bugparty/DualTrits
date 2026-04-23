//
// HIP/ROCm implementation of dual trits packing.
// Mirrors src/cuda/dual_trits_pack.cu with HIP-compatible API.
//

#include <hip/hip_runtime.h>
#include "hip/dual_trits_pack.h"
#include "hip/kernels/pack_kernels.h"

// Explicit template instantiations for common types
template __global__ void pack_kernel_hip<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template __global__ void pack_kernel_hip<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template __global__ void pack_kernel_hip<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template __global__ void unpack_kernel_stride_hip<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template __global__ void unpack_kernel_stride_hip<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template __global__ void unpack_kernel_stride_hip<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);

// Advanced API: uses pre-allocated device memory, optionally returns timing
template <std::size_t TritsPerPack, class UInt>
void pack_dual_trits_batch_hip_device(
    DualTrits const* d_input,
    UInt* d_output,
    int n,
    float* elapsed_ms
) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (elapsed_ms) {
        // Benchmark mode: use HIP events for timing
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, 0);
        pack_kernel_hip<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);

        hipEventElapsedTime(elapsed_ms, start, stop);

        hipEventDestroy(start);
        hipEventDestroy(stop);
    } else {
        pack_kernel_hip<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        hipDeviceSynchronize();
    }
}

// Simple API: allocates device memory internally
template <std::size_t TritsPerPack, class UInt>
void pack_dual_trits_batch_hip(DualTrits const* h_input, UInt* h_output, int n) {
    DualTrits* d_input;
    UInt* d_output;

    hipMalloc(&d_input, n * TritsPerPack * sizeof(DualTrits));
    hipMalloc(&d_output, n * sizeof(UInt));

    hipError_t err = hipMemcpy(d_input, h_input, n * TritsPerPack * sizeof(DualTrits), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy (HostToDevice) failed: %s\n", hipGetErrorString(err));
        hipFree(d_input);
        hipFree(d_output);
        return;
    }

    pack_dual_trits_batch_hip_device<TritsPerPack, UInt>(d_input, d_output, n, nullptr);

    err = hipMemcpy(h_output, d_output, n * sizeof(UInt), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy (DeviceToHost) failed: %s\n", hipGetErrorString(err));
    }

    hipFree(d_input);
    hipFree(d_output);
}

// Advanced API: stride-loop unpack, uses pre-allocated device memory
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_stride_batch_hip_device(
    UInt const* d_input,
    DualTrits* d_output,
    int n,
    float* elapsed_ms
) {
    // Grid tuned for typical AMD GPU wavefront/CU layout; users can override
    // via the ROCM_BLOCK_SIZE and ROCM_GRID_SIZE environment variables or by
    // choosing architecture targets at build time.
    int blockSize = 128;
    int gridSize = 216 * 2;

    if (elapsed_ms) {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, 0);
        unpack_kernel_stride_hip<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);

        hipEventElapsedTime(elapsed_ms, start, stop);

        hipEventDestroy(start);
        hipEventDestroy(stop);
    } else {
        unpack_kernel_stride_hip<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        hipDeviceSynchronize();
    }
}

template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_hip_device(
    UInt const* d_input,
    DualTrits* d_output,
    int n,
    float* elapsed_ms
) {
    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (elapsed_ms) {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, 0);
        unpack_kernel_hip<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);

        hipEventElapsedTime(elapsed_ms, start, stop);

        hipEventDestroy(start);
        hipEventDestroy(stop);
    } else {
        unpack_kernel_hip<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
        hipDeviceSynchronize();
    }
}

// Simple API: allocates device memory internally
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_hip(UInt const* h_input, DualTrits* h_output, int n) {
    UInt* d_input;
    DualTrits* d_output;

    hipMalloc(&d_input, n * sizeof(UInt));
    hipMalloc(&d_output, n * TritsPerPack * sizeof(DualTrits));

    hipError_t err = hipMemcpy(d_input, h_input, n * sizeof(UInt), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy (HostToDevice) failed: %s\n", hipGetErrorString(err));
        hipFree(d_input);
        hipFree(d_output);
        return;
    }

    unpack_dual_trits_stride_batch_hip_device<TritsPerPack, UInt>(d_input, d_output, n, nullptr);

    err = hipMemcpy(h_output, d_output, n * TritsPerPack * sizeof(DualTrits), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy (DeviceToHost) failed: %s\n", hipGetErrorString(err));
    }

    hipFree(d_input);
    hipFree(d_output);
}

// Explicit template instantiations for host API
template void pack_dual_trits_batch_hip<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template void pack_dual_trits_batch_hip<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template void pack_dual_trits_batch_hip<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template void unpack_dual_trits_batch_hip<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_hip<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_hip<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);

// Explicit template instantiations for device API
template void pack_dual_trits_batch_hip_device<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int, float*);
template void pack_dual_trits_batch_hip_device<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int, float*);
template void pack_dual_trits_batch_hip_device<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int, float*);

template void unpack_dual_trits_batch_hip_device<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_batch_hip_device<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_batch_hip_device<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int, float*);

template void unpack_dual_trits_stride_batch_hip_device<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_stride_batch_hip_device<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int, float*);
template void unpack_dual_trits_stride_batch_hip_device<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int, float*);
