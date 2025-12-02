//
// Created by bowman on 11/9/25.
// CUDA implementation of dual trits packing
//

#include "cuda/dual_trits_pack.cuh"
#include "cuda/kernels/pack_kernels.cuh"

// Note: Kernel templates are defined in pack_kernels.cuh and will be
// instantiated in each compilation unit that uses them. No explicit
// instantiation needed here to avoid multiple definition errors.

// Precomputed power table for BASE^n (up to BASE^40 for 20 dual-trits)
// Each dual-trit uses 2 trits, so max exponent is 40
__device__ __constant__ unsigned long long c_powers_base3[] = {
    1ULL, 3ULL, 9ULL, 27ULL, 81ULL, 243ULL, 729ULL, 2187ULL,
    6561ULL, 19683ULL, 59049ULL, 177147ULL, 531441ULL, 1594323ULL,
    4782969ULL, 14348907ULL, 43046721ULL, 129140163ULL, 387420489ULL,
    1162261467ULL, 3486784401ULL, 10460353203ULL, 31381059609ULL,
    94143178827ULL, 282429536481ULL, 847288609443ULL, 2541865828329ULL,
    7625597484987ULL, 22876792454961ULL, 68630377364883ULL, 205891132094649ULL,
    617673396283947ULL, 1853020188851841ULL, 5559060566555523ULL,
    16677181699666569ULL, 50031545098999707ULL, 150094635296999121ULL,
    450283905890997363ULL, 1350851717672992089ULL, 4052555153018976267ULL,
    12157665459056928801ULL
};

// Helper function to get precomputed power
template <typename UInt>
__device__ __forceinline__ UInt pow_of_base(int exp) {
    return static_cast<UInt>(c_powers_base3[exp]);
}

// Helper for warp lane ID
__device__ __forceinline__ int lane_id() {
    return threadIdx.x & 31;
}

// Optimized warp-cooperative unpack kernel (generic template)
template <std::size_t TritsPerPack, class UInt>
__global__ void unpack_kernel_warp(UInt const* d_input, DualTrits* d_output, int n) {
    // blockIdx.x is the packed integer index
    // threadIdx.x is the lane ID within the warp
    if (blockIdx.x < n) {
        UInt packed;
        // First thread in the warp fetches the packed integer
        int laneid = lane_id();
        if (laneid == 0) {
            packed = d_input[blockIdx.x];
        }
        // Broadcast lane0's value to all lanes in the warp
        unsigned mask = 0xffffffff;
        packed = __shfl_sync(mask, packed, 0);
        
        // Each lane handles one DualTrit (if within TritsPerPack)
        if (laneid < TritsPerPack) {
            // Divide by BASE^(laneid * 2) to position the desired dual-trit
            packed = packed / pow_of_base<UInt>(laneid * 2);
            auto dir = static_cast<std::uint8_t>(packed % DualTrits::BASE);
            packed /= DualTrits::BASE;
            auto exp = static_cast<std::uint8_t>(packed % DualTrits::BASE);
            d_output[blockIdx.x * TritsPerPack + laneid] = DualTrits(exp, dir);
        }
        // Remaining warp lanes (if any) do nothing
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
// Standard unpack implementation (one thread per packed integer)
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_cuda_standard(UInt const* h_input, DualTrits* h_output, int n) {
    // Allocate device memory
    UInt* d_input;
    DualTrits* d_output;

    cudaMalloc(&d_input, n * sizeof(UInt));
    cudaMalloc(&d_output, n * TritsPerPack * sizeof(DualTrits));

    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(UInt), cudaMemcpyHostToDevice);
    
    // Launch standard kernel: one thread per packed integer
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    unpack_kernel<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * TritsPerPack * sizeof(DualTrits), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Optimized warp-cooperative host API
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_cuda(UInt const* h_input, DualTrits* h_output, int n) {
    // Allocate device memory
    UInt* d_input;
    DualTrits* d_output;

    cudaMalloc(&d_input, n * sizeof(UInt));
    cudaMalloc(&d_output, n * TritsPerPack * sizeof(DualTrits));

    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(UInt), cudaMemcpyHostToDevice);
    
    // Launch warp kernel: one block per packed integer, one warp per block
    int blockSize = 32;
    int gridSize = n;
    unpack_kernel_warp<TritsPerPack, UInt><<<gridSize, blockSize>>>(d_input, d_output, n);
    
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

template void unpack_dual_trits_batch_cuda_standard<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_cuda_standard<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_cuda_standard<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);
