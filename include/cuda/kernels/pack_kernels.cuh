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

#include "cuda/dual_trits_pack.cuh"
#include "cuda/kernels/pack_kernels.cu"

// Explicit template instantiations for common types
template __global__ void pack_kernel<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template __global__ void pack_kernel<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template __global__ void pack_kernel<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template __global__ void unpack_kernel<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template __global__ void unpack_kernel<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template __global__ void unpack_kernel<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);

// Explicit template instantiations for host API
template void pack_dual_trits_batch_cuda<5, std::uint16_t>(DualTrits const*, std::uint16_t*, int);
template void pack_dual_trits_batch_cuda<10, std::uint32_t>(DualTrits const*, std::uint32_t*, int);
template void pack_dual_trits_batch_cuda<20, std::uint64_t>(DualTrits const*, std::uint64_t*, int);

template void unpack_dual_trits_batch_cuda<5, std::uint16_t>(std::uint16_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_cuda<10, std::uint32_t>(std::uint32_t const*, DualTrits*, int);
template void unpack_dual_trits_batch_cuda<20, std::uint64_t>(std::uint64_t const*, DualTrits*, int);

#endif //PROJECT_FLOAT_CUDA_KERNELS_H
