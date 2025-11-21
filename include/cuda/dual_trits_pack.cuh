//
// Created by bowman on 11/9/25.
// CUDA version of dual trits packing
//

#ifndef PROJECT_FLOAT_CUDA_PACKING_H
#define PROJECT_FLOAT_CUDA_PACKING_H

#include <cuda_runtime.h>
#include <cstdint>
#include "common/DualTrits.hpp"

template <std::size_t Count, class UInt>
void pack_dual_trits_batch_cuda(DualTrits const* h_input, UInt* h_output, int n);

template <std::size_t Count, class UInt>
void unpack_dual_trits_batch_cuda(UInt const* h_input, DualTrits* h_output, int n);
#endif //PROJECT_FLOAT_CUDA_PACKING_H
