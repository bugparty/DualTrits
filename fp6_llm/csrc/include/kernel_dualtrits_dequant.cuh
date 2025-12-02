/*
 * DualTrits GPU Dequantization Kernel
 * 
 * This file provides GPU-side dequantization for DualTrits encoded weights.
 * DualTrits represents 9 values: {-3, -1, -1/3, 0, 1/3, 1, 3, +inf, -inf}
 * 
 * Encoding: storage = (exponent << 2) | direction
 *   exponent: 0 -> 3^0=1, 1 -> 3^1=3, 2 -> 3^-1=1/3
 *   direction: 0 -> 0/inf, 1 -> +1, 2 -> -1
 * 
 * Packing: 5 DualTrits -> 1 uint16_t using base-3 encoding (3^10 = 59049 < 65536)
 * Storage: 64 elements -> 16 uint16_t (13 valid packs + 3 padding for coalesced access)
 */

#ifndef KERNEL_DUALTRITS_DEQUANT_CUH
#define KERNEL_DUALTRITS_DEQUANT_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>  // for CUDART_INF_F

/*****************************************************************************
 * DualTrits LUT as Device Function
 *****************************************************************************/

// LUT for DualTrits decoding: trit_encoding (4-bit) -> float value
// Index = (exp << 2) | dir, where exp ∈ {0,1,2}, dir ∈ {0,1,2}
// Only indices 0-10 are valid (3^2 * 3 = 9 values + some invalid)
// Note: Using device function to avoid __constant__ initialization issues
__device__ __forceinline__ float get_dualtrit_lut_value(int idx) {
    switch(idx) {
        case 0:  return 0.0f;           // (exp=0, dir=0) -> 0
        case 1:  return 1.0f;           // (exp=0, dir=1) -> +1
        case 2:  return -1.0f;          // (exp=0, dir=2) -> -1
        case 3:  return 0.0f;           // invalid
        case 4:  return __int_as_float(0x7F800000);  // (exp=1, dir=0) -> +inf
        case 5:  return 3.0f;           // (exp=1, dir=1) -> +3
        case 6:  return -3.0f;          // (exp=1, dir=2) -> -3
        case 7:  return 0.0f;           // invalid
        case 8:  return __int_as_float(0xFF800000);  // (exp=2, dir=0) -> -inf
        case 9:  return 0.333333333333333f;  // (exp=2, dir=1) -> +1/3
        case 10: return -0.333333333333333f; // (exp=2, dir=2) -> -1/3
        default: return 0.0f;           // invalid
    }
}

// Macro for backward compatibility with existing code
#define DUALTRIT_LUT_GPU(idx) get_dualtrit_lut_value(idx)

/*****************************************************************************
 * Magic Number Division for Base-3 Unpacking
 *****************************************************************************/

/*
 * Fast division by 3 using magic number multiplication.
 * 
 * Mathematical basis:
 *   x / 3 = floor((x * M) >> S)
 *   where M = ceil(2^S / 3)
 * 
 * For 32-bit arithmetic with x <= 59049:
 *   M = 0xAAAAAAAB (2863311531)
 *   S = 33
 * 
 * Using __umulhi(x, M) computes (x * M) >> 32, so we need >> 1 more.
 */
__device__ __forceinline__ uint32_t div3_magic(uint32_t x) {
    return __umulhi(x, 0xAAAAAAABu) >> 1;
}

/*
 * Fast modulo 3 using magic division.
 * x % 3 = x - (x / 3) * 3
 */
__device__ __forceinline__ uint32_t mod3_magic(uint32_t x) {
    uint32_t q = div3_magic(x);
    return x - q - q - q;  // x - q * 3, avoiding multiplication
}

/*
 * Unpack 5 DualTrits from a base-3 encoded uint16_t.
 * 
 * The packed format stores: packed = Σ(dir[i] * 3^(2i) + exp[i] * 3^(2i+1))
 * for i = 0..4
 * 
 * Output: 5 trit encodings, each in format (exp << 2) | dir
 */
__device__ __forceinline__ void unpack5_dualtrits_gpu(
    uint32_t packed,
    uint32_t& trit0, uint32_t& trit1, uint32_t& trit2,
    uint32_t& trit3, uint32_t& trit4
) {
    // Trit 0
    uint32_t dir0 = mod3_magic(packed);
    packed = div3_magic(packed);
    uint32_t exp0 = mod3_magic(packed);
    packed = div3_magic(packed);
    trit0 = (exp0 << 2) | dir0;
    
    // Trit 1
    uint32_t dir1 = mod3_magic(packed);
    packed = div3_magic(packed);
    uint32_t exp1 = mod3_magic(packed);
    packed = div3_magic(packed);
    trit1 = (exp1 << 2) | dir1;
    
    // Trit 2
    uint32_t dir2 = mod3_magic(packed);
    packed = div3_magic(packed);
    uint32_t exp2 = mod3_magic(packed);
    packed = div3_magic(packed);
    trit2 = (exp2 << 2) | dir2;
    
    // Trit 3
    uint32_t dir3 = mod3_magic(packed);
    packed = div3_magic(packed);
    uint32_t exp3 = mod3_magic(packed);
    packed = div3_magic(packed);
    trit3 = (exp3 << 2) | dir3;
    
    // Trit 4 (last one, packed is now <= 8)
    uint32_t dir4 = mod3_magic(packed);
    packed = div3_magic(packed);
    uint32_t exp4 = packed;  // No need for mod on last one
    trit4 = (exp4 << 2) | dir4;
}

/*
 * Dequantize 5 DualTrits to 5 half values.
 * Applies scale factor to each value.
 */
__device__ __forceinline__ void dequant5_dualtrits_to_half(
    uint32_t packed,
    half& out0, half& out1, half& out2, half& out3, half& out4,
    half scale
) {
    uint32_t t0, t1, t2, t3, t4;
    unpack5_dualtrits_gpu(packed, t0, t1, t2, t3, t4);
    
    // LUT lookup and scale
    float s = __half2float(scale);
    out0 = __float2half_rn(get_dualtrit_lut_value(t0) * s);
    out1 = __float2half_rn(get_dualtrit_lut_value(t1) * s);
    out2 = __float2half_rn(get_dualtrit_lut_value(t2) * s);
    out3 = __float2half_rn(get_dualtrit_lut_value(t3) * s);
    out4 = __float2half_rn(get_dualtrit_lut_value(t4) * s);
}

/*
 * Dequantize 5 DualTrits to half2 pairs (for vectorized operations).
 * Returns 2 half2 values + 1 half value (5 values total).
 */
__device__ __forceinline__ void dequant5_dualtrits_to_half2(
    uint32_t packed,
    half2& out01, half2& out23, half& out4,
    half scale
) {
    uint32_t t0, t1, t2, t3, t4;
    unpack5_dualtrits_gpu(packed, t0, t1, t2, t3, t4);
    
    float s = __half2float(scale);
    float v0 = get_dualtrit_lut_value(t0) * s;
    float v1 = get_dualtrit_lut_value(t1) * s;
    float v2 = get_dualtrit_lut_value(t2) * s;
    float v3 = get_dualtrit_lut_value(t3) * s;
    float v4 = get_dualtrit_lut_value(t4) * s;
    
    out01 = __floats2half2_rn(v0, v1);
    out23 = __floats2half2_rn(v2, v3);
    out4 = __float2half_rn(v4);
}

/*****************************************************************************
 * DualTrits Memory Layout Constants
 *****************************************************************************/

// Storage layout: 64 DualTrits -> 16 uint16_t (13 data + 3 padding)
#define DUALTRITS_PER_GROUP         64
#define DUALTRITS_PACKS_PER_GROUP_VALID       13    // ceil(64 / 5) = 13 packs
#define DUALTRITS_PACKS_PER_GROUP_PADDED      16    // Padded to 32 bytes for coalesced access
#define DUALTRITS_TRITS_PER_PACK              5

// Each warp processes 64x64 = 4096 weights
// = 64 groups of 64 elements each
// = 64 * 16 = 1024 uint16_t = 2048 bytes per warp
#define SMEM_SIZE_PER_WARP_DUALTRITS    (64 * DUALTRITS_PACKS_PER_GROUP_PADDED * sizeof(uint16_t))  // 2048 bytes

// Number of uint4 (16 bytes) per warp for global memory access
#define NUM_INT4_PER_WARP_DUALTRITS     (SMEM_SIZE_PER_WARP_DUALTRITS / 16)  // 128

/*****************************************************************************
 * Dequantization Functions for GEMM Integration
 *****************************************************************************/

/*
 * Dequantize a fragment of DualTrits weights for MMA operation.
 * 
 * Each thread in a warp loads and dequantizes a portion of the weights.
 * The layout is designed to match Tensor Core MMA requirements.
 * 
 * Parameters:
 *   Frag_SPTR  - Pointer to packed DualTrits in shared memory
 *   Frag_RPTR  - Output registers (FP16 values packed as uint32_t)
 *   ScalesFrag - Scale factors for dequantization
 *   slice_id   - Which K-slice we're processing (0-3 for K=64, each slice is K=16)
 */
template<int NUM_REG_SETS>
__device__ __forceinline__ void Dequant_DualTrits_FromSharedToReg(
    uint32_t* Frag_SPTR,          // Shared memory pointer to packed DualTrits
    uint32_t  Frag_RPTR[][4],     // Output: FP16 values in registers
    uint32_t* ScalesFrag,         // Scale factors
    int slice_id                  // K-slice index (0-3)
) {
    const int lane_id = threadIdx.x % 32;
    
    // Each thread processes elements based on its lane_id
    // For m16n8k16 MMA, each thread needs specific elements
    
    // Calculate which packed values this thread should read
    // Layout: 64 rows x 16 cols (K-slice), packed as 16 uint16_t per 64 elements
    
    const uint16_t* packed_ptr = reinterpret_cast<const uint16_t*>(Frag_SPTR);
    
    #pragma unroll
    for (int reg_set = 0; reg_set < NUM_REG_SETS; reg_set++) {
        // Each reg_set corresponds to a 16x16 MMA block within the 64x64 tile
        int row_base = reg_set * 16;
        
        // Get scale for this row group
        half scale = reinterpret_cast<half*>(ScalesFrag)[row_base / 16];  // Simplified scale access
        
        // Thread lane determines which elements within the MMA fragment
        // Standard m16n8k16 fragment layout for A matrix (row-major)
        int row_in_frag = lane_id / 4;           // 0-7, repeated twice
        int col_in_frag = (lane_id % 4) * 2;     // 0, 2, 4, 6
        
        int global_row = row_base + row_in_frag;
        int global_col = slice_id * 16 + col_in_frag;
        
        // Calculate which pack contains our elements
        // Each row is packed as: group_id = col / 64, pack_id = (col % 64) / 5
        int elem_in_row = global_col;
        int group_in_row = elem_in_row / DUALTRITS_PER_GROUP;
        int elem_in_group = elem_in_row % DUALTRITS_PER_GROUP;
        int pack_in_group = elem_in_group / DUALTRITS_TRITS_PER_PACK;
        int trit_in_pack = elem_in_group % DUALTRITS_TRITS_PER_PACK;
        
        // Read the packed value
        int pack_offset = global_row * (DUALTRITS_PACKS_PER_GROUP_PADDED) + pack_in_group;
        uint32_t packed = packed_ptr[pack_offset];
        
        // Unpack and get the specific trit values we need
        uint32_t t0, t1, t2, t3, t4;
        unpack5_dualtrits_gpu(packed, t0, t1, t2, t3, t4);
        
        // Select the right trits based on position
        uint32_t trits[5] = {t0, t1, t2, t3, t4};
        
        // Get two consecutive half values and pack into uint32_t
        float s = __half2float(scale);
        uint32_t trit_val0 = trits[trit_in_pack];
        uint32_t trit_val1 = trits[(trit_in_pack + 1) % 5];
        half h0 = __float2half_rn(get_dualtrit_lut_value(trit_val0) * s);
        half h1 = __float2half_rn(get_dualtrit_lut_value(trit_val1) * s);
        
        // Pack two halfs into uint32_t for MMA
        Frag_RPTR[reg_set][0] = *reinterpret_cast<uint32_t*>(&h0);
        // Note: Full implementation needs to handle all 4 registers per set
    }
}

/*
 * Simplified dequantization for testing: dequant entire 64-element group
 */
__device__ __forceinline__ void Dequant_DualTrits_64Elements(
    const uint16_t* packed_in,    // 16 packed values (13 valid + 3 padding)
    half* fp16_out,               // 64 FP16 output values
    half scale
) {
    float s = __half2float(scale);
    int out_idx = 0;
    
    #pragma unroll
    for (int p = 0; p < DUALTRITS_PACKS_PER_GROUP_VALID; p++) {
        uint32_t packed = packed_in[p];
        
        uint32_t t0, t1, t2, t3, t4;
        unpack5_dualtrits_gpu(packed, t0, t1, t2, t3, t4);
        
        // Output 5 values (or fewer for last pack)
        if (out_idx < 64) fp16_out[out_idx++] = __float2half_rn(get_dualtrit_lut_value(t0) * s);
        if (out_idx < 64) fp16_out[out_idx++] = __float2half_rn(get_dualtrit_lut_value(t1) * s);
        if (out_idx < 64) fp16_out[out_idx++] = __float2half_rn(get_dualtrit_lut_value(t2) * s);
        if (out_idx < 64) fp16_out[out_idx++] = __float2half_rn(get_dualtrit_lut_value(t3) * s);
        if (out_idx < 64) fp16_out[out_idx++] = __float2half_rn(get_dualtrit_lut_value(t4) * s);
    }
}

#endif // KERNEL_DUALTRITS_DEQUANT_CUH
