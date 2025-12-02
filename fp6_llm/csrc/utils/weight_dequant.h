#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.h"

/*
 * DualTrits value lookup table for dequantization.
 * 
 * DualTrits encoding: storage = (exponent << 2) | direction
 *   exponent: 0 -> 3^0=1, 1 -> 3^1=3, 2 -> 3^-1=1/3
 *   direction: 0 -> 0, 1 -> +1, 2 -> -1
 * 
 * Value = direction_value * exponent_value
 * 
 * Index mapping (storage = (exp << 2) | dir):
 *   0: (0,0) -> 0      4: (1,0) -> +inf    8: (2,0) -> -inf
 *   1: (0,1) -> 1      5: (1,1) -> 3       9: (2,1) -> 1/3
 *   2: (0,2) -> -1     6: (1,2) -> -3     10: (2,2) -> -1/3
 *   3: invalid         7: invalid         11-15: invalid
 */
static const float DUALTRIT_TO_FLOAT_LUT[16] = {
    0.0f,                                       // 0: (exp=0, dir=0) = 0
    1.0f,                                       // 1: (exp=0, dir=1) = 1
    -1.0f,                                      // 2: (exp=0, dir=2) = -1
    0.0f,                                       // 3: invalid
    std::numeric_limits<float>::infinity(),     // 4: (exp=1, dir=0) = +inf
    3.0f,                                       // 5: (exp=1, dir=1) = 3
    -3.0f,                                      // 6: (exp=1, dir=2) = -3
    0.0f,                                       // 7: invalid
    -std::numeric_limits<float>::infinity(),    // 8: (exp=2, dir=0) = -inf
    1.0f / 3.0f,                                // 9: (exp=2, dir=1) = 1/3
    -1.0f / 3.0f,                               // 10: (exp=2, dir=2) = -1/3
    0.0f,                                       // 11: invalid
    0.0f,                                       // 12: invalid
    0.0f,                                       // 13: invalid
    0.0f,                                       // 14: invalid
    0.0f                                        // 15: invalid
};

/*
 * Dequantize an FPx weight matrix to FP16 format with per-row scaling.
 *
 * This function converts a packed FPx (low-bit floating point) weight matrix
 * back to half-precision (FP16) format. Each row is scaled by a corresponding
 * scale factor to restore the original magnitude.
 *
 * Template Parameters:
 *   EXPONENT - Number of exponent bits in the FPx format
 *   MANTISSA - Number of mantissa bits in the FPx format
 *
 * Parameters:
 *   A_16bit_h  - Output: FP16 matrix of size [M x K], stored in row-major order
 *   A_x_bit_h  - Input: Packed FPx matrix, size = M * K * BIT_WIDTH / 8 bytes
 *   M          - Number of rows (must be multiple of 64)
 *   K          - Number of columns (must be multiple of 64)
 *   scale      - Per-row scale factors array of size M (one scale per row)
 *
 * Dequantization Formula:
 *   FP16_value = FPx_to_float(packed_value) * bias_correction * scale[row]
 *
 * Processing:
 *   Each iteration processes BIT_WIDTH bytes of input to produce 8 FP16 values.
 *   The bias correction factor accounts for the exponent bias difference
 *   between FPx and FP16 formats.
 */
template<int EXPONENT, int MANTISSA>
void DeQuantMatrix_FPx_To_FP16(half* A_16bit_h, unsigned char* A_x_bit_h, size_t M, size_t K, half* scale) {
    //
    assert(M%64==0);                 // Currently, M must be a multiple of 64.
    assert(K%64==0);                 // Currently, K must be a multiple of 64.
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    assert(BIT_WIDTH<=8);
    size_t TotalSizeInByte = M * K * BIT_WIDTH / 8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte/BIT_WIDTH; i++) {    // Processing BIT_WIDTH Bytes for each Loop, generating 8 FP16.
        unsigned char Bytes[BIT_WIDTH];
        for(int x=0; x<BIT_WIDTH; x++)  Bytes[x] = A_x_bit_h[i*BIT_WIDTH+x];
        unsigned char OUT[8];
        for(int x=0; x<8; x++) {                        // Prepare Initial memory layout for Dequant
            int ByteOffset  = BIT_WIDTH * x / 8;
            int BitOffset   = BIT_WIDTH * x % 8;
            OUT[x] = Extract_X_Bits_To_A_Byte<EXPONENT, MANTISSA>(Bytes, ByteOffset, BitOffset);
        }
        // Dequant
        constexpr int MASK1 = 0x80000000;
        constexpr int MASK2 = MASK1 >> EXPONENT + MANTISSA;
        constexpr int MASK  = MASK2 & 0x7fffffff;
        constexpr int RIGHT_SHIFT = 5 - EXPONENT;
        constexpr int BIAS_OFFSET = (int(1) << (5-1)) - (int(1) << (EXPONENT-1));
        constexpr int BIAS        = int(1) << BIAS_OFFSET;
        for(int x=0; x<8; x++) {
            unsigned int OUT_fp16;        // Storing fp16 in the high 16 bits.
            OUT_fp16 = int(OUT[x]) << 24;
            OUT_fp16 = (OUT_fp16 & 0x80000000) | ( (OUT_fp16 & MASK) >> RIGHT_SHIFT );
            OUT_fp16 = OUT_fp16 >> 16;
            //
            half* OUT_FP16_PTR = reinterpret_cast<half*>(&OUT_fp16);
            OutPTR[x] = __float2half_rn ( __half2float(*OUT_FP16_PTR) * (1.0f*BIAS) * __half2float(scale[(8*i)/K]) );
        }   
        //
        OutPTR +=8;
    }
}


/*
 * Dequantize FP6 (E3M2) weight matrix to FP16 format.
 *
 * Convenience wrapper for DeQuantMatrix_FPx_To_FP16 with FP6 format
 * (3 exponent bits + 2 mantissa bits + 1 sign bit = 6 bits total).
 *
 * Parameters:
 *   A_16bit_h - Output: FP16 matrix of size [M x K]
 *   A_6bit_h  - Input: Packed FP6 matrix
 *   M         - Number of rows (must be multiple of 64)
 *   K         - Number of columns (must be multiple of 64)
 *   scale     - Per-row scale factors array of size M
 */
void DeQuantMatrix_FP6_To_FP16(half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale) {
    DeQuantMatrix_FPx_To_FP16<3, 2>(A_16bit_h, A_6bit_h, M, K, scale);
}

/*
 * Dequantize FPx weight matrix to FP16 with runtime-specified format.
 *
 * This function provides runtime selection of the FPx format for dequantization,
 * dispatching to the appropriate template instantiation.
 *
 * Parameters:
 *   EXPONENT  - Number of exponent bits (2 or 3 supported)
 *   MANTISSA  - Number of mantissa bits (2 supported)
 *   A_16bit_h - Output: FP16 matrix of size [M x K]
 *   A_6bit_h  - Input: Packed FPx matrix
 *   M         - Number of rows (must be multiple of 64)
 *   K         - Number of columns (must be multiple of 64)
 *   scale     - Per-row scale factors array of size M
 *
 * Supported Formats:
 *   E2M2 (FP5): EXPONENT=2, MANTISSA=2
 *   E3M2 (FP6): EXPONENT=3, MANTISSA=2
 */
void dequant_matrix_fp_eXmY_to_fp16(const int EXPONENT, const int MANTISSA, half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale){
    if(EXPONENT==2 && MANTISSA==2)
        return DeQuantMatrix_FPx_To_FP16<2, 2>(A_16bit_h, A_6bit_h, M, K, scale);
    if(EXPONENT==3 && MANTISSA==2)
        return DeQuantMatrix_FPx_To_FP16<3, 2>(A_16bit_h, A_6bit_h, M, K, scale);
    printf("DeQuantMatrix Error: Unsupported EXPONENT=%d, MANTISSA=%d!\n", EXPONENT, MANTISSA);
    exit(-1);
}

/*
 * Dequantize a DualTrits weight matrix to FP16 format with per-row scaling.
 *
 * This function converts a packed DualTrits weight matrix back to half-precision
 * (FP16) format. DualTrits uses base-3 packing where each DualTrit represents
 * one of 9 values: {-3, -1, -1/3, 0, 1/3, 1, 3, +inf, -inf}.
 *
 * Packing scheme: 5 DualTrits -> 1 uint16_t (base-3 encoding, 3^10 = 59049 < 65536)
 * Storage layout: 13 packs (65 DualTrits) store 64 valid elements + 1 reserved
 *
 * Parameters:
 *   A_16bit_h  - Output: FP16 matrix of size [M x K], stored in row-major order
 *   A_packed_h - Input: Packed DualTrits as uint16_t array, size = (M * K / 64) * 13
 *   M          - Number of rows (must be multiple of 64)
 *   K          - Number of columns (must be multiple of 64)
 *   scale      - Per-row scale factors array of size M (one scale per row)
 *
 * Dequantization Formula:
 *   FP16_value = DualTrit_to_float(packed_value) * scale[row]
 */
void DeQuantMatrix_DualTrit5_To_FP16(half* A_16bit_h, std::uint16_t* A_packed_h, size_t M, size_t K, half* scale) {
    assert(M % 64 == 0);                // M must be a multiple of 64
    assert(K % 64 == 0);                // K must be a multiple of 64
    
    constexpr size_t TRITS_PER_PACK = 5;
    constexpr size_t PACKS_PER_64 = 13;     // 13 packs hold 65 DualTrits, use first 64
    constexpr size_t VALID_PER_GROUP = 64;
    
    const size_t total_elements = M * K;
    const size_t num_groups = total_elements / VALID_PER_GROUP;
    
    half* out_ptr = A_16bit_h;
    std::uint16_t* pack_ptr = A_packed_h;
    
    for (size_t g = 0; g < num_groups; ++g) {
        // Temporary buffer for unpacked values (65 slots, use first 64)
        float unpacked[PACKS_PER_64 * TRITS_PER_PACK];  // 65 floats
        
        // Unpack 13 uint16_t packs into 65 float values
        for (size_t p = 0; p < PACKS_PER_64; ++p) {
            std::uint16_t packed = pack_ptr[p];
            
            // Extract 5 DualTrits from this pack using base-3 division
            for (size_t t = 0; t < TRITS_PER_PACK; ++t) {
                std::uint8_t dir = packed % 3;
                packed /= 3;
                std::uint8_t exp = packed % 3;
                packed /= 3;
                
                // Compute storage index: (exp << 2) | dir
                std::uint8_t storage = (exp << 2) | dir;
                unpacked[p * TRITS_PER_PACK + t] = DUALTRIT_TO_FLOAT_LUT[storage];
            }
        }
        
        // Calculate base index for this group
        size_t base_idx = g * VALID_PER_GROUP;
        
        // Write first 64 values to output with per-row scaling
        for (size_t i = 0; i < VALID_PER_GROUP; ++i) {
            size_t global_idx = base_idx + i;
            size_t row = global_idx / K;
            float value = unpacked[i];
            float scaled = value * __half2float(scale[row]);
            out_ptr[i] = __float2half_rn(scaled);
        }
        
        out_ptr += VALID_PER_GROUP;
        pack_ptr += PACKS_PER_64;
    }
}

/*
 * Dequantize a packed DualTrits matrix with GPU-optimized padding to FP16.
 *
 * This version handles the padded format: 16 uint16_t per 64 elements
 * (13 valid packs + 3 zero padding).
 *
 * Parameters:
 *   A_16bit_h  - Output: FP16 matrix of size [M x K], stored in row-major order
 *   A_packed_h - Input: Padded packed DualTrits, size = (M * K / 64) * 16 uint16_t
 *   M          - Number of rows (must be multiple of 64)
 *   K          - Number of columns (must be multiple of 64)
 *   scale      - Per-row scale factors array of size M
 */
void DeQuantMatrix_DualTrit5_Padded_To_FP16(half* A_16bit_h, std::uint16_t* A_packed_h, size_t M, size_t K, half* scale) {
    assert(M % 64 == 0);
    assert(K % 64 == 0);
    
    constexpr size_t TRITS_PER_PACK = 5;
    constexpr size_t PACKS_PER_64_VALID = 13;   // 13 packs hold 65 DualTrits
    constexpr size_t PACKS_PER_64_PADDED = 16;  // Padded to 16 for alignment
    constexpr size_t VALID_PER_GROUP = 64;
    
    const size_t total_elements = M * K;
    const size_t num_groups = total_elements / VALID_PER_GROUP;
    
    half* out_ptr = A_16bit_h;
    std::uint16_t* pack_ptr = A_packed_h;
    
    for (size_t g = 0; g < num_groups; ++g) {
        // Temporary buffer for unpacked values
        float unpacked[PACKS_PER_64_VALID * TRITS_PER_PACK];  // 65 floats
        
        // Unpack first 13 uint16_t packs (ignore 3 padding packs)
        for (size_t p = 0; p < PACKS_PER_64_VALID; ++p) {
            std::uint16_t packed = pack_ptr[p];
            
            // Extract 5 DualTrits from this pack using base-3 division
            for (size_t t = 0; t < TRITS_PER_PACK; ++t) {
                std::uint8_t dir = packed % 3;
                packed /= 3;
                std::uint8_t exp = packed % 3;
                packed /= 3;
                
                // Compute storage index: (exp << 2) | dir
                std::uint8_t storage = (exp << 2) | dir;
                unpacked[p * TRITS_PER_PACK + t] = DUALTRIT_TO_FLOAT_LUT[storage];
            }
        }
        
        // Calculate base index for this group
        size_t base_idx = g * VALID_PER_GROUP;
        
        // Write first 64 values to output with per-row scaling
        for (size_t i = 0; i < VALID_PER_GROUP; ++i) {
            size_t global_idx = base_idx + i;
            size_t row = global_idx / K;
            float value = unpacked[i];
            float scaled = value * __half2float(scale[row]);
            out_ptr[i] = __float2half_rn(scaled);
        }
        
        out_ptr += VALID_PER_GROUP;
        pack_ptr += PACKS_PER_64_PADDED;  // Skip padding too
    }
}

/*
 * Dequantize a DualTrits weight matrix to FP16 format (template version).
 *
 * This is a generic template wrapper. Currently only supports TritsPerPack=5
 * with UInt=uint16_t, which delegates to DeQuantMatrix_DualTrit5_To_FP16.
 *
 * Template Parameters:
 *   TritsPerPack - Number of DualTrits packed together (currently only 5 supported)
 *   UInt         - Unsigned integer type for packing (currently only uint16_t supported)
 *
 * Parameters:
 *   A_16bit_h  - Output: FP16 matrix of size [M x K], stored in row-major order
 *   A_x_bit_h  - Input: Packed DualTrits matrix as byte array
 *   M          - Number of rows (must be multiple of 64)
 *   K          - Number of columns (must be multiple of 64)
 *   scale      - Per-row scale factors array of size M (one scale per row)
 */
template<std::size_t TritsPerPack, class UInt>
void DeQuantMatrix_DualTrit_To_FP16(half* A_16bit_h, unsigned char* A_x_bit_h, size_t M, size_t K, half* scale) {
    static_assert(TritsPerPack == 5 && std::is_same<UInt, std::uint16_t>::value,
                  "Currently only TritsPerPack=5 with uint16_t is supported");
    
    assert(M % 64 == 0);                // M must be a multiple of 64
    assert(K % 64 == 0);                // K must be a multiple of 64
    
    // Delegate to the optimized uint16_t version
    DeQuantMatrix_DualTrit5_To_FP16(A_16bit_h, reinterpret_cast<std::uint16_t*>(A_x_bit_h), M, K, scale);
}
