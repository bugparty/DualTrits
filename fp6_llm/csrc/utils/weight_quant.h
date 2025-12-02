// Author: Zhen Zheng
// To be used in the future as a tool to generating the FP6 matrix from the FP16 matrix.

#include<iostream>
#include<cstdint>
#include<cmath>

/*
 * DualTrits quantization constants and lookup tables.
 * 
 * DualTrits represents 9 values: {-3, -1, -1/3, 0, 1/3, 1, 3, +inf, -inf}
 * Encoding: storage = (exponent << 2) | direction
 *   exponent: 0 -> 3^0=1, 1 -> 3^1=3, 2 -> 3^-1=1/3
 *   direction: 0 -> 0, 1 -> +1, 2 -> -1
 * 
 * Quantization thresholds (midpoints between adjacent values):
 *   T0 = (0 + 1/3) / 2 = 1/6
 *   T1 = (1/3 + 1) / 2 = 2/3
 *   T2 = (1 + 3) / 2 = 2
 */

// Epsilon for floating-point comparison
constexpr float DUALTRIT_EPSILON = 1e-6f;

// Quantization thresholds
constexpr float DUALTRIT_T0 = 1.0f / 6.0f;   // 0 vs 1/3 boundary
constexpr float DUALTRIT_T1 = 2.0f / 3.0f;   // 1/3 vs 1 boundary
constexpr float DUALTRIT_T2 = 2.0f;          // 1 vs 3 boundary

// Quantization point values (absolute)
constexpr float DUALTRIT_Q0 = 0.0f;
constexpr float DUALTRIT_Q1 = 1.0f / 3.0f;
constexpr float DUALTRIT_Q2 = 1.0f;
constexpr float DUALTRIT_Q3 = 3.0f;

/*
 * Quantize a single float value to DualTrits encoding.
 *
 * Uses round-to-nearest with tie-to-even strategy.
 * 
 * Parameters:
 *   value - Input float value (should be normalized by scale)
 *
 * Returns:
 *   4-bit encoding: (exponent << 2) | direction
 *   
 * Encoding table:
 *   0  -> (exp=0, dir=0) -> 0
 *   1  -> (exp=0, dir=1) -> +1
 *   2  -> (exp=0, dir=2) -> -1
 *   5  -> (exp=1, dir=1) -> +3
 *   6  -> (exp=1, dir=2) -> -3
 *   9  -> (exp=2, dir=1) -> +1/3
 *   10 -> (exp=2, dir=2) -> -1/3
 *   4  -> (exp=1, dir=0) -> +inf
 *   8  -> (exp=2, dir=0) -> -inf
 */
inline std::uint8_t cast_fp16_to_dualtrit(float value) {
    // Handle special cases
    if (std::isinf(value)) {
        return (value > 0) ? 4 : 8;  // +inf: (exp=1,dir=0), -inf: (exp=2,dir=0)
    }
    if (std::isnan(value)) {
        return 0;  // NaN -> 0
    }
    
    // Extract sign and work with absolute value
    bool negative = value < 0;
    float abs_val = std::fabs(value);
    
    // Determine quantization point using thresholds
    // Round to nearest, tie to even (even index = 0, 2)
    std::uint8_t exp, dir;
    
    if (abs_val < DUALTRIT_T0 - DUALTRIT_EPSILON) {
        // Clearly closer to 0
        exp = 0; dir = 0;  // -> 0
    } 
    else if (std::fabs(abs_val - DUALTRIT_T0) < DUALTRIT_EPSILON) {
        // Tie at T0 = 1/6, between 0 (index 0, even) and 1/3 (index 1, odd)
        // Tie to even -> choose 0
        exp = 0; dir = 0;  // -> 0
    }
    else if (abs_val < DUALTRIT_T1 - DUALTRIT_EPSILON) {
        // Clearly closer to 1/3
        exp = 2; dir = 1;  // -> 1/3
    }
    else if (std::fabs(abs_val - DUALTRIT_T1) < DUALTRIT_EPSILON) {
        // Tie at T1 = 2/3, between 1/3 (index 1, odd) and 1 (index 2, even)
        // Tie to even -> choose 1
        exp = 0; dir = 1;  // -> 1
    }
    else if (abs_val < DUALTRIT_T2 - DUALTRIT_EPSILON) {
        // Clearly closer to 1
        exp = 0; dir = 1;  // -> 1
    }
    else if (std::fabs(abs_val - DUALTRIT_T2) < DUALTRIT_EPSILON) {
        // Tie at T2 = 2, between 1 (index 2, even) and 3 (index 3, odd)
        // Tie to even -> choose 1
        exp = 0; dir = 1;  // -> 1
    }
    else {
        // Clearly closer to 3
        exp = 1; dir = 1;  // -> 3
    }
    
    // Apply sign (dir: 0->0, 1->positive, 2->negative)
    if (negative && dir == 1) {
        dir = 2;
    }
    
    return (exp << 2) | dir;
}

/*
 * Pack 5 DualTrits encodings into a single uint16_t using base-3 encoding.
 *
 * Each DualTrit has (exp, dir) where both are in {0, 1, 2}.
 * Packing order: dir first, then exp, for each trit.
 * Formula: packed = Σ(dir[i] * 3^(2i) + exp[i] * 3^(2i+1))
 *
 * Parameters:
 *   trits - Array of 5 DualTrits encodings (each is (exp<<2)|dir format)
 *
 * Returns:
 *   Packed uint16_t value (max value = 3^10 - 1 = 59048 < 65535)
 */
inline std::uint16_t pack5_dualtrit(std::uint8_t trits[5]) {
    std::uint16_t packed = 0;
    std::uint16_t multiplier = 1;
    
    for (int i = 0; i < 5; ++i) {
        std::uint8_t dir = trits[i] & 0x03;
        std::uint8_t exp = (trits[i] >> 2) & 0x03;
        
        packed += dir * multiplier;
        multiplier *= 3;
        packed += exp * multiplier;
        multiplier *= 3;
    }
    return packed;
}

/*
 * Convert an FP16 weight matrix to packed DualTrits format with per-row scaling.
 *
 * This function quantizes FP16 values to DualTrits using round-to-nearest-tie-to-even,
 * then packs them using base-3 encoding (5 DualTrits -> 1 uint16_t).
 *
 * Storage layout: 13 packs (65 DualTrits) store 64 valid elements + 1 reserved (set to 0)
 *
 * Parameters:
 *   weight_16bit   - Input: FP16 weight matrix of size [M x K], row-major order
 *   weight_packed  - Output: Packed DualTrits as uint16_t array, size = (M * K / 64) * 13
 *   M              - Number of rows (must be multiple of 64)
 *   K              - Number of columns (must be multiple of 64)
 *   scale          - Per-row scale factors array of size M (values are divided by scale before quantizing)
 *
 * Memory Layout:
 *   Input:  M * K * 2 bytes (FP16)
 *   Output: (M * K / 64) * 13 * 2 bytes (packed uint16_t)
 */
void weight_prepacking_fp16_to_dual_trits(half* weight_16bit,
                                          std::uint16_t* weight_packed,
                                          size_t M,
                                          size_t K,
                                          half* scale)
{
    assert(M % 64 == 0);
    assert(K % 64 == 0);
    
    constexpr size_t TRITS_PER_PACK = 5;
    constexpr size_t PACKS_PER_64 = 13;     // 13 packs hold 65 DualTrits, use first 64
    constexpr size_t VALID_PER_GROUP = 64;
    
    const size_t total_elements = M * K;
    const size_t num_groups = total_elements / VALID_PER_GROUP;
    
    std::uint16_t* pack_ptr = weight_packed;
    
    for (size_t g = 0; g < num_groups; ++g) {
        // Temporary buffer for quantized values (65 slots, use first 64)
        std::uint8_t quantized[PACKS_PER_64 * TRITS_PER_PACK];  // 65 values
        
        // Calculate base index for this group
        size_t base_idx = g * VALID_PER_GROUP;
        
        // Quantize 64 FP16 values to DualTrits
        for (size_t i = 0; i < VALID_PER_GROUP; ++i) {
            size_t global_idx = base_idx + i;
            size_t row = global_idx / K;
            
            // Get FP16 value and normalize by scale
            float value = __half2float(weight_16bit[global_idx]);
            float scale_val = __half2float(scale[row]);
            float normalized = (scale_val != 0.0f) ? (value / scale_val) : 0.0f;
            
            // Quantize to DualTrits
            quantized[i] = cast_fp16_to_dualtrit(normalized);
        }
        
        // Set reserved slot (65th) to zero
        quantized[64] = 0;
        
        // Pack into 13 uint16_t values
        for (size_t p = 0; p < PACKS_PER_64; ++p) {
            std::uint8_t trits[5];
            for (int t = 0; t < 5; ++t) {
                trits[t] = quantized[p * TRITS_PER_PACK + t];
            }
            pack_ptr[p] = pack5_dualtrit(trits);
        }
        
        pack_ptr += PACKS_PER_64;
    }
}

/*
 * Convert an FP16 weight matrix to GPU-optimized packed DualTrits format.
 *
 * This version adds padding for coalesced memory access:
 * - 64 elements -> 16 uint16_t (13 valid packs + 3 zero padding)
 * - Enables 32 threads to load 32 * 2B = 64B in one coalesced transaction
 *
 * Storage layout per warp (64x64 = 4096 elements):
 *   64 groups × 16 uint16_t = 1024 uint16_t = 2048 bytes
 *
 * Parameters:
 *   weight_16bit   - Input: FP16 weight matrix of size [M x K], row-major order
 *   weight_packed  - Output: Padded packed DualTrits, size = (M * K / 64) * 16 uint16_t
 *   M              - Number of rows (must be multiple of 64)
 *   K              - Number of columns (must be multiple of 64)
 *   scale          - Per-row scale factors array of size M
 *
 * Memory Layout:
 *   Input:  M * K * 2 bytes (FP16)
 *   Output: (M * K / 64) * 16 * 2 bytes = M * K / 2 bytes
 */
void weight_prepacking_fp16_to_dual_trits_gpu(half* weight_16bit,
                                               std::uint16_t* weight_packed,
                                               size_t M,
                                               size_t K,
                                               half* scale)
{
    assert(M % 64 == 0);
    assert(K % 64 == 0);
    
    constexpr size_t TRITS_PER_PACK = 5;
    constexpr size_t PACKS_PER_64_VALID = 13;   // 13 packs hold 65 DualTrits
    constexpr size_t PACKS_PER_64_PADDED = 16;  // Padded to 32 bytes
    constexpr size_t VALID_PER_GROUP = 64;
    
    const size_t total_elements = M * K;
    const size_t num_groups = total_elements / VALID_PER_GROUP;
    
    std::uint16_t* pack_ptr = weight_packed;
    
    for (size_t g = 0; g < num_groups; ++g) {
        // Temporary buffer for quantized values (65 slots, use first 64)
        std::uint8_t quantized[PACKS_PER_64_VALID * TRITS_PER_PACK];  // 65 values
        
        // Calculate base index for this group
        size_t base_idx = g * VALID_PER_GROUP;
        
        // Quantize 64 FP16 values to DualTrits
        for (size_t i = 0; i < VALID_PER_GROUP; ++i) {
            size_t global_idx = base_idx + i;
            size_t row = global_idx / K;
            
            // Get FP16 value and normalize by scale
            float value = __half2float(weight_16bit[global_idx]);
            float scale_val = __half2float(scale[row]);
            float normalized = (scale_val != 0.0f) ? (value / scale_val) : 0.0f;
            
            // Quantize to DualTrits
            quantized[i] = cast_fp16_to_dualtrit(normalized);
        }
        
        // Set reserved slot (65th) to zero
        quantized[64] = 0;
        
        // Pack into 13 uint16_t values (valid packs)
        for (size_t p = 0; p < PACKS_PER_64_VALID; ++p) {
            std::uint8_t trits[5];
            for (int t = 0; t < 5; ++t) {
                trits[t] = quantized[p * TRITS_PER_PACK + t];
            }
            pack_ptr[p] = pack5_dualtrit(trits);
        }
        
        // Add padding zeros (3 uint16_t)
        for (size_t p = PACKS_PER_64_VALID; p < PACKS_PER_64_PADDED; ++p) {
            pack_ptr[p] = 0;
        }
        
        pack_ptr += PACKS_PER_64_PADDED;
    }
}

/*
 * Interleaved prepacking for optimal warp coalesced access pattern.
 * 
 * This version reorders the packed data so that consecutive threads
 * access consecutive memory addresses during kernel execution.
 *
 * Layout: For a 64x64 tile processed by one warp:
 *   - 64 rows × 64 cols = 4096 elements
 *   - Organized as 64 groups of 64 elements (one group per row within K=64)
 *   - Each group: 16 uint16_t (13 valid + 3 padding)
 *   - Total: 64 × 16 = 1024 uint16_t = 2048 bytes per warp tile
 *
 * Interleaving: Threads 0-31 load consecutive uint16_t values
 *   pack[0..31]   <- first 32 threads load first word each
 *   pack[32..63]  <- first 32 threads load second word each
 *   ...
 */
void weight_prepacking_fp16_to_dual_trits_gpu_interleaved(
    half* weight_16bit,
    std::uint16_t* weight_packed,
    size_t M,
    size_t K,
    half* scale)
{
    assert(M % 64 == 0);
    assert(K % 64 == 0);
    
    constexpr size_t TRITS_PER_PACK_LOCAL = 5;
    constexpr size_t PACKS_PER_64_VALID = 13;
    constexpr size_t PACKS_PER_64_PADDED = 16;
    constexpr size_t TILE_M = 64;
    constexpr size_t TILE_K = 64;
    constexpr size_t ELEMENTS_PER_TILE = TILE_M * TILE_K;  // 4096
    constexpr size_t PACKS_PER_TILE = (ELEMENTS_PER_TILE / 64) * PACKS_PER_64_PADDED;  // 64 * 16 = 1024
    
    const size_t num_tiles_M = M / TILE_M;
    const size_t num_tiles_K = K / TILE_K;
    
    // Process each warp tile
    for (size_t wm = 0; wm < num_tiles_M; ++wm) {
        for (size_t wk = 0; wk < num_tiles_K; ++wk) {
            // Output pointer for this warp tile
            size_t tile_idx = wm * num_tiles_K + wk;
            std::uint16_t* tile_pack_ptr = weight_packed + tile_idx * PACKS_PER_TILE;
            
            // Temporary buffer for this warp's packed data (before interleaving)
            std::uint16_t temp_packs[PACKS_PER_TILE];
            
            // Pack each row within the warp tile
            for (size_t row_in_tile = 0; row_in_tile < TILE_M; ++row_in_tile) {
                size_t global_row = wm * TILE_M + row_in_tile;
                size_t global_col_start = wk * TILE_K;
                
                // Get scale for this row
                float scale_val = __half2float(scale[global_row]);
                
                // Quantize 64 elements in this row segment
                std::uint8_t quantized[65];
                for (size_t c = 0; c < TILE_K; ++c) {
                    size_t global_col = global_col_start + c;
                    float value = __half2float(weight_16bit[global_row * K + global_col]);
                    float normalized = (scale_val != 0.0f) ? (value / scale_val) : 0.0f;
                    quantized[c] = cast_fp16_to_dualtrit(normalized);
                }
                quantized[64] = 0;  // Reserved slot
                
                // Pack into 13 valid + 3 padding
                size_t pack_base = row_in_tile * PACKS_PER_64_PADDED;
                for (size_t p = 0; p < PACKS_PER_64_VALID; ++p) {
                    std::uint8_t trits[5];
                    for (int t = 0; t < 5; ++t) {
                        trits[t] = quantized[p * TRITS_PER_PACK_LOCAL + t];
                    }
                    temp_packs[pack_base + p] = pack5_dualtrit(trits);
                }
                for (size_t p = PACKS_PER_64_VALID; p < PACKS_PER_64_PADDED; ++p) {
                    temp_packs[pack_base + p] = 0;
                }
            }
            
            // Interleave for coalesced access
            // Original: row0[0..15], row1[0..15], ..., row63[0..15]
            // Interleaved: [row0[0], row1[0], ..., row31[0]], [row32[0], row33[0], ..., row63[0]], ...
            // This way, 32 consecutive threads load 32 consecutive uint16_t
            for (size_t pack_col = 0; pack_col < PACKS_PER_64_PADDED; ++pack_col) {
                for (size_t row = 0; row < TILE_M; ++row) {
                    size_t src_idx = row * PACKS_PER_64_PADDED + pack_col;
                    size_t dst_idx = pack_col * TILE_M + row;
                    tile_pack_ptr[dst_idx] = temp_packs[src_idx];
                }
            }
        }
    }
}

/*
 * Convert and pack 4 FP16 values into 3 bytes of continuous FP6 storage.
 *
 * This function takes 4 half-precision (FP16) floating point values and converts
 * them to FP6 format (1 sign bit + 3 exponent bits + 2 mantissa bits = 6 bits each).
 * The 4 FP6 values (24 bits total) are then packed into 3 bytes.
 *
 * FP6 Format: Sign(1) + Exponent(3) + Mantissa(2)
 *   - Exponent bias: 3 (2^(3-1) - 1)
 *   - Representable range: [0.0625, 28] (non-zero absolute values)
 *   - Handles subnormal FP6 values
 *
 * Parameters:
 *   FP16x4 - Input: Pointer to 4 consecutive FP16 values (8 bytes)
 *   FP6x4  - Output: Pointer to 3 bytes that will store 4 packed FP6 values
 *
 * Throws:
 *   std::invalid_argument if any input value is out of FP6 representable range
 *
 * Packing Layout:
 *   FP6x4[0] = FP6[0](6bits) << 2 | FP6[1](high 2bits)
 *   FP6x4[1] = FP6[1](low 4bits) << 4 | FP6[2](high 4bits)
 *   FP6x4[2] = FP6[2](low 2bits) << 6 | FP6[3](6bits)
 */
void cast_fp16_fp6(uint16_t* FP16x4, uint8_t* FP6x4)
{
    // Constants for FP6
    constexpr int exponent_nbits_fp6 = 3;
    constexpr int mantissa_nbits_fp6 = 2;
    constexpr int exp_bias_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
    // Constants for FP16
    constexpr int exponent_nbits_fp16 = 5;
    constexpr int mantissa_nbits_fp16 = 10;
    constexpr int exp_bias_fp16 = (1 << (exponent_nbits_fp16 - 1)) - 1;

    int fp6_temp[4];

    float absmin_nonzero_fp6 = 0.0625;
    // Note that we regard the exponent of '111' as a regular value rather than NaN or inf. This is
    // the same with that in qtorch.
    float absmax_fp6 = 28;

    for (int i = 0; i < 4; ++i) {
        uint16_t source = FP16x4[i];
        float fp6_value_abs = std::abs(__half2float(*((half*)(&source))));
        if ((fp6_value_abs != 0 && fp6_value_abs < absmin_nonzero_fp6) ||
            fp6_value_abs > absmax_fp6) {
            // TODO(zhen): a better way may be rounding it to the nearest FP6 value.
            throw std::invalid_argument("Input value out of range for FP6.");
        }

        // It is not safe to do shift operation on uint16_t. So we promote it to int.
        int source_promote = int(source);

        int sign_bit = (source_promote >> 15);
        // Extracting exponent represented in FP16. The sign mask 0x7FFF is '0111 1111 1111 1111'
        int exp_bit = (source_promote & 0x7FFF) >> mantissa_nbits_fp16;
        // Extracting mantissa represented in FP16
        int mant_bit = source_promote & ((1 << mantissa_nbits_fp16) - 1);

        int new_exp_bit;
        int new_mant_bit;

        if (exp_bit == 0) {
            // Subnormal FP16 number. Too small for FP6.
            new_exp_bit = 0;
            new_mant_bit = 0;
        } else {
            new_mant_bit = mant_bit >> (mantissa_nbits_fp16 - mantissa_nbits_fp6);
            new_exp_bit = exp_bit - exp_bias_fp16 + exp_bias_fp6;

            // Deal with subnormal FP6 values.
            int target_exp_val = exp_bit - exp_bias_fp16;
            int min_fp6_exp_val = -exp_bias_fp6 + 1;
            bool subnormal_fp6 = target_exp_val < min_fp6_exp_val;
            if (subnormal_fp6) {
                // TODO(zhen): add the rounding logic.
                new_exp_bit = 0;
                // The implicit 1 in the mantissa of FP16 is not present in subnormal FP6. Thus we
                // need to add it
                new_mant_bit = (new_mant_bit | (1 << mantissa_nbits_fp6)) >>
                               (min_fp6_exp_val - target_exp_val);
            }
        }

        fp6_temp[i] = (sign_bit << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                      (new_exp_bit << mantissa_nbits_fp6) | new_mant_bit;
    }
    // Pack the values
    FP6x4[0] = fp6_temp[0] << 2 | (fp6_temp[1] >> 4);
    FP6x4[1] = (fp6_temp[1] & 0x0F) << 4 | (fp6_temp[2] >> 2);
    FP6x4[2] = (fp6_temp[2] & 0x03) << 6 | fp6_temp[3];
}

/*
 * Convert an entire FP16 weight matrix to continuously packed FP6 format.
 *
 * This function iterates through the weight matrix row by row, converting
 * every 4 FP16 values into 3 bytes of packed FP6 storage using cast_fp16_fp6().
 *
 * Parameters:
 *   weight_16bit      - Input: FP16 weight matrix of size [M x K], stored in row-major order
 *   weight_6bit_packed - Output: Packed FP6 weight array of size [M x K x 6 / 8] bytes
 *   M                 - Number of rows in the weight matrix
 *   K                 - Number of columns in the weight matrix (must satisfy K * 6 % 8 == 0)
 *
 * Memory Layout:
 *   Input:  M rows, each with K FP16 values (2 bytes each) = M * K * 2 bytes
 *   Output: M rows, each with K FP6 values packed = M * K * 6 / 8 bytes
 *
 * Constraint:
 *   (K * 6) must be divisible by 8 to ensure byte-aligned packing
 */
void weight_prepacking_fp16_to_fp6(uint16_t* weight_16bit,
                                   uint8_t* weight_6bit_packed,
                                   size_t M,
                                   size_t K)
{
    // Every four 16-bit elements are packed into three 6-bit values (4*6bit == 3*8bit).
    if (K * 6 % 8 != 0) { throw std::invalid_argument("(K * 6 % 8) should be 0"); }
    size_t K_fp6_packed = K * 6 / 8;
    // #pragma omp parallel for
    for (auto m = 0; m < M; m++) {
        uint8_t* ptr_6bit = weight_6bit_packed + m * K_fp6_packed;
        uint16_t* ptr_16bit = weight_16bit + m * K;
        for (auto k = 0; k < K; k += 4) {
            cast_fp16_fp6(ptr_16bit, ptr_6bit);
            ptr_16bit += 4;
            ptr_6bit += 3;
        }
    }
}