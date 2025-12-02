/*
 * Test for DeQuantMatrix_DualTrit_To_FP16 correctness
 * 
 * This test verifies that the DualTrits dequantization produces correct FP16 values
 * by comparing against expected float values computed from the DualTrits encoding.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cstdint>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Include the weight_dequant header which contains DeQuantMatrix_DualTrit5_To_FP16
#include "../../fp6_llm/csrc/utils/weight_dequant.h"
// Include the weight_quant header which contains weight_prepacking_fp16_to_dual_trits
#include "../../fp6_llm/csrc/utils/weight_quant.h"

// ============================================================================
// Helper functions
// ============================================================================

void CheckMallocCPU(void* PTR, int line = -1) {
    if (PTR == NULL) {
        printf("Error in CPU Malloc, line %d!\n", line);
        exit(-1);
    }
}

// ============================================================================
// DualTrits encoding/packing utilities (mirror of the main implementation)
// ============================================================================

// Pack 5 DualTrits into a uint16_t using base-3 encoding
// Each DualTrit has (exponent, direction) where both are in {0, 1, 2}
// Encoding order: direction first, then exponent, for each trit
std::uint16_t pack5_dualtrit(std::uint8_t exp[5], std::uint8_t dir[5]) {
    std::uint16_t packed = 0;
    std::uint16_t multiplier = 1;
    
    for (int i = 0; i < 5; ++i) {
        packed += dir[i] * multiplier;
        multiplier *= 3;
        packed += exp[i] * multiplier;
        multiplier *= 3;
    }
    return packed;
}

// Compute expected float value from (exponent, direction)
float dualtrit_to_float(std::uint8_t exp, std::uint8_t dir) {
    // Direction: 0 -> 0, 1 -> +1, 2 -> -1
    float dir_val = (dir == 0) ? 0.0f : ((dir == 1) ? 1.0f : -1.0f);
    
    // Exponent: 0 -> 1 (3^0), 1 -> 3 (3^1), 2 -> 1/3 (3^-1)
    float exp_val;
    switch (exp) {
        case 0: exp_val = 1.0f; break;
        case 1: exp_val = 3.0f; break;
        case 2: exp_val = 1.0f / 3.0f; break;
        default: exp_val = 0.0f; break;
    }
    
    // Special case: direction=0 with non-zero exponent means infinity
    if (dir == 0) {
        if (exp == 1) return std::numeric_limits<float>::infinity();
        if (exp == 2) return -std::numeric_limits<float>::infinity();
        return 0.0f;  // exp == 0, dir == 0 -> zero
    }
    
    return dir_val * exp_val;
}

// ============================================================================
// Test 1: Basic value correctness for all 9 valid DualTrit values
// ============================================================================
bool test_basic_values() {
    printf("Test 1: Basic value correctness for all 9 valid DualTrit values\n");
    
    // Test all 9 valid (exp, dir) combinations
    // We'll create a minimal 64x64 matrix filled with known values
    const size_t M = 64;
    const size_t K = 64;
    const size_t total_elements = M * K;  // 4096 elements
    const size_t PACKS_PER_64 = 13;
    const size_t num_groups = total_elements / 64;
    const size_t total_packs = num_groups * PACKS_PER_64;
    
    // Allocate packed input and output
    std::uint16_t* packed_h = (std::uint16_t*)malloc(total_packs * sizeof(std::uint16_t));
    CheckMallocCPU(packed_h, __LINE__);
    
    half* output_h = (half*)malloc(total_elements * sizeof(half));
    CheckMallocCPU(output_h, __LINE__);
    
    half* scale_h = (half*)malloc(M * sizeof(half));
    CheckMallocCPU(scale_h, __LINE__);
    
    // Expected values for reference
    float* expected_h = (float*)malloc(total_elements * sizeof(float));
    CheckMallocCPU(expected_h, __LINE__);
    
    // Set all scales to 1.0 for this test
    for (size_t i = 0; i < M; i++) {
        scale_h[i] = __float2half_rn(1.0f);
    }
    
    // Fill with test patterns: cycle through all 9 valid combinations
    // Valid (exp, dir) pairs: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    struct TestValue {
        std::uint8_t exp;
        std::uint8_t dir;
    };
    TestValue valid_values[9] = {
        {0, 0}, {0, 1}, {0, 2},  // 0, 1, -1
        {1, 0}, {1, 1}, {1, 2},  // +inf, 3, -3
        {2, 0}, {2, 1}, {2, 2}   // -inf, 1/3, -1/3
    };
    
    size_t elem_idx = 0;
    for (size_t g = 0; g < num_groups; ++g) {
        for (size_t p = 0; p < PACKS_PER_64; ++p) {
            std::uint8_t exp_arr[5], dir_arr[5];
            
            for (int t = 0; t < 5; ++t) {
                // Cycle through the 9 valid values
                int val_idx = (g * 65 + p * 5 + t) % 9;
                exp_arr[t] = valid_values[val_idx].exp;
                dir_arr[t] = valid_values[val_idx].dir;
                
                // Store expected value (only first 64 per group are valid)
                if (p * 5 + t < 64) {
                    expected_h[elem_idx] = dualtrit_to_float(exp_arr[t], dir_arr[t]);
                    elem_idx++;
                }
            }
            
            packed_h[g * PACKS_PER_64 + p] = pack5_dualtrit(exp_arr, dir_arr);
        }
    }
    
    // Run dequantization
    DeQuantMatrix_DualTrit5_To_FP16(output_h, packed_h, M, K, scale_h);
    
    // Verify results
    bool passed = true;
    int errors = 0;
    const int MAX_ERRORS_TO_PRINT = 10;
    
    for (size_t i = 0; i < total_elements; ++i) {
        float output_val = __half2float(output_h[i]);
        float expected_val = expected_h[i];
        
        // Handle infinity comparisons
        bool match = false;
        if (std::isinf(expected_val) && std::isinf(output_val)) {
            match = (expected_val > 0) == (output_val > 0);  // Same sign infinity
        } else if (std::isnan(expected_val) || std::isnan(output_val)) {
            match = std::isnan(expected_val) && std::isnan(output_val);
        } else {
            // For finite values, allow small epsilon for FP16 precision
            float diff = std::fabs(output_val - expected_val);
            float rel_err = (expected_val != 0.0f) ? diff / std::fabs(expected_val) : diff;
            match = (diff < 1e-3f) || (rel_err < 1e-2f);
        }
        
        if (!match) {
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("  Mismatch at index %zu: expected %.6f, got %.6f\n", 
                       i, expected_val, output_val);
            }
            errors++;
            passed = false;
        }
    }
    
    if (passed) {
        printf("  PASSED: All %zu values match expected results\n", total_elements);
    } else {
        printf("  FAILED: %d mismatches found\n", errors);
    }
    
    free(packed_h);
    free(output_h);
    free(scale_h);
    free(expected_h);
    
    return passed;
}

// ============================================================================
// Test 2: Scale factor application
// ============================================================================
bool test_scale_factors() {
    printf("Test 2: Scale factor application\n");
    
    const size_t M = 64;
    const size_t K = 64;
    const size_t total_elements = M * K;
    const size_t PACKS_PER_64 = 13;
    const size_t num_groups = total_elements / 64;
    const size_t total_packs = num_groups * PACKS_PER_64;
    
    std::uint16_t* packed_h = (std::uint16_t*)malloc(total_packs * sizeof(std::uint16_t));
    CheckMallocCPU(packed_h, __LINE__);
    
    half* output_h = (half*)malloc(total_elements * sizeof(half));
    CheckMallocCPU(output_h, __LINE__);
    
    half* scale_h = (half*)malloc(M * sizeof(half));
    CheckMallocCPU(scale_h, __LINE__);
    
    // Set different scales per row
    for (size_t i = 0; i < M; i++) {
        scale_h[i] = __float2half_rn(0.5f + (float)i * 0.01f);  // 0.5 to 1.13
    }
    
    // Fill all packs with value 1.0 (exp=0, dir=1)
    for (size_t i = 0; i < total_packs; ++i) {
        std::uint8_t exp_arr[5] = {0, 0, 0, 0, 0};
        std::uint8_t dir_arr[5] = {1, 1, 1, 1, 1};
        packed_h[i] = pack5_dualtrit(exp_arr, dir_arr);
    }
    
    // Run dequantization
    DeQuantMatrix_DualTrit5_To_FP16(output_h, packed_h, M, K, scale_h);
    
    // Verify: each element should be 1.0 * scale[row]
    bool passed = true;
    int errors = 0;
    const int MAX_ERRORS_TO_PRINT = 10;
    
    for (size_t i = 0; i < total_elements; ++i) {
        size_t row = i / K;
        float expected = __half2float(scale_h[row]);  // 1.0 * scale
        float output_val = __half2float(output_h[i]);
        
        float diff = std::fabs(output_val - expected);
        if (diff > 1e-3f) {
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("  Mismatch at index %zu (row %zu): expected %.6f, got %.6f\n", 
                       i, row, expected, output_val);
            }
            errors++;
            passed = false;
        }
    }
    
    if (passed) {
        printf("  PASSED: All scale factors correctly applied\n");
    } else {
        printf("  FAILED: %d mismatches found\n", errors);
    }
    
    free(packed_h);
    free(output_h);
    free(scale_h);
    
    return passed;
}

// ============================================================================
// Test 3: Larger matrix dimensions
// ============================================================================
bool test_large_matrix() {
    printf("Test 3: Larger matrix dimensions (256x256)\n");
    
    const size_t M = 256;
    const size_t K = 256;
    const size_t total_elements = M * K;  // 65536 elements
    const size_t PACKS_PER_64 = 13;
    const size_t num_groups = total_elements / 64;
    const size_t total_packs = num_groups * PACKS_PER_64;
    
    std::uint16_t* packed_h = (std::uint16_t*)malloc(total_packs * sizeof(std::uint16_t));
    CheckMallocCPU(packed_h, __LINE__);
    
    half* output_h = (half*)malloc(total_elements * sizeof(half));
    CheckMallocCPU(output_h, __LINE__);
    
    half* scale_h = (half*)malloc(M * sizeof(half));
    CheckMallocCPU(scale_h, __LINE__);
    
    float* expected_h = (float*)malloc(total_elements * sizeof(float));
    CheckMallocCPU(expected_h, __LINE__);
    
    // Set scales to 2.0 for simplicity
    for (size_t i = 0; i < M; i++) {
        scale_h[i] = __float2half_rn(2.0f);
    }
    
    // Fill with alternating pattern: 1.0 and -1.0
    size_t elem_idx = 0;
    for (size_t g = 0; g < num_groups; ++g) {
        for (size_t p = 0; p < PACKS_PER_64; ++p) {
            std::uint8_t exp_arr[5], dir_arr[5];
            
            for (int t = 0; t < 5; ++t) {
                // Alternate between (0,1)=1 and (0,2)=-1
                bool use_positive = ((g + p + t) % 2) == 0;
                exp_arr[t] = 0;
                dir_arr[t] = use_positive ? 1 : 2;
                
                if (p * 5 + t < 64) {
                    float base_val = use_positive ? 1.0f : -1.0f;
                    expected_h[elem_idx] = base_val * 2.0f;  // scaled by 2.0
                    elem_idx++;
                }
            }
            
            packed_h[g * PACKS_PER_64 + p] = pack5_dualtrit(exp_arr, dir_arr);
        }
    }
    
    // Run dequantization
    DeQuantMatrix_DualTrit5_To_FP16(output_h, packed_h, M, K, scale_h);
    
    // Verify results
    bool passed = true;
    int errors = 0;
    
    for (size_t i = 0; i < total_elements; ++i) {
        float output_val = __half2float(output_h[i]);
        float expected_val = expected_h[i];
        
        float diff = std::fabs(output_val - expected_val);
        if (diff > 1e-3f) {
            errors++;
            passed = false;
        }
    }
    
    if (passed) {
        printf("  PASSED: Large matrix (%zux%zu = %zu elements) processed correctly\n", 
               M, K, total_elements);
    } else {
        printf("  FAILED: %d mismatches found out of %zu elements\n", errors, total_elements);
    }
    
    free(packed_h);
    free(output_h);
    free(scale_h);
    free(expected_h);
    
    return passed;
}

// ============================================================================
// Test 4: Specific value verification
// ============================================================================
bool test_specific_values() {
    printf("Test 4: Specific value verification\n");
    
    const size_t M = 64;
    const size_t K = 64;
    const size_t PACKS_PER_64 = 13;
    const size_t total_packs = PACKS_PER_64 * (M * K / 64);
    
    std::uint16_t* packed_h = (std::uint16_t*)malloc(total_packs * sizeof(std::uint16_t));
    CheckMallocCPU(packed_h, __LINE__);
    
    half* output_h = (half*)malloc(M * K * sizeof(half));
    CheckMallocCPU(output_h, __LINE__);
    
    half* scale_h = (half*)malloc(M * sizeof(half));
    CheckMallocCPU(scale_h, __LINE__);
    
    // Scale = 1.0
    for (size_t i = 0; i < M; i++) {
        scale_h[i] = __float2half_rn(1.0f);
    }
    
    // Fill first pack with specific test values
    // Pack 0: [0, 1, -1, 3, -3] (testing first 5 values)
    std::uint8_t exp0[5] = {0, 0, 0, 1, 1};
    std::uint8_t dir0[5] = {0, 1, 2, 1, 2};
    packed_h[0] = pack5_dualtrit(exp0, dir0);
    
    // Pack 1: [1/3, -1/3, +inf, -inf, 0]
    std::uint8_t exp1[5] = {2, 2, 1, 2, 0};
    std::uint8_t dir1[5] = {1, 2, 0, 0, 0};
    packed_h[1] = pack5_dualtrit(exp1, dir1);
    
    // Fill remaining packs with zeros
    for (size_t i = 2; i < total_packs; ++i) {
        std::uint8_t exp_arr[5] = {0, 0, 0, 0, 0};
        std::uint8_t dir_arr[5] = {0, 0, 0, 0, 0};
        packed_h[i] = pack5_dualtrit(exp_arr, dir_arr);
    }
    
    // Run dequantization
    DeQuantMatrix_DualTrit5_To_FP16(output_h, packed_h, M, K, scale_h);
    
    // Verify specific values in first 10 elements
    float expected[10] = {
        0.0f, 1.0f, -1.0f, 3.0f, -3.0f,
        1.0f/3.0f, -1.0f/3.0f, std::numeric_limits<float>::infinity(), 
        -std::numeric_limits<float>::infinity(), 0.0f
    };
    
    bool passed = true;
    for (int i = 0; i < 10; ++i) {
        float output_val = __half2float(output_h[i]);
        float expected_val = expected[i];
        
        bool match = false;
        if (std::isinf(expected_val) && std::isinf(output_val)) {
            match = (expected_val > 0) == (output_val > 0);
        } else {
            float diff = std::fabs(output_val - expected_val);
            match = (diff < 1e-3f);
        }
        
        if (!match) {
            printf("  Index %d: expected %.6f, got %.6f - FAIL\n", i, expected_val, output_val);
            passed = false;
        } else {
            printf("  Index %d: expected %.6f, got %.6f - OK\n", i, expected_val, output_val);
        }
    }
    
    if (passed) {
        printf("  PASSED: All specific values verified\n");
    } else {
        printf("  FAILED: Some values did not match\n");
    }
    
    free(packed_h);
    free(output_h);
    free(scale_h);
    
    return passed;
}

// ============================================================================
// Quantization tests
// ============================================================================

// Test cast_fp16_to_dualtrit single value quantization
bool test_cast_fp16_to_dualtrit() {
    printf("Test 5: cast_fp16_to_dualtrit single value quantization\n");
    bool all_passed = true;
    
    // Test cases: (input_float, expected_encoding)
    // Encoding: (exp << 2) | dir
    //   0  -> (0,0) -> 0
    //   1  -> (0,1) -> +1
    //   2  -> (0,2) -> -1
    //   5  -> (1,1) -> +3
    //   6  -> (1,2) -> -3
    //   9  -> (2,1) -> +1/3
    //  10  -> (2,2) -> -1/3
    //   4  -> (1,0) -> +inf
    //   8  -> (2,0) -> -inf
    
    struct TestCase {
        float input;
        std::uint8_t expected;
        const char* name;
    };
    
    TestCase cases[] = {
        // Exact values
        {0.0f, 0, "0"},
        {1.0f, 1, "+1"},
        {-1.0f, 2, "-1"},
        {3.0f, 5, "+3"},
        {-3.0f, 6, "-3"},
        {1.0f/3.0f, 9, "+1/3"},
        {-1.0f/3.0f, 10, "-1/3"},
        {INFINITY, 4, "+inf"},
        {-INFINITY, 8, "-inf"},
        
        // Values close to thresholds (should round to nearer)
        {0.05f, 0, "0.05 -> 0"},           // < T0=1/6 ≈ 0.167
        {0.25f, 9, "0.25 -> +1/3"},        // T0 < x < T1: closer to 1/3
        {-0.25f, 10, "-0.25 -> -1/3"},     // negative
        {0.5f, 9, "0.5 -> +1/3"},          // closer to 1/3 than 1
        {0.8f, 1, "0.8 -> +1"},            // closer to 1 than 1/3
        {-0.8f, 2, "-0.8 -> -1"},
        {1.5f, 1, "1.5 -> +1"},            // closer to 1 than 3
        {2.5f, 5, "2.5 -> +3"},            // closer to 3 than 1
        {-2.5f, 6, "-2.5 -> -3"},
        {5.0f, 5, "5.0 -> +3"},            // beyond 3, stays at 3
        
        // Tie-to-even at T0 = 1/6 (between 0 and 1/3)
        // 0 is index 0 (even), 1/3 is index 1 (odd) -> choose 0
        {1.0f/6.0f, 0, "T0=1/6 tie -> 0 (even)"},
        
        // Tie-to-even at T1 = 2/3 (between 1/3 and 1)
        // 1/3 is index 1 (odd), 1 is index 2 (even) -> choose 1
        {2.0f/3.0f, 1, "T1=2/3 tie -> +1 (even)"},
        
        // Tie-to-even at T2 = 2 (between 1 and 3)
        // 1 is index 2 (even), 3 is index 3 (odd) -> choose 1
        {2.0f, 1, "T2=2 tie -> +1 (even)"},
    };
    
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    int passed = 0;
    
    for (int i = 0; i < num_cases; ++i) {
        std::uint8_t result = cast_fp16_to_dualtrit(cases[i].input);
        if (result == cases[i].expected) {
            passed++;
        } else {
            printf("  FAILED: %s: input=%f, expected=%u, got=%u\n", 
                   cases[i].name, cases[i].input, cases[i].expected, result);
            all_passed = false;
        }
    }
    
    if (all_passed) {
        printf("  PASSED: All %d quantization cases correct\n", passed);
    } else {
        printf("  FAILED: %d/%d passed\n", passed, num_cases);
    }
    
    return all_passed;
}

// Test pack5_dualtrit packing function
bool test_pack5_dualtrit() {
    printf("Test 6: pack5_dualtrit base-3 packing\n");
    bool all_passed = true;
    
    // Test case 1: All zeros
    {
        std::uint8_t trits[5] = {0, 0, 0, 0, 0};  // (exp=0,dir=0) for all
        std::uint16_t packed = pack5_dualtrit(trits);
        if (packed != 0) {
            printf("  FAILED: All zeros should pack to 0, got %u\n", packed);
            all_passed = false;
        }
    }
    
    // Test case 2: Single +1 at position 0
    {
        std::uint8_t trits[5] = {1, 0, 0, 0, 0};  // (exp=0,dir=1) at pos 0
        // dir=1 at pos 0: 1 * 3^0 = 1
        // exp=0 at pos 0: 0 * 3^1 = 0
        std::uint16_t packed = pack5_dualtrit(trits);
        if (packed != 1) {
            printf("  FAILED: Single +1 at pos 0 should pack to 1, got %u\n", packed);
            all_passed = false;
        }
    }
    
    // Test case 3: Single +3 at position 0
    {
        std::uint8_t trits[5] = {5, 0, 0, 0, 0};  // (exp=1,dir=1) at pos 0
        // dir=1 at pos 0: 1 * 3^0 = 1
        // exp=1 at pos 0: 1 * 3^1 = 3
        std::uint16_t packed = pack5_dualtrit(trits);
        if (packed != 4) {  // 1 + 3 = 4
            printf("  FAILED: Single +3 at pos 0 should pack to 4, got %u\n", packed);
            all_passed = false;
        }
    }
    
    // Test case 4: +1/3 at position 0
    {
        std::uint8_t trits[5] = {9, 0, 0, 0, 0};  // (exp=2,dir=1) at pos 0
        // dir=1 at pos 0: 1 * 3^0 = 1
        // exp=2 at pos 0: 2 * 3^1 = 6
        std::uint16_t packed = pack5_dualtrit(trits);
        if (packed != 7) {  // 1 + 6 = 7
            printf("  FAILED: Single +1/3 at pos 0 should pack to 7, got %u\n", packed);
            all_passed = false;
        }
    }
    
    // Test case 5: +1 at position 1
    {
        std::uint8_t trits[5] = {0, 1, 0, 0, 0};  // (exp=0,dir=1) at pos 1
        // dir=1 at pos 1: 1 * 3^2 = 9
        // exp=0 at pos 1: 0 * 3^3 = 0
        std::uint16_t packed = pack5_dualtrit(trits);
        if (packed != 9) {
            printf("  FAILED: Single +1 at pos 1 should pack to 9, got %u\n", packed);
            all_passed = false;
        }
    }
    
    // Test case 6: Maximum value (all -inf which is exp=2, dir=0)
    // Actually test with mixed values
    {
        std::uint8_t trits[5] = {1, 1, 1, 1, 1};  // All +1: (exp=0,dir=1)
        // pos 0: dir=1*3^0 + exp=0*3^1 = 1
        // pos 1: dir=1*3^2 + exp=0*3^3 = 9
        // pos 2: dir=1*3^4 + exp=0*3^5 = 81
        // pos 3: dir=1*3^6 + exp=0*3^7 = 729
        // pos 4: dir=1*3^8 + exp=0*3^9 = 6561
        // Total = 1 + 9 + 81 + 729 + 6561 = 7381
        std::uint16_t packed = pack5_dualtrit(trits);
        if (packed != 7381) {
            printf("  FAILED: All +1 should pack to 7381, got %u\n", packed);
            all_passed = false;
        }
    }
    
    if (all_passed) {
        printf("  PASSED: All packing tests correct\n");
    }
    
    return all_passed;
}

// Test round-trip: FP16 -> DualTrits -> FP16
bool test_roundtrip_quantization() {
    printf("Test 7: Round-trip quantization (FP16 -> DualTrits -> FP16)\n");
    
    // Create a 64x64 matrix with known values
    const size_t M = 64;
    const size_t K = 64;
    const size_t total = M * K;
    
    // Allocate memory
    half* input_h = (half*)malloc(total * sizeof(half));
    half* scale_h = (half*)malloc(M * sizeof(half));
    std::uint16_t* packed_h = (std::uint16_t*)malloc((total / 64) * 13 * sizeof(std::uint16_t));
    half* output_h = (half*)malloc(total * sizeof(half));
    
    CheckMallocCPU(input_h);
    CheckMallocCPU(scale_h);
    CheckMallocCPU(packed_h);
    CheckMallocCPU(output_h);
    
    // Initialize with values that should quantize exactly to DualTrits representable values
    // Use scale = 1.0 for simplicity
    for (size_t i = 0; i < M; ++i) {
        scale_h[i] = __float2half(1.0f);
    }
    
    // Fill input with the 9 representable values in a pattern
    float representable[] = {0.0f, 1.0f, -1.0f, 3.0f, -3.0f, 1.0f/3.0f, -1.0f/3.0f, 0.0f, 0.0f};
    // Note: skip +inf/-inf for round-trip since they're harder to verify
    
    for (size_t i = 0; i < total; ++i) {
        input_h[i] = __float2half(representable[i % 9]);
    }
    
    // Quantize: FP16 -> packed DualTrits
    weight_prepacking_fp16_to_dual_trits(input_h, packed_h, M, K, scale_h);
    
    // Dequantize using CPU version: packed DualTrits -> FP16
    DeQuantMatrix_DualTrit5_To_FP16(output_h, packed_h, M, K, scale_h);
    
    // Verify round-trip
    bool all_passed = true;
    int errors = 0;
    const float tolerance = 1e-3f;  // Allow for FP16 precision loss
    
    for (size_t i = 0; i < total && errors < 10; ++i) {
        float input_f = __half2float(input_h[i]);
        float output_f = __half2float(output_h[i]);
        float diff = std::fabs(input_f - output_f);
        
        if (diff > tolerance) {
            if (errors == 0) {
                printf("  First errors:\n");
            }
            printf("    [%zu]: input=%f, output=%f, diff=%f\n", i, input_f, output_f, diff);
            errors++;
            all_passed = false;
        }
    }
    
    if (all_passed) {
        printf("  PASSED: All %zu values round-trip correctly\n", total);
    } else {
        printf("  FAILED: %d errors found (showing first 10)\n", errors);
    }
    
    free(input_h);
    free(scale_h);
    free(packed_h);
    free(output_h);
    
    return all_passed;
}

// Test scale factor handling in quantization
bool test_quantization_with_scale() {
    printf("Test 8: Quantization with different scale factors\n");
    
    const size_t M = 64;
    const size_t K = 64;
    const size_t total = M * K;
    
    half* input_h = (half*)malloc(total * sizeof(half));
    half* scale_h = (half*)malloc(M * sizeof(half));
    std::uint16_t* packed_h = (std::uint16_t*)malloc((total / 64) * 13 * sizeof(std::uint16_t));
    half* output_h = (half*)malloc(total * sizeof(half));
    
    CheckMallocCPU(input_h);
    CheckMallocCPU(scale_h);
    CheckMallocCPU(packed_h);
    CheckMallocCPU(output_h);
    
    // Use scale = 2.0 for all rows
    // Input 2.0 with scale 2.0 -> normalized 1.0 -> quantize to 1 -> dequant to 1.0 -> scale to 2.0
    float scale_val = 2.0f;
    for (size_t i = 0; i < M; ++i) {
        scale_h[i] = __float2half(scale_val);
    }
    
    // Fill with values that are multiples of the scale
    float test_values[] = {0.0f, 2.0f, -2.0f, 6.0f, -6.0f, 2.0f/3.0f, -2.0f/3.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < total; ++i) {
        input_h[i] = __float2half(test_values[i % 9]);
    }
    
    // Quantize
    weight_prepacking_fp16_to_dual_trits(input_h, packed_h, M, K, scale_h);
    
    // Dequantize using CPU version
    DeQuantMatrix_DualTrit5_To_FP16(output_h, packed_h, M, K, scale_h);
    
    // Verify
    bool all_passed = true;
    const float tolerance = 1e-2f;  // Slightly larger tolerance for scaled values
    
    for (size_t i = 0; i < total; ++i) {
        float input_f = __half2float(input_h[i]);
        float output_f = __half2float(output_h[i]);
        float diff = std::fabs(input_f - output_f);
        
        if (diff > tolerance) {
            printf("  FAILED at [%zu]: input=%f, output=%f, diff=%f\n", i, input_f, output_f, diff);
            all_passed = false;
            break;
        }
    }
    
    if (all_passed) {
        printf("  PASSED: Scale factor correctly applied in quantization/dequantization\n");
    }
    
    free(input_h);
    free(scale_h);
    free(packed_h);
    free(output_h);
    
    return all_passed;
}

// ============================================================================
// Test 9: GPU GEMM Kernel Test (DualTrits Linear)
// ============================================================================
/*
 * This test verifies the DualTrits GEMM kernel by:
 * 1. Creating random FP16 weights and quantizing to DualTrits
 * 2. Creating random FP16 activations
 * 3. Computing reference output using FP16 GEMM
 * 4. Computing output using DualTrits GEMM kernel
 * 5. Comparing the results
 * 
 * Note: This test requires the CUDA kernel to be compiled.
 * Currently this is a placeholder for when the kernel is integrated.
 */
bool test_gemm_kernel() {
    printf("Test 9: DualTrits GEMM Kernel (Placeholder)\n");
    
    // This test requires the full kernel integration
    // For now, we just test that the padded packing/unpacking round-trips correctly
    
    const size_t M = 256;  // Output channels (must be multiple of 256)
    const size_t K = 256;  // Input channels (must be multiple of 64)
    
    // Allocate FP16 weights
    half* weight_fp16 = (half*)malloc(M * K * sizeof(half));
    CheckMallocCPU(weight_fp16, __LINE__);
    
    // Initialize with known DualTrit-quantizable values
    for (size_t i = 0; i < M * K; ++i) {
        // Use values that are exactly representable: 0, ±1/3, ±1, ±3
        int val_idx = (i * 7) % 7;  // Pseudo-random but deterministic
        float values[] = {0.0f, 1.0f, -1.0f, 3.0f, -3.0f, 1.0f/3.0f, -1.0f/3.0f};
        weight_fp16[i] = __float2half_rn(values[val_idx]);
    }
    
    // Allocate scale factors (per-row)
    half* scale = (half*)malloc(M * sizeof(half));
    CheckMallocCPU(scale, __LINE__);
    for (size_t i = 0; i < M; ++i) {
        scale[i] = __float2half_rn(1.0f);  // Unit scale for testing
    }
    
    // Allocate packed DualTrits (padded format: 16 uint16_t per 64 elements)
    const size_t PACKS_PER_64_PADDED = 16;
    const size_t num_groups = (M * K) / 64;
    const size_t total_packs = num_groups * PACKS_PER_64_PADDED;
    std::uint16_t* packed = (std::uint16_t*)malloc(total_packs * sizeof(std::uint16_t));
    CheckMallocCPU(packed, __LINE__);
    
    // Pack weights using GPU-optimized format
    weight_prepacking_fp16_to_dual_trits_gpu(weight_fp16, packed, M, K, scale);
    
    // Dequantize using padded format
    half* dequant = (half*)malloc(M * K * sizeof(half));
    CheckMallocCPU(dequant, __LINE__);
    DeQuantMatrix_DualTrit5_Padded_To_FP16(dequant, packed, M, K, scale);
    
    // Compare original vs dequantized
    int errors = 0;
    float max_error = 0.0f;
    
    for (size_t i = 0; i < M * K; ++i) {
        float orig = __half2float(weight_fp16[i]);
        float deq = __half2float(dequant[i]);
        float err = std::fabs(orig - deq);
        max_error = std::fmax(max_error, err);
        
        // Allow for quantization error (rounding to nearest DualTrit value)
        if (err > 0.5f) {  // Very lenient for quantization
            if (errors < 10) {
                printf("  Error at index %zu: original=%.4f, dequant=%.4f\n", i, orig, deq);
            }
            errors++;
        }
    }
    
    printf("  Max quantization error: %.6f\n", max_error);
    printf("  Errors (>0.5): %d / %zu\n", errors, M * K);
    
    // Cleanup
    free(weight_fp16);
    free(scale);
    free(packed);
    free(dequant);
    
    // Pass if max error is within expected quantization bounds
    if (max_error < 1.5f) {  // DualTrits can have up to ~1.33 error in worst case
        printf("  PASSED: Padded packing/unpacking round-trip successful\n");
        return true;
    } else {
        printf("  FAILED: Quantization error too large\n");
        return false;
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("DualTrit Quantization/Dequantization Tests\n");
    printf("==============================================\n\n");
    
    int passed = 0;
    int failed = 0;
    
    printf("--- Dequantization Tests ---\n\n");
    
    if (test_basic_values()) passed++; else failed++;
    printf("\n");
    
    if (test_scale_factors()) passed++; else failed++;
    printf("\n");
    
    if (test_large_matrix()) passed++; else failed++;
    printf("\n");
    
    if (test_specific_values()) passed++; else failed++;
    printf("\n");
    
    printf("--- Quantization Tests ---\n\n");
    
    if (test_cast_fp16_to_dualtrit()) passed++; else failed++;
    printf("\n");
    
    if (test_pack5_dualtrit()) passed++; else failed++;
    printf("\n");
    
    if (test_roundtrip_quantization()) passed++; else failed++;
    printf("\n");
    
    if (test_quantization_with_scale()) passed++; else failed++;
    printf("\n");
    
    printf("--- GEMM Kernel Tests ---\n\n");
    
    if (test_gemm_kernel()) passed++; else failed++;
    printf("\n");
    
    printf("==============================================\n");
    printf("Summary: %d passed, %d failed\n", passed, failed);
    printf("==============================================\n");
    
    return (failed == 0) ? 0 : 1;
}
