#include <stdio.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "../../fp6_llm/csrc/utils/weight_quant.h"
#include "../../fp6_llm/csrc/utils/weight_dequant.h"

int main() {
    // Small test: 64x64 matrix with known values
    const int M = 64, K = 64;
    
    half* A_orig = new half[M * K];
    half* A_dequant = new half[M * K];
    half* scale = new half[M];
    uint16_t* packed = new uint16_t[M * K / 64 * 16];
    
    // Initialize with 1.0
    for (int i = 0; i < M * K; i++) A_orig[i] = __float2half_rn(1.0f);
    for (int i = 0; i < M; i++) scale[i] = __float2half_rn(1.0f);
    
    // Pack
    weight_prepacking_fp16_to_dual_trits_gpu_warptile(A_orig, packed, M, K, scale);
    
    // Unpack
    DeQuantMatrix_DualTrit5_Warptile_To_FP16(A_dequant, packed, M, K, scale);
    
    // Compare
    int errors = 0;
    for (int i = 0; i < M * K; i++) {
        float orig = __half2float(A_orig[i]);
        float dq = __half2float(A_dequant[i]);
        if (fabs(orig - dq) > 0.01f) {
            if (errors < 10) printf("[%d] orig=%f, dequant=%f\n", i, orig, dq);
            errors++;
        }
    }
    printf("Errors: %d / %d\n", errors, M * K);
    
    delete[] A_orig;
    delete[] A_dequant;
    delete[] scale;
    delete[] packed;
    return 0;
}
