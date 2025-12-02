#include <stdio.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include <cublas_v2.h>
#include "../../fp6_llm/csrc/utils/weight_quant.h"
#include "../../fp6_llm/csrc/utils/weight_dequant.h"

cudaError_t dualtrits_linear_kernel(cudaStream_t stream,
                                     const uint4* Weight, const half* Scales,
                                     const half* B, half* C,
                                     size_t M_Global, size_t N_Global, size_t K_Global,
                                     float* Reduction_Workspace,
                                     int Split_K);

int main() {
    // Minimal test: 256x256 x 256x8
    const size_t M = 256, K = 256, N = 8;
    
    // Allocate host memory
    half* A_fp16 = new half[M * K];
    half* A_dequant = new half[M * K];
    half* B_fp16 = new half[K * N];
    half* C_dualtrit = new half[M * N];
    half* scale = new half[M];
    uint16_t* A_packed = new uint16_t[M * K / 64 * 16];
    
    // Initialize: A = all 1.0, B[:,0] = 1, rest = 0
    for (size_t i = 0; i < M * K; i++) A_fp16[i] = __float2half_rn(1.0f);
    for (size_t i = 0; i < M; i++) scale[i] = __float2half_rn(1.0f);
    for (size_t i = 0; i < K * N; i++) B_fp16[i] = __float2half_rn(0.0f);
    for (size_t k = 0; k < K; k++) B_fp16[k * N + 0] = __float2half_rn(1.0f);
    
    // Pack
    weight_prepacking_fp16_to_dual_trits_gpu_warptile(A_fp16, A_packed, M, K, scale);
    DeQuantMatrix_DualTrit5_Warptile_To_FP16(A_dequant, A_packed, M, K, scale);
    
    // Device memory
    uint16_t *d_A; half *d_scale, *d_B, *d_C; float *d_work;
    cudaMalloc(&d_A, M * K / 64 * 16 * sizeof(uint16_t));
    cudaMalloc(&d_scale, M * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMalloc(&d_work, M * N * sizeof(float));
    
    cudaMemcpy(d_A, A_packed, M * K / 64 * 16 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, scale, M * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_fp16, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(half));
    
    printf("Before kernel launch...\n");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Pre-launch CUDA error: %s\n", cudaGetErrorString(err));
    
    // Call kernel
    err = dualtrits_linear_kernel(0, (uint4*)d_A, d_scale, d_B, d_C, M, N, K, d_work, 1);
    printf("Kernel launch returned: %s\n", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
    
    err = cudaGetLastError();
    printf("Post-sync CUDA error: %s\n", cudaGetErrorString(err));
    
    // Get result
    cudaMemcpy(C_dualtrit, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    printf("\nFirst 5 results (C[i,0]):\n");
    for (int i = 0; i < 5; i++) {
        printf("  C[%d,0] = %.2f\n", i, __half2float(C_dualtrit[i]));
    }
    
    // Check if any non-zero
    int nonzero = 0;
    for (size_t i = 0; i < M * N; i++) {
        if (__half2float(C_dualtrit[i]) != 0.0f) nonzero++;
    }
    printf("\nNon-zero count: %d / %zu\n", nonzero, M * N);
    
    // Cleanup
    delete[] A_fp16; delete[] A_dequant; delete[] B_fp16;
    delete[] C_dualtrit; delete[] scale; delete[] A_packed;
    cudaFree(d_A); cudaFree(d_scale); cudaFree(d_B); cudaFree(d_C); cudaFree(d_work);
    
    return 0;
}
