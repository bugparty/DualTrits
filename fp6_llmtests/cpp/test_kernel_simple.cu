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
    // Minimal test: 256x256 x 256x32
    const size_t M = 256, K = 256, N = 32;
    
    // Allocate host memory
    half* A_fp16 = new half[M * K];
    half* A_dequant = new half[M * K];
    half* B_fp16 = new half[K * N];
    half* C_cublas = new half[M * N];
    half* C_dualtrit = new half[M * N];
    half* scale = new half[M];
    uint16_t* A_packed = new uint16_t[M * K / 64 * 16];
    
    // Initialize with simple values
    // A = all 1.0, B = all 1.0
    for (size_t i = 0; i < M * K; i++) A_fp16[i] = __float2half_rn(1.0f);
    for (size_t i = 0; i < M; i++) scale[i] = __float2half_rn(1.0f);
    for (size_t i = 0; i < K * N; i++) B_fp16[i] = __float2half_rn(1.0f);
    
    // Pack A
    weight_prepacking_fp16_to_dual_trits_gpu_warptile(A_fp16, A_packed, M, K, scale);
    DeQuantMatrix_DualTrit5_Warptile_To_FP16(A_dequant, A_packed, M, K, scale);
    
    // Verify pack/unpack
    printf("Pack/Unpack verification:\n");
    int pack_errors = 0;
    for (size_t i = 0; i < M * K; i++) {
        if (fabsf(__half2float(A_fp16[i]) - __half2float(A_dequant[i])) > 0.01f) pack_errors++;
    }
    printf("  Pack/Unpack errors: %d / %zu\n", pack_errors, M * K);
    
    // Allocate device memory
    uint16_t *d_A_packed;
    half *d_A_dequant, *d_scale, *d_B, *d_C_cublas, *d_C_dualtrit;
    float *d_workspace;
    
    cudaMalloc(&d_A_packed, M * K / 64 * 16 * sizeof(uint16_t));
    cudaMalloc(&d_A_dequant, M * K * sizeof(half));
    cudaMalloc(&d_scale, M * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C_cublas, M * N * sizeof(half));
    cudaMalloc(&d_C_dualtrit, M * N * sizeof(half));
    cudaMalloc(&d_workspace, M * N * sizeof(float));
    
    cudaMemcpy(d_A_packed, A_packed, M * K / 64 * 16 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_dequant, A_dequant, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, scale, M * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_fp16, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C_cublas, 0, M * N * sizeof(half));
    cudaMemset(d_C_dualtrit, 0, M * N * sizeof(half));
    
    // cuBLAS reference: C = A^T * B
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 M, N, K, &alpha,
                 d_A_dequant, CUDA_R_16F, K,
                 d_B, CUDA_R_16F, K,
                 &beta, d_C_cublas, CUDA_R_16F, M,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    
    // DualTrits kernel
    dualtrits_linear_kernel(0, (uint4*)d_A_packed, d_scale, d_B, d_C_dualtrit, M, N, K, d_workspace, 1);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(C_cublas, d_C_cublas, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_dualtrit, d_C_dualtrit, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("\nResults (first 5 elements of column 0):\n");
    printf("  Expected (K*1.0 = %zu):\n", K);
    for (int i = 0; i < 5; i++) {
        float cublas_val = __half2float(C_cublas[i]);  // C[i,0] in column-major
        float dt_val = __half2float(C_dualtrit[i]);
        printf("  C[%d,0]: cuBLAS=%.2f, DualTrits=%.2f\n", i, cublas_val, dt_val);
    }
    
    // Check for errors
    int errors = 0;
    printf("\nChecking for mismatches...\n");
    for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
            size_t idx = n * M + m; // Column-major
            float cb = __half2float(C_cublas[idx]);
            float dt = __half2float(C_dualtrit[idx]);
            if (fabsf(cb - dt) > 1.0f) {
                if (errors < 10) {
                    printf("  Mismatch at C[%zu,%zu]: cuBLAS=%.2f, DualTrits=%.2f\n", m, n, cb, dt);
                }
                errors++;
            }
        }
    }
    printf("Total mismatches: %d / %zu\n", errors, M * N);
    
    // Total error
    double abs_sum = 0, error_sum = 0;
    for (size_t i = 0; i < M * N; i++) {
        float cb = __half2float(C_cublas[i]);
        float dt = __half2float(C_dualtrit[i]);
        abs_sum += fabsf(cb);
        error_sum += fabsf(cb - dt);
    }
    printf("\nTotal Error/Sum: %.6f\n", error_sum / abs_sum);
    
    // Cleanup
    delete[] A_fp16; delete[] A_dequant; delete[] B_fp16;
    delete[] C_cublas; delete[] C_dualtrit; delete[] scale; delete[] A_packed;
    cudaFree(d_A_packed); cudaFree(d_A_dequant); cudaFree(d_scale);
    cudaFree(d_B); cudaFree(d_C_cublas); cudaFree(d_C_dualtrit); cudaFree(d_workspace);
    cublasDestroy(handle);
    
    return 0;
}
