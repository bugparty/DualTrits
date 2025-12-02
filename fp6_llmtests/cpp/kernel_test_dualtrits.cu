/*
 * DualTrits GEMM Kernel Test
 * 
 * Tests the DualTrits linear kernel against cuBLAS FP16 GEMM baseline.
 * 
 * Usage: ./kernel_test_dualtrits M K N SplitK
 *   M      - Number of output channels (must be multiple of 256)
 *   K      - Number of input channels (must be multiple of 64)
 *   N      - Batch size
 *   SplitK - Split-K factor for the kernel
 */

#include "kernel_test.h"
#include "fp6_linear.cuh"

// Include weight quantization/dequantization utilities
#include "../../fp6_llm/csrc/utils/weight_quant.h"
#include "../../fp6_llm/csrc/utils/weight_dequant.h"

int main(int argc, char** argv)
{
    // Parsing the inputs from CLI.
    if (argc != 5) {
        printf("Wrong Inputs! Correct input format: ./kernel_test_dualtrits #Row_Weight #Column_Weight BatchSize SplitK\n");
        printf("Example: ./kernel_test_dualtrits 256 256 32 1\n");
        return -1;
    }
    size_t M_GLOBAL = atoi(argv[1]);
    size_t K_GLOBAL = atoi(argv[2]);
    size_t N_GLOBAL = atoi(argv[3]);
    int    SPLIT_K  = atoi(argv[4]);
    
    // Constraints for DualTrits kernel
    assert(M_GLOBAL % 256 == 0);  // M must be a multiple of 256
    assert(K_GLOBAL % 64 == 0);   // K must be a multiple of 64
    
    printf("DualTrits GEMM Test: M=%zu, K=%zu, N=%zu, SplitK=%d\n", M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Weight Matrix Preparation
    // 
    // For DualTrits: 64 elements -> 16 uint16_t (padded format)
    // Storage size = M * K / 64 * 16 * sizeof(uint16_t) = M * K / 2 bytes
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Allocate FP16 weights (for generating test data and cuBLAS reference)
    half* A_16bit_h = (half*)malloc(M_GLOBAL * K_GLOBAL * sizeof(half));
    CheckMallocCPU(A_16bit_h, __LINE__);
    
    // Initialize with values that are exactly representable in DualTrits
    // DualTrits values: {-3, -1, -1/3, 0, 1/3, 1, 3}
    float dualtrit_values[] = {0.0f, 1.0f, -1.0f, 3.0f, -3.0f, 1.0f/3.0f, -1.0f/3.0f};
    for (size_t i = 0; i < M_GLOBAL * K_GLOBAL; i++) {
        int val_idx = rand() % 7;
        A_16bit_h[i] = __float2half_rn(dualtrit_values[val_idx]);
    }
    
    // Allocate quantization scales (per-row)
    half* A_Scale_h = (half*)malloc(M_GLOBAL * sizeof(half));
    CheckMallocCPU(A_Scale_h, __LINE__);
    for (size_t i = 0; i < M_GLOBAL; i++) {
        // Use scale of 1.0 so the values remain exactly representable
        A_Scale_h[i] = __float2half_rn(1.0f);
    }
    
    // Allocate packed DualTrits weights (padded format: 16 uint16_t per 64 elements)
    // Size = M * K / 64 * 16 uint16_t = M * K / 4 uint16_t = M * K / 2 bytes
    size_t packed_size_bytes = M_GLOBAL * K_GLOBAL / 64 * 16 * sizeof(uint16_t);
    uint16_t* A_packed_h = (uint16_t*)malloc(packed_size_bytes);
    CheckMallocCPU(A_packed_h, __LINE__);
    
    // Quantize and pack weights (using warptile version for correct GEMM kernel layout)
    weight_prepacking_fp16_to_dual_trits_gpu_warptile(A_16bit_h, A_packed_h, M_GLOBAL, K_GLOBAL, A_Scale_h);
    
    // Dequantize for cuBLAS reference (to verify quantization is correct)
    half* A_dequant_h = (half*)malloc(M_GLOBAL * K_GLOBAL * sizeof(half));
    CheckMallocCPU(A_dequant_h, __LINE__);
    DeQuantMatrix_DualTrit5_Warptile_To_FP16(A_dequant_h, A_packed_h, M_GLOBAL, K_GLOBAL, A_Scale_h);
    
    // Device Memory for Weights
    uint16_t* A_packed;
    half*     A_Scale;
    half*     A_16bit;
    cudaMalloc(reinterpret_cast<void**>(&A_packed), packed_size_bytes);
    CheckMallocCUDA(A_packed, __LINE__);
    cudaMalloc(reinterpret_cast<void**>(&A_Scale), M_GLOBAL * sizeof(half));
    CheckMallocCUDA(A_Scale, __LINE__);
    cudaMalloc(reinterpret_cast<void**>(&A_16bit), M_GLOBAL * K_GLOBAL * sizeof(half));
    CheckMallocCUDA(A_16bit, __LINE__);
    
    // Memory Copy from CPU to GPU
    cudaMemcpy(A_packed, A_packed_h, packed_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A_Scale, A_Scale_h, M_GLOBAL * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(A_16bit, A_dequant_h, M_GLOBAL * K_GLOBAL * sizeof(half), cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // B Matrix: Activations (column-major)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    half* B_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    CheckMallocCPU(B_h, __LINE__);
    for (size_t i = 0; i < N_GLOBAL * K_GLOBAL; i++) {
        B_h[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
    }
    
    // Device memory
    half* B = NULL;
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    CheckMallocCUDA(B, __LINE__);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CUDA Events for Timing
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    checkLastCudaError(__LINE__);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // cuBLAS Reference GEMM
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    half* D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
    CheckMallocCUDA(D_cublas, __LINE__);
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // Tensor core enabled
    cudaDeviceSynchronize();
    
    int m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    
    // Warm-up
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     m, n, k,
                                     &alpha,
                                     A_16bit, CUDA_R_16F, k,
                                     B, CUDA_R_16F, k,
                                     &beta,
                                     D_cublas, CUDA_R_16F, m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     m, n, k,
                                     &alpha,
                                     A_16bit, CUDA_R_16F, k,
                                     B, CUDA_R_16F, k,
                                     &beta,
                                     D_cublas, CUDA_R_16F, m,
                                     CUDA_R_32F,
                                     CuBlasALG);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);
    milliseconds_cublas = milliseconds_cublas / BENCHMARK_ITERATION;
    float tflops_cublas = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas / 1000.)) / 1e12;
    
    // Copy result back to host
    half* D_cublas_h = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    CheckMallocCPU(D_cublas_h, __LINE__);
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);
    cudaFree(D_cublas);
    checkLastCudaError(__LINE__);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DualTrits GEMM Kernel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    half* D_dualtrit = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_dualtrit), sizeof(half) * M_GLOBAL * N_GLOBAL);
    CheckMallocCUDA(D_dualtrit, __LINE__);
    cudaMemset(D_dualtrit, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    
    int Split_K = SPLIT_K;
    float* Reduction_Workspace = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(float) * M_GLOBAL * N_GLOBAL * Split_K);
    CheckMallocCUDA(Reduction_Workspace, __LINE__);
    
    // Warm-up
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        dualtrits_linear_kernel(
            0,  // stream
            (uint4*)A_packed, A_Scale,
            B,
            D_dualtrit,
            M_GLOBAL, N_GLOBAL, K_GLOBAL,
            Reduction_Workspace,
            Split_K);
    }
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        dualtrits_linear_kernel(
            0,  // stream
            (uint4*)A_packed, A_Scale,
            B,
            D_dualtrit,
            M_GLOBAL, N_GLOBAL, K_GLOBAL,
            Reduction_Workspace,
            Split_K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    
    float milliseconds_dualtrit = 0.0f;
    cudaEventElapsedTime(&milliseconds_dualtrit, start, stop);
    milliseconds_dualtrit = milliseconds_dualtrit / BENCHMARK_ITERATION;
    float tflops_dualtrit = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_dualtrit / 1000.)) / 1e12;
    
    // Copy result back to host
    half* D_dualtrit_h = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    CheckMallocCPU(D_dualtrit_h, __LINE__);
    cudaMemcpy(D_dualtrit_h, D_dualtrit, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);
    cudaFree(D_dualtrit);
    cudaFree(Reduction_Workspace);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compare Results
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double totalRelativeError = ComputeTotalError(D_cublas_h, D_dualtrit_h, M_GLOBAL, N_GLOBAL);
    
    printf("************************************* ");
    printf("[DualTrits 4-bit] M: %zu N: %zu K: %zu SplitK: %d", M_GLOBAL, N_GLOBAL, K_GLOBAL, SPLIT_K);
    printf(" ************************************\n");
    PrintPerformance("cuBLAS", milliseconds_cublas, tflops_cublas, 0.0);
    PrintPerformance("DualTrits", milliseconds_dualtrit, tflops_dualtrit, totalRelativeError);
    
    // Print speedup
    float speedup = milliseconds_cublas / milliseconds_dualtrit;
    printf("Speedup vs cuBLAS: %.2fx\n", speedup);
    
    // Memory savings calculation
    size_t fp16_size = M_GLOBAL * K_GLOBAL * sizeof(half);
    size_t dualtrit_size = packed_size_bytes;
    float compression_ratio = (float)fp16_size / dualtrit_size;
    printf("Memory: FP16=%zu bytes, DualTrits=%zu bytes, Compression=%.2fx\n", 
           fp16_size, dualtrit_size, compression_ratio);
    
    // Cleanup
    free(A_16bit_h);
    free(A_Scale_h);
    free(A_packed_h);
    free(A_dequant_h);
    free(B_h);
    free(D_cublas_h);
    free(D_dualtrit_h);
    cudaFree(A_packed);
    cudaFree(A_Scale);
    cudaFree(A_16bit);
    cudaFree(B);
    cublasDestroy(handle);
    
    return 0;
}
