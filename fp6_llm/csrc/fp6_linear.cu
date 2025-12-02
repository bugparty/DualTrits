#include "include/kernel_matmul.cuh"
#include "include/kernel_reduction.cuh"
#include "include/kernel_matmul_dualtrits.cuh"
#include "utils/weight_prepacking.h"
#include "utils/weight_dequant.h"
#include "utils/weight_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>  // for std::max

template<typename TilingConfig, typename OutputDataType, int EXPONENT, int MANTISSA>
static void Kernel_Ex(cudaStream_t    stream,
                      const uint4     *Weight,
                      const half      *Scales,
                      const half      *B,
                      OutputDataType  *C,
                      const size_t    M_Global,
                      const size_t    N_Global,
                      const size_t    K_Global, 
                      int             Split_K) 
{
    #ifdef DEBUG_MODE
        printf("\n");
        printf("Launcher.cu->Kernel_Ex():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d\n", M_Global, N_Global, K_Global, Split_K);
        printf("TILE_M: %d, TILE_K: %d, TILE_N: %d\n", TilingConfig::TILE_M, TilingConfig::TILE_K, TilingConfig::TILE_N);
    #endif
    static size_t SHMEM_SZ = max(TilingConfig::SMEM_SIZE_B_TILE+SMEM_SIZE_PER_TB_A_TILE, TilingConfig::SMEM_SIZE_C_TILE);
    cudaFuncSetAttribute(QUANT_GEMM_Kernel<TilingConfig, OutputDataType, EXPONENT, MANTISSA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    size_t  dimN = (N_Global-1) / TilingConfig::TILE_N + 1;
    size_t  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3    GridDim(dimN, dimM, 1);
    dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
                GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    QUANT_GEMM_Kernel<TilingConfig, OutputDataType, EXPONENT, MANTISSA><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
}

template<int EXPONENT, int MANTISSA>
cudaError_t fpx_linear_kernel(cudaStream_t    stream,
                              const uint4     *Weight,
                              const half      *Scales,
                              const half      *B,
                              half            *C,
                              const size_t    M_Global,
                              const size_t    N_Global,
                              const size_t    K_Global, 
                              float           *Reduction_Workspace,  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
                              int             Split_K)
{
    assert(M_Global % 256 == 0);
    assert(K_Global % 64 == 0);
    assert(N_Global>0);

    // Work around to support more N shapes:
    size_t N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;

    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, half, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, half, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, half, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, half, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, half, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex<TilingConfig<4, 1, 8>, half, EXPONENT, MANTISSA>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
        }
    }
    else {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex<TilingConfig<4, 1, 8>, float, EXPONENT, MANTISSA>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    return cudaGetLastError();
}

cudaError_t fp6_linear_kernel(
    cudaStream_t    stream,
    const uint4     *Weight,
    const half      *Scales,
    const half      *B,
    half            *C,
    const size_t    M_Global,
    const size_t    N_Global,
    const size_t    K_Global, 
    float           *Reduction_Workspace,
    int             Split_K) {
    //
    return fpx_linear_kernel<3,2>( stream, Weight, Scales, B, C, M_Global, N_Global, K_Global,  Reduction_Workspace, Split_K);
}

cudaError_t fp_eXmY_linear_kernel(
    const int       EXPONENT,
    const int       MANTISSA,
    cudaStream_t    stream,
    const uint4     *Weight,
    const half      *Scales,
    const half      *B,
    half            *C,
    const size_t    M_Global,
    const size_t    N_Global,
    const size_t    K_Global, 
    float           *Reduction_Workspace,
    int             Split_K) {
    //
    if(EXPONENT==2 && MANTISSA==2)
        return fpx_linear_kernel<2,2>( stream, Weight, Scales, B, C, M_Global, N_Global, K_Global,  Reduction_Workspace, Split_K);
    if(EXPONENT==3 && MANTISSA==2)
        return fpx_linear_kernel<3,2>( stream, Weight, Scales, B, C, M_Global, N_Global, K_Global,  Reduction_Workspace, Split_K);
    printf("QuantLLM_API Error: Unsupported EXPONENT=%d, MANTISSA=%d!\n", EXPONENT, MANTISSA);
    exit(-1);
}

/*****************************************************************************
 * DualTrits GEMM Kernel Launcher
 *****************************************************************************/

template<typename TilingConfig, typename OutputDataType>
static void DualTrits_Kernel_Ex(
    cudaStream_t    stream,
    const uint4     *Weight,
    const half      *Scales,
    const half      *B,
    OutputDataType  *C,
    const size_t    M_Global,
    const size_t    N_Global,
    const size_t    K_Global, 
    int             Split_K
) {
    #ifdef DEBUG_MODE
        printf("\n");
        printf("DualTrits_Kernel_Ex():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d\n", M_Global, N_Global, K_Global, Split_K);
        printf("TILE_M: %d, TILE_K: %d, TILE_N: %d\n", TilingConfig::TILE_M, TilingConfig::TILE_K, TilingConfig::TILE_N);
    #endif
    
    // Shared memory size: A (DualTrits) + B tiles
    static size_t SHMEM_SZ = std::max<size_t>(
        TilingConfig::SMEM_SIZE_B_TILE + SMEM_SIZE_PER_TB_DT,
        TilingConfig::SMEM_SIZE_C_TILE
    );
    
    cudaFuncSetAttribute(
        DUALTRITS_GEMM_Kernel<TilingConfig, OutputDataType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SHMEM_SZ
    );
    
    size_t dimN = (N_Global - 1) / TilingConfig::TILE_N + 1;
    size_t dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
               GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    
    DUALTRITS_GEMM_Kernel<TilingConfig, OutputDataType>
        <<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K
        );
}

/*
 * DualTrits Linear Kernel Interface.
 * 
 * DualTrits encoding: 4-bit base-3 encoding representing 9 values.
 * Storage: 64 elements -> 16 uint16_t (13 valid packs + 3 padding)
 * Memory layout: Row-major, prepacked for coalesced access.
 * 
 * Parameters:
 *   Weight - Prepacked DualTrits weights [M, K/64 * 16 * sizeof(uint16_t)]
 *   Scales - Per-row quantization scales [M]
 *   B      - FP16 activations [K, N] (column-major)
 *   C      - FP16 output [M, N] (column-major)
 */
cudaError_t dualtrits_linear_kernel(
    cudaStream_t    stream,
    const uint4     *Weight,
    const half      *Scales,
    const half      *B,
    half            *C,
    const size_t    M_Global,
    const size_t    N_Global,
    const size_t    K_Global, 
    float           *Reduction_Workspace,
    int             Split_K
) {
    assert(M_Global % 256 == 0);
    assert(K_Global % 64 == 0);
    assert(N_Global > 0);
    
    // Determine optimal TILE_N based on N dimension
    size_t N_PowerOf2;
    if (N_Global > 0 && N_Global <= 8)       N_PowerOf2 = 8;
    else if (N_Global > 8 && N_Global <= 16) N_PowerOf2 = 16;
    else if (N_Global > 16 && N_Global <= 32) N_PowerOf2 = 32;
    else if (N_Global > 32 && N_Global <= 64) N_PowerOf2 = 64;
    else if (N_Global > 64 && N_Global <= 128) N_PowerOf2 = 128;
    else N_PowerOf2 = ((N_Global - 1) / 128 + 1) * 128;
    
    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 1>, half>(
                    stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
                break;
            case 16:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 2>, half>(
                    stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
                break;
            case 32:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 4>, half>(
                    stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
                break;
            case 64:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 8>, half>(
                    stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
                break;
            case 128:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 8>, half>(
                    stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
                break;
            default:
                if (N_PowerOf2 % 128 != 0) {
                    printf("DualTrits_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                    return cudaErrorUnknown;
                }
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 8>, half>(
                    stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
                break;
        }
    } else {
        switch (N_PowerOf2) {
            case 8:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 1>, float>(
                    stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
                break;
            case 16:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 2>, float>(
                    stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
                break;
            case 32:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 4>, float>(
                    stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
                break;
            case 64:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 8>, float>(
                    stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
                break;
            case 128:
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 8>, float>(
                    stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
                break;
            default:
                if (N_PowerOf2 % 128 != 0) {
                    printf("DualTrits_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                    return cudaErrorUnknown;
                }
                DualTrits_Kernel_Ex<TilingConfig<4, 1, 8>, float>(
                    stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
                break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    
    return cudaGetLastError();
}

#ifndef NO_PYTORCH
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h> // For CUDA stream management

/////////////////////////////////////////////////// Old Interface only Supporting FP6 /////////////////////////////////////////////////////////////////////
/*
Computes FP6-FP16 GEMM (PyTorch interface).

[Mathmatical Formula]
Standard definition of linear layer:    Out = In * trans(W), where In, Out, and W are stored in row-major.
After Equivalent transformation    :    trans(Out) = W * trans(In). Note that we do not perform "transpose" during runtime, we instead interpret the In/Out as column-major matrices when calling our CUDA kernel.

[Inputs]
  _in_feats:  tensor of shape [B, IC];                  // half 
  _weights:   int tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6 weights.
  _scales:    tensor of shape [OC];                     // half
  splitK:     spliting the MatMul problem along K dimension for higher GPU utilization, default 1.
[Outputs]
  _out_feats: tensor of shape [B, OC];                  // half
*/
torch::Tensor fp6_linear_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _weights,
    torch::Tensor _scales,
    int           splitK=1)
{
    int num_in_feats      = _in_feats.size(0);
    int num_in_channels   = _in_feats.size(1);
    int num_out_channels  = _weights.size(0);
    assert( num_in_channels%64 == 0 );
    assert( (num_in_channels/16*3) == _weights.size(1) );    // Making sure the K dimension is matched.
    //
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    // Input Tensors
    auto weight = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());  // weights is [OC, IC] but in FP6.
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    // Output Tensors
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    options = torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)

    // Get the current device and stream
    int device_id = _in_feats.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_id).stream();

    fp6_linear_kernel(stream, // Using current stream.
                      weight,
                      scales,
                      in_feats,
                      out_feats,
                      M,
                      N,
                      K, 
                      Reduction_Workspace,  
                      splitK);

    return _out_feats;
}


/*
 * Weight prepacking (Pytorch interface).
 * [Input & Output]
 *  fp6_tensor: int tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6 weights.
 * [Output]
 *  packed_tensor: int tensor of shape [OC, IC // 16 * 3];
 */
torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor fp6_tensor)
{
    size_t OC = fp6_tensor.size(0);
    size_t IC = fp6_tensor.size(1);
    assert (IC%3==0);   
    IC = IC*16/3;
    assert( (OC%256==0) && (IC%64==0) );
    auto packed_tensor = torch::empty_like(fp6_tensor);
    auto packed_tensor_ptr = reinterpret_cast<int*>(packed_tensor.data_ptr<int>());
    auto fp6_tensor_ptr = reinterpret_cast<int*>(fp6_tensor.data_ptr<int>());
    weight_matrix_prepacking(packed_tensor_ptr, fp6_tensor_ptr, OC, IC);
    return packed_tensor;
}

/*
 * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
 * A useful tool to construct input matrices for the FP16 GEMM baseline.
 * [Input]
 *  fp6_tensor:  int  tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6  weights.
 *  fp16_scale:  half tensor of shape [OC];                 // for row-wise quantization.
 * [Output]
 *  fp16_tensor: half tensor of shape [OC, IC].     
 */
torch::Tensor weight_matrix_dequant_cpu(torch::Tensor fp6_tensor, torch::Tensor fp16_scale) 
{
    int OC = fp6_tensor.size(0);
    assert(fp6_tensor.size(1) % 3 == 0);
    int IC = fp6_tensor.size(1) / 3 * 16;
    assert(fp16_scale.size(0)==OC);
    //
    auto fp6_tensor_ptr = reinterpret_cast<int*>(fp6_tensor.data_ptr<int>());
    auto fp16_scale_ptr = reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>());
    //
    auto options = torch::TensorOptions().dtype(fp16_scale.dtype()).device(fp16_scale.device());
    at::Tensor fp16_tensor = torch::empty({OC, IC}, options);
    auto fp16_tensor_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    //
    DeQuantMatrix_FP6_To_FP16(fp16_tensor_ptr, (unsigned char*)fp6_tensor_ptr, OC, IC, fp16_scale_ptr);
    //
    return fp16_tensor;
}

/////////////////////////////////////////////////// New Interface Supporting FPx /////////////////////////////////////////////////////////////////////
/*
Computes FPx-FP16 GEMM (PyTorch interface).

[Mathmatical Formula]
Standard definition of linear layer:    Out = In * trans(W), where In, Out, and W are stored in row-major.
After Equivalent transformation    :    trans(Out) = W * trans(In). Note that we do not perform "transpose" during runtime, we instead interpret the In/Out as column-major matrices when calling our CUDA kernel.

[Inputs]
  _in_feats:  tensor of shape [B, IC];                  // half 
  _weights:   int tensor of shape [OC, IC // 32 * x];   // x INT32 words contains 32 FPx weights.
  _scales:    tensor of shape [OC];                     // half
  splitK:     spliting the MatMul problem along K dimension for higher GPU utilization, default 1.
[Outputs]
  _out_feats: tensor of shape [B, OC];                  // half
*/
torch::Tensor fp_eXmY_linear_forward_cuda(
    int             EXPONENT,
    int             MANTISSA,
    torch::Tensor   _in_feats,
    torch::Tensor   _weights,
    torch::Tensor   _scales,
    int             splitK=1)
{
    int num_in_feats      = _in_feats.size(0);
    int num_in_channels   = _in_feats.size(1);
    int num_out_channels  = _weights.size(0);
    assert( num_in_channels%64 == 0 );
    assert( (num_in_channels/32*(1+EXPONENT+MANTISSA)) == _weights.size(1) );    // Making sure the K dimension is matched.
    //
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    // Input Tensors
    auto weight = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());  // weights is [OC, IC] but in FP6.
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    // Output Tensors
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    options = torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)

    // Get the current device and stream
    int device_id = _in_feats.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_id).stream();

    //
    fp_eXmY_linear_kernel(
        EXPONENT,
        MANTISSA,
        stream, // Using torch's current stream here.
        weight,
        scales,
        in_feats,
        out_feats,
        M,
        N,
        K, 
        Reduction_Workspace,  
        splitK);
    return _out_feats;
}


/*
 * Weight prepacking (Pytorch interface).
 * [Input & Output]
 *  fpx_tensor: int tensor of shape [OC, IC // 32 * x];
 * [Output]
 *  packed_tensor: int tensor of shape [OC, IC // 32 * x];
 */
torch::Tensor weight_matrix_prepacking_fp_eXmY_cpu(
    int EXPONENT,
    int MANTISSA,
    torch::Tensor fpx_tensor)
{
    int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    //
    size_t OC = fpx_tensor.size(0);
    size_t IC = fpx_tensor.size(1);
    assert (IC%BIT_WIDTH==0);   
    IC = IC*32/BIT_WIDTH;
    assert( (OC%256==0) && (IC%64==0) );
    auto packed_tensor = torch::empty_like(fpx_tensor);
    auto packed_tensor_ptr = reinterpret_cast<int*>(packed_tensor.data_ptr<int>());
    auto fpx_tensor_ptr = reinterpret_cast<int*>(fpx_tensor.data_ptr<int>());
    //
    weight_matrix_prepacking_fp_eXmY(EXPONENT, MANTISSA, packed_tensor_ptr, fpx_tensor_ptr, OC, IC);
    return packed_tensor;
}

/*
 * Dequant a FPx matrix to a equivalent FP16 matrix using CPUs.
 * A useful tool to construct input matrices for the FP16 GEMM baseline.
 * [Input]
 *  fpx_tensor:  int  tensor of shape [OC, IC // 32 * x];   //
 *  fp16_scale:  half tensor of shape [OC];                 // for row-wise quantization.
 * [Output]
 *  fp16_tensor: half tensor of shape [OC, IC].     
 */
torch::Tensor weight_matrix_dequant_fp_eXmY_cpu(
    int EXPONENT,
    int MANTISSA,
    torch::Tensor fpx_tensor,
    torch::Tensor fp16_scale) 
{
    int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    //
    int OC = fpx_tensor.size(0);
    assert(fpx_tensor.size(1) % BIT_WIDTH == 0);
    int IC = fpx_tensor.size(1) / BIT_WIDTH * 32;
    assert(fp16_scale.size(0)==OC);
    //
    auto fpx_tensor_ptr = reinterpret_cast<int*>(fpx_tensor.data_ptr<int>());
    auto fp16_scale_ptr = reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>());
    //
    auto options = torch::TensorOptions().dtype(fp16_scale.dtype()).device(fp16_scale.device());
    at::Tensor fp16_tensor = torch::empty({OC, IC}, options);
    auto fp16_tensor_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    //
    dequant_matrix_fp_eXmY_to_fp16(EXPONENT, MANTISSA, fp16_tensor_ptr, (unsigned char*)fpx_tensor_ptr, OC, IC, fp16_scale_ptr);
    //
    return fp16_tensor;
}

/*****************************************************************************
 * DualTrits PyTorch Interface
 *****************************************************************************/

/*
 * DualTrits Linear Forward (PyTorch interface).
 * 
 * [Mathematical Formula]
 * Standard definition of linear layer: Out = In * trans(W)
 * After equivalent transformation:     trans(Out) = W * trans(In)
 * 
 * [Inputs]
 *   _in_feats:  tensor of shape [B, IC];                    // half
 *   _weights:   int tensor of shape [OC, IC // 64 * 8];     // 16 uint16_t = 8 int32 per 64 DualTrits
 *   _scales:    tensor of shape [OC];                       // half, per-row scales
 *   splitK:     splitting MatMul along K dimension, default 1
 * [Outputs]
 *   _out_feats: tensor of shape [B, OC];                    // half
 */
torch::Tensor dualtrits_linear_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _weights,
    torch::Tensor _scales,
    int           splitK = 1)
{
    int num_in_feats     = _in_feats.size(0);
    int num_in_channels  = _in_feats.size(1);
    int num_out_channels = _weights.size(0);
    
    assert(num_in_channels % 64 == 0);
    // Verify K dimension: IC / 64 * 8 int32 (16 uint16_t per 64 elements)
    assert((num_in_channels / 64 * 8) == _weights.size(1));
    
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    
    // Input Tensors
    auto weight   = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    
    // Output Tensors
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    
    // Workspace for SplitK reduction
    options = torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());
    
    // Get current CUDA stream
    int device_id = _in_feats.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_id).stream();
    
    dualtrits_linear_kernel(
        stream,
        weight,
        scales,
        in_feats,
        out_feats,
        M,
        N,
        K,
        Reduction_Workspace,
        splitK
    );
    
    return _out_feats;
}

/*
 * DualTrits Weight Quantization for GPU (PyTorch interface).
 * 
 * Quantizes FP16 weights to DualTrits format with per-row scaling.
 * Uses padded layout (16 uint16_t per 64 elements) for coalesced GPU access.
 * 
 * [Input]
 *   fp16_tensor: half tensor of shape [OC, IC]
 * [Output]
 *   Tuple of:
 *     - packed_tensor: int tensor of shape [OC, IC // 64 * 8]  (16 uint16_t = 8 int32 per 64 elements)
 *     - scales: half tensor of shape [OC]
 */
std::tuple<torch::Tensor, torch::Tensor> weight_matrix_quant_dualtrits_cpu(
    torch::Tensor fp16_tensor)
{
    size_t OC = fp16_tensor.size(0);
    size_t IC = fp16_tensor.size(1);
    
    assert(OC % 256 == 0);
    assert(IC % 64 == 0);
    
    // Packed output: 16 uint16_t per 64 elements = 32 bytes per 64 elements = 8 int32
    size_t packed_IC = IC / 64 * 8;  // Number of int32 per row
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(fp16_tensor.device());
    at::Tensor packed_tensor = torch::empty({(long)OC, (long)packed_IC}, options_int);
    
    auto options_half = torch::TensorOptions().dtype(torch::kFloat16).device(fp16_tensor.device());
    at::Tensor scales_tensor = torch::empty({(long)OC}, options_half);
    
    auto fp16_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    auto packed_ptr = reinterpret_cast<std::uint16_t*>(packed_tensor.data_ptr<int>());
    auto scales_ptr = reinterpret_cast<half*>(scales_tensor.data_ptr<at::Half>());
    
    // Use GPU-optimized packing with padding
    weight_prepacking_fp16_to_dual_trits_gpu(fp16_ptr, packed_ptr, OC, IC, scales_ptr);
    
    return std::make_tuple(packed_tensor, scales_tensor);
}

/*
 * DualTrits Weight Dequantization (PyTorch interface).
 * 
 * Dequantizes DualTrits weights back to FP16 for verification.
 * Supports padded format (16 uint16_t per 64 elements).
 * 
 * [Input]
 *   packed_tensor: int tensor of shape [OC, IC // 64 * 8]
 *   scales:        half tensor of shape [OC]
 * [Output]
 *   fp16_tensor:   half tensor of shape [OC, IC]
 */
torch::Tensor weight_matrix_dequant_dualtrits_cpu(
    torch::Tensor packed_tensor,
    torch::Tensor scales_tensor)
{
    size_t OC = packed_tensor.size(0);
    size_t packed_IC = packed_tensor.size(1);
    
    // Recover IC: packed_IC = IC / 64 * 8 => IC = packed_IC * 8
    size_t IC = packed_IC * 8;
    
    assert(scales_tensor.size(0) == OC);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(packed_tensor.device());
    at::Tensor fp16_tensor = torch::empty({(long)OC, (long)IC}, options);
    
    auto packed_ptr = reinterpret_cast<std::uint16_t*>(packed_tensor.data_ptr<int>());
    auto scales_ptr = reinterpret_cast<half*>(scales_tensor.data_ptr<at::Half>());
    auto fp16_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    
    // Dequantize padded format
    DeQuantMatrix_DualTrit5_Padded_To_FP16(fp16_ptr, packed_ptr, OC, IC, scales_ptr);
    
    return fp16_tensor;
}
#endif