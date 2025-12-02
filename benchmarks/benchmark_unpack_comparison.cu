#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "common/DualTrits.hpp"
#include "dual_trits_pack.cuh"
#include "cuda/kernels/pack_kernels.cuh"

// Random number generator for creating test data
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<> dis(0, 2);

// Helper function to generate random DualTrits
static DualTrits randomDualTrits() {
    return DualTrits(dis(gen), static_cast<DualTrits::wide_t>(dis(gen)));
}

// Forward declaration of standard unpack
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits_batch_cuda_standard(UInt const* h_input, DualTrits* h_output, int n);

// ============================================================================
// Standard Unpack Benchmarks (baseline)
// ============================================================================

static void BM_Unpack5_Standard(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 5;
    
    // Prepare valid input data
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    std::vector<std::uint16_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits_cuda<COUNT, std::uint16_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    // Allocate device memory
    std::uint16_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint16_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    unpack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        unpack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint16_t) + COUNT * sizeof(DualTrits)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

static void BM_Unpack10_Standard(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 10;
    
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    std::vector<std::uint32_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits_cuda<COUNT, std::uint32_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint32_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint32_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    unpack_kernel<COUNT, std::uint32_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        unpack_kernel<COUNT, std::uint32_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint32_t) + COUNT * sizeof(DualTrits)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

static void BM_Unpack20_Standard(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 20;
    
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    std::vector<std::uint64_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits_cuda<COUNT, std::uint64_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint64_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint64_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    unpack_kernel<COUNT, std::uint64_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        unpack_kernel<COUNT, std::uint64_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint64_t) + COUNT * sizeof(DualTrits)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// Warp-Cooperative Unpack Benchmarks (optimized)
// ============================================================================

static void BM_Unpack5_Warp(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 5;
    
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    std::vector<std::uint16_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits_cuda<COUNT, std::uint16_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint16_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint16_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    
    dim3 block(32);
    dim3 grid(N);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    unpack_kernel_warp<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        unpack_kernel_warp<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint16_t) + COUNT * sizeof(DualTrits)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

static void BM_Unpack10_Warp(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 10;
    
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    std::vector<std::uint32_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits_cuda<COUNT, std::uint32_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint32_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint32_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    
    dim3 block(32);
    dim3 grid(N);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    unpack_kernel_warp<COUNT, std::uint32_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        unpack_kernel_warp<COUNT, std::uint32_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint32_t) + COUNT * sizeof(DualTrits)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

static void BM_Unpack20_Warp(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 20;
    
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    std::vector<std::uint64_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits_cuda<COUNT, std::uint64_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint64_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint64_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    
    dim3 block(32);
    dim3 grid(N);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    unpack_kernel_warp<COUNT, std::uint64_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        unpack_kernel_warp<COUNT, std::uint64_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint64_t) + COUNT * sizeof(DualTrits)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// Register Benchmarks
// ============================================================================

// Standard version (baseline)
BENCHMARK(BM_Unpack5_Standard)
    ->UseManualTime()
    ->Arg(1<<16)   // 64K
    ->Arg(1<<18)   // 256K
    ->Arg(1<<20)   // 1M
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Unpack10_Standard)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Unpack20_Standard)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// Warp-cooperative version (optimized)
BENCHMARK(BM_Unpack5_Warp)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Unpack10_Warp)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Unpack20_Warp)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
