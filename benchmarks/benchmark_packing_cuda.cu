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

// ============================================================================
// Pack5 Kernel Benchmark (uint16_t)
// ============================================================================
static void BM_Pack5_CUDA(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 5;
    
    // Allocate and initialize host memory
    std::vector<DualTrits> h_input(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_input[i] = randomDualTrits();
    }
    std::vector<std::uint16_t> h_output(N);
    
    // Allocate device memory
    DualTrits* d_input{};
    std::uint16_t* d_output{};
    cudaMalloc(&d_input, N * COUNT * sizeof(DualTrits));
    cudaMalloc(&d_output, N * sizeof(std::uint16_t));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * COUNT * sizeof(DualTrits), cudaMemcpyHostToDevice);
    
    // Setup grid and block dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        // Reset if needed
        state.ResumeTiming();
        
        cudaEventRecord(start);
        pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0); // Convert to seconds
    }
    
    // Set performance metrics
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint16_t)));
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK(BM_Pack5_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)   // 64K elements
    ->Arg(1<<18)   // 256K elements
    ->Arg(1<<20)   // 1M elements
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack5 Kernel Benchmark (uint16_t)
// ============================================================================
static void BM_Unpack5_CUDA(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 5;
    
    // Allocate and initialize host memory
    std::vector<DualTrits> h_temp(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_temp[i] = randomDualTrits();
    }
    
    // Pack data first to get valid input
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
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    
    // Setup grid and block dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Setup CUDA events for timing
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

BENCHMARK(BM_Unpack5_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Pack10 Kernel Benchmark (uint32_t)
// ============================================================================
static void BM_Pack10_CUDA(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 10;
    
    std::vector<DualTrits> h_input(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_input[i] = randomDualTrits();
    }
    std::vector<std::uint32_t> h_output(N);
    
    DualTrits* d_input{};
    std::uint32_t* d_output{};
    cudaMalloc(&d_input, N * COUNT * sizeof(DualTrits));
    cudaMalloc(&d_output, N * sizeof(std::uint32_t));
    
    cudaMemcpy(d_input, h_input.data(), N * COUNT * sizeof(DualTrits), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    pack_kernel<COUNT, std::uint32_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        pack_kernel<COUNT, std::uint32_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint32_t)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK(BM_Pack10_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack10 Kernel Benchmark (uint32_t)
// ============================================================================
static void BM_Unpack10_CUDA(benchmark::State& state) {
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

BENCHMARK(BM_Unpack10_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Pack20 Kernel Benchmark (uint64_t)
// ============================================================================
static void BM_Pack20_CUDA(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 20;
    
    std::vector<DualTrits> h_input(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_input[i] = randomDualTrits();
    }
    std::vector<std::uint64_t> h_output(N);
    
    DualTrits* d_input{};
    std::uint64_t* d_output{};
    cudaMalloc(&d_input, N * COUNT * sizeof(DualTrits));
    cudaMalloc(&d_output, N * sizeof(std::uint64_t));
    
    cudaMemcpy(d_input, h_input.data(), N * COUNT * sizeof(DualTrits), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    pack_kernel<COUNT, std::uint64_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        pack_kernel<COUNT, std::uint64_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint64_t)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK(BM_Pack20_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack20 Kernel Benchmark (uint64_t)
// ============================================================================
static void BM_Unpack20_CUDA(benchmark::State& state) {
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

BENCHMARK(BM_Unpack20_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Round-trip Benchmark (Pack + Unpack) for Pack5
// ============================================================================
static void BM_RoundTrip5_CUDA(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int COUNT = 5;
    
    std::vector<DualTrits> h_input(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_input[i] = randomDualTrits();
    }
    
    DualTrits* d_input{};
    std::uint16_t* d_packed{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * COUNT * sizeof(DualTrits));
    cudaMalloc(&d_packed, N * sizeof(std::uint16_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * COUNT * sizeof(DualTrits), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_packed, N);
    unpack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_packed, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_packed, N);
        unpack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_packed, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) * 2 + sizeof(std::uint16_t)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_packed);
    cudaFree(d_output);
}

BENCHMARK(BM_RoundTrip5_CUDA)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Varying Block Size Benchmark for Pack5
// ============================================================================
static void BM_Pack5_VaryBlockSize(benchmark::State& state) {
    const int N = 1 << 20; // 1M elements
    const int blockSize = static_cast<int>(state.range(0));
    constexpr int COUNT = 5;
    
    std::vector<DualTrits> h_input(N * COUNT);
    for (int i = 0; i < N * COUNT; ++i) {
        h_input[i] = randomDualTrits();
    }
    
    DualTrits* d_input{};
    std::uint16_t* d_output{};
    cudaMalloc(&d_input, N * COUNT * sizeof(DualTrits));
    cudaMalloc(&d_output, N * sizeof(std::uint16_t));
    
    cudaMemcpy(d_input, h_input.data(), N * COUNT * sizeof(DualTrits), cudaMemcpyHostToDevice);
    
    dim3 block(blockSize);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
        
        cudaEventRecord(start);
        pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint16_t)));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

BENCHMARK(BM_Pack5_VaryBlockSize)
    ->UseManualTime()
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
