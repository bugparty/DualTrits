#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "common/DualTrits.hpp"
#include "cpu/dual_trits_pack.hpp"
#include "dual_trits_pack.cuh"
#include "cuda/kernels/pack_kernels.cuh"

// Random number generator for creating test data
static std::random_device rd;
static std::mt19937 gen([]{ std::random_device r; return r(); }());
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
    
    // Warmup
    pack_dual_trits_batch_cuda_device<COUNT, std::uint16_t>(d_input, d_output, N, nullptr);
    
    for (auto _ : state) {
        float ms = 0.f;
        pack_dual_trits_batch_cuda_device<COUNT, std::uint16_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0); // Convert to seconds
    }
    
    // Set performance metrics
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint16_t)));
    
    // Cleanup
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
        h_input[i] = pack_dual_trits<COUNT, std::uint64_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    // Allocate device memory
    std::uint16_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint16_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    
    // Warmup
    unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint16_t>(d_input, d_output, N, nullptr);
    
    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint16_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint16_t) + COUNT * sizeof(DualTrits)));
    
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
    
    pack_dual_trits_batch_cuda_device<COUNT, std::uint32_t>(d_input, d_output, N, nullptr);
    
    for (auto _ : state) {
        float ms = 0.f;
        pack_dual_trits_batch_cuda_device<COUNT, std::uint32_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint32_t)));
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
        h_input[i] = pack_dual_trits<COUNT, std::uint32_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint32_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint32_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint32_t>(d_input, d_output, N, nullptr);
    
    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint32_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint32_t) + COUNT * sizeof(DualTrits)));
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
    
    pack_dual_trits_batch_cuda_device<COUNT, std::uint64_t>(d_input, d_output, N, nullptr);
    
    for (auto _ : state) {
        float ms = 0.f;
        pack_dual_trits_batch_cuda_device<COUNT, std::uint64_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint64_t)));
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
        h_input[i] = pack_dual_trits<COUNT, std::uint64_t>(&h_temp[i * COUNT]);
    }
    std::vector<DualTrits> h_output(N * COUNT);
    
    std::uint64_t* d_input{};
    DualTrits* d_output{};
    cudaMalloc(&d_input, N * sizeof(std::uint64_t));
    cudaMalloc(&d_output, N * COUNT * sizeof(DualTrits));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

    unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint64_t>(d_input, d_output, N, nullptr);
    
    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint64_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (sizeof(std::uint64_t) + COUNT * sizeof(DualTrits)));
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
    
    // Warmup
    pack_dual_trits_batch_cuda_device<COUNT, std::uint16_t>(d_input, d_packed, N, nullptr);
    unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint16_t>(d_packed, d_output, N, nullptr);
    
    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        pack_dual_trits_batch_cuda_device<COUNT, std::uint16_t>(d_input, d_packed, N, nullptr);
        unpack_dual_trits_stride_batch_cuda_device<COUNT, std::uint16_t>(d_packed, d_output, N, nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) * 2 + sizeof(std::uint16_t)));
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
    
    // Warmup
    pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        pack_kernel<COUNT, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * 
                           (COUNT * sizeof(DualTrits) + sizeof(std::uint16_t)));
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
