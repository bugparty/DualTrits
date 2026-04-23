#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <random>
#include <vector>
#include "common/DualTrits.hpp"
#include "cpu/dual_trits_pack.hpp"
#include "hip/dual_trits_pack.h"
#include "hip/kernels/pack_kernels.h"

// Random number generator for creating test data
static std::random_device rd;
static std::mt19937 gen([]{ std::random_device r; return r(); }());
static std::uniform_int_distribution<> dis(0, 2);

static DualTrits randomDualTrits() {
    return DualTrits(dis(gen), static_cast<DualTrits::wide_t>(dis(gen)));
}

// ============================================================================
// Pack5 Kernel Benchmark (uint16_t)
// ============================================================================
static void BM_Pack5_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 5;

    std::vector<DualTrits> h_input(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_input[i] = randomDualTrits();
    }
    std::vector<std::uint16_t> h_output(N);

    DualTrits* d_input{};
    std::uint16_t* d_output{};
    hipMalloc(&d_input, N * TRITS_PER_PACK * sizeof(DualTrits));
    hipMalloc(&d_output, N * sizeof(std::uint16_t));

    hipMemcpy(d_input, h_input.data(), N * TRITS_PER_PACK * sizeof(DualTrits), hipMemcpyHostToDevice);

    // Warmup
    pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (TRITS_PER_PACK * sizeof(DualTrits) + sizeof(std::uint16_t)));

    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Pack5_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack5 Kernel Benchmark (uint16_t)
// ============================================================================
static void BM_Unpack5_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 5;

    std::vector<DualTrits> h_temp(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_temp[i] = randomDualTrits();
    }

    std::vector<std::uint16_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits<TRITS_PER_PACK, std::uint16_t>(&h_temp[i * TRITS_PER_PACK]);
    }
    std::vector<DualTrits> h_output(N * TRITS_PER_PACK);

    std::uint16_t* d_input{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * sizeof(std::uint16_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * sizeof(std::uint16_t), hipMemcpyHostToDevice);

    // Warmup
    unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (sizeof(std::uint16_t) + TRITS_PER_PACK * sizeof(DualTrits)));

    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Unpack5_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack5 Stride Kernel Benchmark (uint16_t)
// ============================================================================
static void BM_Unpack5_Stride_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 5;

    std::vector<DualTrits> h_temp(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_temp[i] = randomDualTrits();
    }

    std::vector<std::uint16_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits<TRITS_PER_PACK, std::uint16_t>(&h_temp[i * TRITS_PER_PACK]);
    }
    std::vector<DualTrits> h_output(N * TRITS_PER_PACK);

    std::uint16_t* d_input{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * sizeof(std::uint16_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * sizeof(std::uint16_t), hipMemcpyHostToDevice);

    // Warmup
    unpack_dual_trits_stride_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_stride_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (sizeof(std::uint16_t) + TRITS_PER_PACK * sizeof(DualTrits)));

    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Unpack5_Stride_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Pack10 Kernel Benchmark (uint32_t)
// ============================================================================
static void BM_Pack10_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 10;

    std::vector<DualTrits> h_input(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_input[i] = randomDualTrits();
    }
    std::vector<std::uint32_t> h_output(N);

    DualTrits* d_input{};
    std::uint32_t* d_output{};
    hipMalloc(&d_input, N * TRITS_PER_PACK * sizeof(DualTrits));
    hipMalloc(&d_output, N * sizeof(std::uint32_t));

    hipMemcpy(d_input, h_input.data(), N * TRITS_PER_PACK * sizeof(DualTrits), hipMemcpyHostToDevice);

    pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint32_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint32_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (TRITS_PER_PACK * sizeof(DualTrits) + sizeof(std::uint32_t)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Pack10_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack10 Kernel Benchmark (uint32_t)
// ============================================================================
static void BM_Unpack10_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 10;

    std::vector<DualTrits> h_temp(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_temp[i] = randomDualTrits();
    }

    std::vector<std::uint32_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits<TRITS_PER_PACK, std::uint32_t>(&h_temp[i * TRITS_PER_PACK]);
    }
    std::vector<DualTrits> h_output(N * TRITS_PER_PACK);

    std::uint32_t* d_input{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * sizeof(std::uint32_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * sizeof(std::uint32_t), hipMemcpyHostToDevice);

    unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint32_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint32_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (sizeof(std::uint32_t) + TRITS_PER_PACK * sizeof(DualTrits)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Unpack10_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack10 Stride Kernel Benchmark (uint32_t)
// ============================================================================
static void BM_Unpack10_Stride_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 10;

    std::vector<DualTrits> h_temp(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_temp[i] = randomDualTrits();
    }

    std::vector<std::uint32_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits<TRITS_PER_PACK, std::uint32_t>(&h_temp[i * TRITS_PER_PACK]);
    }
    std::vector<DualTrits> h_output(N * TRITS_PER_PACK);

    std::uint32_t* d_input{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * sizeof(std::uint32_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * sizeof(std::uint32_t), hipMemcpyHostToDevice);

    unpack_dual_trits_stride_batch_hip_device<TRITS_PER_PACK, std::uint32_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_stride_batch_hip_device<TRITS_PER_PACK, std::uint32_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (sizeof(std::uint32_t) + TRITS_PER_PACK * sizeof(DualTrits)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Unpack10_Stride_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Pack20 Kernel Benchmark (uint64_t)
// ============================================================================
static void BM_Pack20_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 20;

    std::vector<DualTrits> h_input(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_input[i] = randomDualTrits();
    }
    std::vector<std::uint64_t> h_output(N);

    DualTrits* d_input{};
    std::uint64_t* d_output{};
    hipMalloc(&d_input, N * TRITS_PER_PACK * sizeof(DualTrits));
    hipMalloc(&d_output, N * sizeof(std::uint64_t));

    hipMemcpy(d_input, h_input.data(), N * TRITS_PER_PACK * sizeof(DualTrits), hipMemcpyHostToDevice);

    pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint64_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint64_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (TRITS_PER_PACK * sizeof(DualTrits) + sizeof(std::uint64_t)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Pack20_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack20 Stride Kernel Benchmark (uint64_t)
// ============================================================================
static void BM_Unpack20_Stride_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 20;

    std::vector<DualTrits> h_temp(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_temp[i] = randomDualTrits();
    }

    std::vector<std::uint64_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits<TRITS_PER_PACK, std::uint64_t>(&h_temp[i * TRITS_PER_PACK]);
    }
    std::vector<DualTrits> h_output(N * TRITS_PER_PACK);

    std::uint64_t* d_input{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * sizeof(std::uint64_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * sizeof(std::uint64_t), hipMemcpyHostToDevice);

    unpack_dual_trits_stride_batch_hip_device<TRITS_PER_PACK, std::uint64_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_stride_batch_hip_device<TRITS_PER_PACK, std::uint64_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (sizeof(std::uint64_t) + TRITS_PER_PACK * sizeof(DualTrits)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Unpack20_Stride_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Unpack20 Kernel Benchmark (uint64_t)
// ============================================================================
static void BM_Unpack20_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 20;

    std::vector<DualTrits> h_temp(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_temp[i] = randomDualTrits();
    }

    std::vector<std::uint64_t> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = pack_dual_trits<TRITS_PER_PACK, std::uint64_t>(&h_temp[i * TRITS_PER_PACK]);
    }
    std::vector<DualTrits> h_output(N * TRITS_PER_PACK);

    std::uint64_t* d_input{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * sizeof(std::uint64_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * sizeof(std::uint64_t), hipMemcpyHostToDevice);

    unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint64_t>(d_input, d_output, N, nullptr);

    for (auto _ : state) {
        float ms = 0.f;
        unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint64_t>(d_input, d_output, N, &ms);
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (sizeof(std::uint64_t) + TRITS_PER_PACK * sizeof(DualTrits)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Unpack20_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Round-trip Benchmark (Pack + Unpack) for Pack5
// ============================================================================
static void BM_RoundTrip5_HIP(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 5;

    std::vector<DualTrits> h_input(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_input[i] = randomDualTrits();
    }

    DualTrits* d_input{};
    std::uint16_t* d_packed{};
    DualTrits* d_output{};
    hipMalloc(&d_input, N * TRITS_PER_PACK * sizeof(DualTrits));
    hipMalloc(&d_packed, N * sizeof(std::uint16_t));
    hipMalloc(&d_output, N * TRITS_PER_PACK * sizeof(DualTrits));

    hipMemcpy(d_input, h_input.data(), N * TRITS_PER_PACK * sizeof(DualTrits), hipMemcpyHostToDevice);

    // Warmup
    pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_packed, N, nullptr);
    unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_packed, d_output, N, nullptr);

    for (auto _ : state) {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, 0);
        pack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_input, d_packed, N, nullptr);
        unpack_dual_trits_batch_hip_device<TRITS_PER_PACK, std::uint16_t>(d_packed, d_output, N, nullptr);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);

        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);

        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (TRITS_PER_PACK * sizeof(DualTrits) * 2 + sizeof(std::uint16_t)));
    hipFree(d_input);
    hipFree(d_packed);
    hipFree(d_output);
}

BENCHMARK(BM_RoundTrip5_HIP)
    ->UseManualTime()
    ->Arg(1<<16)
    ->Arg(1<<18)
    ->Arg(1<<20)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Varying Block Size Benchmark for Pack5
// ============================================================================
static void BM_Pack5_VaryBlockSize_HIP(benchmark::State& state) {
    const int N = 1 << 20; // 1M elements
    const int blockSize = static_cast<int>(state.range(0));
    constexpr int TRITS_PER_PACK = 5;

    std::vector<DualTrits> h_input(N * TRITS_PER_PACK);
    for (int i = 0; i < N * TRITS_PER_PACK; ++i) {
        h_input[i] = randomDualTrits();
    }

    DualTrits* d_input{};
    std::uint16_t* d_output{};
    hipMalloc(&d_input, N * TRITS_PER_PACK * sizeof(DualTrits));
    hipMalloc(&d_output, N * sizeof(std::uint16_t));

    hipMemcpy(d_input, h_input.data(), N * TRITS_PER_PACK * sizeof(DualTrits), hipMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((N + block.x - 1) / block.x);

    // Warmup
    pack_kernel_hip<TRITS_PER_PACK, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    for (auto _ : state) {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, 0);
        pack_kernel_hip<TRITS_PER_PACK, std::uint16_t><<<grid, block>>>(d_input, d_output, N);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);

        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);

        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N *
                           (TRITS_PER_PACK * sizeof(DualTrits) + sizeof(std::uint16_t)));
    hipFree(d_input);
    hipFree(d_output);
}

BENCHMARK(BM_Pack5_VaryBlockSize_HIP)
    ->UseManualTime()
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
