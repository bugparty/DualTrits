#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <array>
#include "common/DualTrits.hpp"
#include "cpu/dual_trits_pack.hpp"

// Random number generator for creating test data
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<> dis(0, 2);

// Helper function to generate random DualTrits
static DualTrits randomDualTrits() {
    return DualTrits(dis(gen), static_cast<DualTrits::wide_t>(dis(gen)));
}

// Benchmark: Pack 5 DualTrits into uint16_t (Batch version)
static void BM_Pack5_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::array<DualTrits,5>> inputs(N);
    // Initialize inputs with random data
    for (auto& a : inputs)
        for (int i = 0; i < 5; ++i)
            a[i] = randomDualTrits();

    for (auto _ : state) {
        uint64_t sink = 0;
        for (int i = 0; i < N; ++i) {
            auto packed = pack5(inputs[i].data());
            benchmark::DoNotOptimize(sink += static_cast<uint64_t>(packed));
        }
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,5>));
}
BENCHMARK(BM_Pack5_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Unpack 5 DualTrits from uint16_t (Batch version)
static void BM_Unpack5_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::uint16_t> packed_inputs(N);
    std::vector<std::array<DualTrits,5>> outputs(N);
    
    // Initialize packed inputs with random data
    for (int i = 0; i < N; ++i) {
        DualTrits arr[5];
        for (int j = 0; j < 5; ++j)
            arr[j] = randomDualTrits();
        packed_inputs[i] = pack5(arr);
    }

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            unpack5(packed_inputs[i], outputs[i].data());
        }
        benchmark::DoNotOptimize(outputs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,5>));
}
BENCHMARK(BM_Unpack5_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Pack 10 DualTrits into uint32_t (Batch version)
static void BM_Pack10_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::array<DualTrits,10>> inputs(N);
    // Initialize inputs with random data
    for (auto& a : inputs)
        for (int i = 0; i < 10; ++i)
            a[i] = randomDualTrits();

    for (auto _ : state) {
        uint64_t sink = 0;
        for (int i = 0; i < N; ++i) {
            auto packed = pack10(inputs[i].data());
            benchmark::DoNotOptimize(sink += static_cast<uint64_t>(packed));
        }
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,10>));
}
BENCHMARK(BM_Pack10_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Unpack 10 DualTrits from uint32_t (Batch version)
static void BM_Unpack10_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::uint32_t> packed_inputs(N);
    std::vector<std::array<DualTrits,10>> outputs(N);
    
    // Initialize packed inputs with random data
    for (int i = 0; i < N; ++i) {
        DualTrits arr[10];
        for (int j = 0; j < 10; ++j)
            arr[j] = randomDualTrits();
        packed_inputs[i] = pack10(arr);
    }

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            unpack10(packed_inputs[i], outputs[i].data());
        }
        benchmark::DoNotOptimize(outputs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,10>));
}
BENCHMARK(BM_Unpack10_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Pack 20 DualTrits into uint64_t (Batch version)
static void BM_Pack20_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::array<DualTrits,20>> inputs(N);
    // Initialize inputs with random data
    for (auto& a : inputs)
        for (int i = 0; i < 20; ++i)
            a[i] = randomDualTrits();

    for (auto _ : state) {
        uint64_t sink = 0;
        for (int i = 0; i < N; ++i) {
            auto packed = pack20(inputs[i].data());
            benchmark::DoNotOptimize(sink += static_cast<uint64_t>(packed));
        }
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,20>));
}
BENCHMARK(BM_Pack20_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Unpack 20 DualTrits from uint64_t (Batch version)
static void BM_Unpack20_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::uint64_t> packed_inputs(N);
    std::vector<std::array<DualTrits,20>> outputs(N);
    
    // Initialize packed inputs with random data
    for (int i = 0; i < N; ++i) {
        DualTrits arr[20];
        for (int j = 0; j < 20; ++j)
            arr[j] = randomDualTrits();
        packed_inputs[i] = pack20(arr);
    }

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            unpack20(packed_inputs[i], outputs[i].data());
        }
        benchmark::DoNotOptimize(outputs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,20>));
}
BENCHMARK(BM_Unpack20_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Round-trip pack and unpack for 5 DualTrits (Batch version)
static void BM_RoundTrip5_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::array<DualTrits,5>> inputs(N);
    std::vector<std::array<DualTrits,5>> outputs(N);
    
    // Initialize inputs with random data
    for (auto& a : inputs)
        for (int i = 0; i < 5; ++i)
            a[i] = randomDualTrits();

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            auto packed = pack5(inputs[i].data());
            unpack5(packed, outputs[i].data());
        }
        benchmark::DoNotOptimize(outputs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,5>) * 2);
}
BENCHMARK(BM_RoundTrip5_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Round-trip pack and unpack for 10 DualTrits (Batch version)
static void BM_RoundTrip10_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::array<DualTrits,10>> inputs(N);
    std::vector<std::array<DualTrits,10>> outputs(N);
    
    // Initialize inputs with random data
    for (auto& a : inputs)
        for (int i = 0; i < 10; ++i)
            a[i] = randomDualTrits();

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            auto packed = pack10(inputs[i].data());
            unpack10(packed, outputs[i].data());
        }
        benchmark::DoNotOptimize(outputs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,10>) * 2);
}
BENCHMARK(BM_RoundTrip10_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Benchmark: Round-trip pack and unpack for 20 DualTrits (Batch version)
static void BM_RoundTrip20_Batch(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<std::array<DualTrits,20>> inputs(N);
    std::vector<std::array<DualTrits,20>> outputs(N);
    
    // Initialize inputs with random data
    for (auto& a : inputs)
        for (int i = 0; i < 20; ++i)
            a[i] = randomDualTrits();

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            auto packed = pack20(inputs[i].data());
            unpack20(packed, outputs[i].data());
        }
        benchmark::DoNotOptimize(outputs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(N) * sizeof(std::array<DualTrits,20>) * 2);
}
BENCHMARK(BM_RoundTrip20_Batch)
    ->Arg(1<<10)->Arg(1<<12)->Arg(1<<14)
    ->MinTime(0.5)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

BENCHMARK_MAIN();
