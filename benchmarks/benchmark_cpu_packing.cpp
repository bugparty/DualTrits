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
    std::vector<DualTrits> inputs(N * 5);
    // Initialize inputs with random data
    for (int i = 0; i < N * 5; ++i)
        inputs[i] = randomDualTrits();

    for (auto _ : state) {
        uint64_t sink = 0;
        auto packed = pack5(inputs.data(), N * 5);
        benchmark::DoNotOptimize(packed.data());
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
    std::vector<DualTrits> unpacked_inputs(N * 5);
    std::vector<std::uint16_t> packed_inputs(N);
    std::vector<DualTrits> outputs(N * 5);
    
    // Initialize packed inputs with random data
    for (int i = 0; i < N * 5; ++i) {
        unpacked_inputs[i] = randomDualTrits();
    }
    packed_inputs = pack5(unpacked_inputs.data(), N * 5);

    for (auto _ : state) {
        unpack5(packed_inputs.data(), outputs.data(), N);
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
    std::vector<DualTrits> inputs(N * 10);
    // Initialize inputs with random data
    for (int i = 0; i < N * 10; ++i)
        inputs[i] = randomDualTrits();

    for (auto _ : state) {
        uint64_t sink = 0;
        auto packed = pack10(inputs.data(), N * 10);
        benchmark::DoNotOptimize(packed.data());
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
    std::vector<DualTrits> unpacked_inputs(N * 10);
    std::vector<std::uint32_t> packed_inputs(N);
    std::vector<DualTrits> outputs(N * 10);
    
    // Initialize packed inputs with random data
    for (int i = 0; i < N * 10; ++i) {
        unpacked_inputs[i] = randomDualTrits();
    }
    packed_inputs = pack10(unpacked_inputs.data(), N * 10);

    for (auto _ : state) {
        unpack10(packed_inputs.data(), outputs.data(), N);
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
    std::vector<DualTrits> inputs(N * 20);
    // Initialize inputs with random data
    for (int i = 0; i < N * 20; ++i)
        inputs[i] = randomDualTrits();

    for (auto _ : state) {
        uint64_t sink = 0;
        auto packed = pack20(inputs.data(), N * 20);
        benchmark::DoNotOptimize(packed.data());
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
    std::vector<DualTrits> unpacked_inputs(N * 20);
    std::vector<std::uint64_t> packed_inputs(N);
    std::vector<DualTrits> outputs(N * 20);
    
    // Initialize packed inputs with random data
    for (int i = 0; i < N * 20; ++i) {
        unpacked_inputs[i] = randomDualTrits();
    }
    packed_inputs = pack20(unpacked_inputs.data(), N * 20);

    for (auto _ : state) {
        unpack20(packed_inputs.data(), outputs.data(), N);
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
    std::vector<DualTrits> inputs(N * 5);
    std::vector<DualTrits> outputs(N * 5);
    
    // Initialize inputs with random data
    for (int i = 0; i < 5 * N; ++i)
        inputs[i] = randomDualTrits();

    for (auto _ : state) {
        auto packed = pack5(inputs.data(), N * 5);
        unpack5(packed.data(), outputs.data(), N);
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
    std::vector<DualTrits> inputs(N * 10);
    std::vector<DualTrits> outputs(N * 10);
    
    // Initialize inputs with random data
    for (int i = 0; i < N * 10; ++i)
        inputs[i] = randomDualTrits();

    for (auto _ : state) {
        auto packed = pack10(inputs.data(), N * 10);
        unpack10(packed.data(), outputs.data(), N);
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
    std::vector<DualTrits> inputs(N * 20);
    std::vector<DualTrits> outputs(N * 20);
    
    // Initialize inputs with random data
    for (int i = 0; i < N * 20; ++i)
        inputs[i] = randomDualTrits();

    for (auto _ : state) {
        auto packed = pack20(inputs.data(), N * 20);
        unpack20(packed.data(), outputs.data(), N);
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
