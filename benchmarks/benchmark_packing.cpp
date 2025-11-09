#include <benchmark/benchmark.h>
#include "../DualTrits.h"
#include "../dual_trits_pack.hpp"

// Benchmark: Pack 5 DualTrits into uint16_t
static void BM_Pack5(benchmark::State& state) {
    DualTrits arr[5] = {
        DualTrits(0, 0),
        DualTrits(1, 2),
        DualTrits(2, 1),
        DualTrits(1, 1),
        DualTrits(2, 0)
    };
    
    for (auto _ : state) {
        auto packed = pack5(arr);
        benchmark::DoNotOptimize(packed);
    }
}
BENCHMARK(BM_Pack5);

// Benchmark: Unpack 5 DualTrits from uint16_t
static void BM_Unpack5(benchmark::State& state) {
    DualTrits arr[5] = {
        DualTrits(0, 0),
        DualTrits(1, 2),
        DualTrits(2, 1),
        DualTrits(1, 1),
        DualTrits(2, 0)
    };
    std::uint16_t packed = pack5(arr);
    DualTrits out[5];
    
    for (auto _ : state) {
        unpack5(packed, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_Unpack5);

// Benchmark: Pack 10 DualTrits into uint32_t
static void BM_Pack10(benchmark::State& state) {
    DualTrits arr[10];
    for (int i = 0; i < 10; ++i) {
        arr[i] = DualTrits(i % 3, static_cast<DualTrits::wide_t>((i+1) % 3));
    }
    
    for (auto _ : state) {
        auto packed = pack10(arr);
        benchmark::DoNotOptimize(packed);
    }
}
BENCHMARK(BM_Pack10);

// Benchmark: Unpack 10 DualTrits from uint32_t
static void BM_Unpack10(benchmark::State& state) {
    DualTrits arr[10];
    for (int i = 0; i < 10; ++i) {
        arr[i] = DualTrits(i % 3, static_cast<DualTrits::wide_t>((i+1) % 3));
    }
    std::uint32_t packed = pack10(arr);
    DualTrits out[10];
    
    for (auto _ : state) {
        unpack10(packed, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_Unpack10);

// Benchmark: Pack 20 DualTrits into uint64_t
static void BM_Pack20(benchmark::State& state) {
    DualTrits arr[20];
    for (int i = 0; i < 20; ++i) {
        arr[i] = DualTrits((i*2) % 3, static_cast<DualTrits::wide_t>((i+2) % 3));
    }
    
    for (auto _ : state) {
        auto packed = pack20(arr);
        benchmark::DoNotOptimize(packed);
    }
}
BENCHMARK(BM_Pack20);

// Benchmark: Unpack 20 DualTrits from uint64_t
static void BM_Unpack20(benchmark::State& state) {
    DualTrits arr[20];
    for (int i = 0; i < 20; ++i) {
        arr[i] = DualTrits((i*2) % 3, static_cast<DualTrits::wide_t>((i+2) % 3));
    }
    std::uint64_t packed = pack20(arr);
    DualTrits out[20];
    
    for (auto _ : state) {
        unpack20(packed, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_Unpack20);

// Benchmark: Round-trip pack and unpack for 5 DualTrits
static void BM_RoundTrip5(benchmark::State& state) {
    DualTrits arr[5] = {
        DualTrits(0, 0),
        DualTrits(1, 2),
        DualTrits(2, 1),
        DualTrits(1, 1),
        DualTrits(2, 0)
    };
    DualTrits out[5];
    
    for (auto _ : state) {
        auto packed = pack5(arr);
        unpack5(packed, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_RoundTrip5);

// Benchmark: Round-trip pack and unpack for 10 DualTrits
static void BM_RoundTrip10(benchmark::State& state) {
    DualTrits arr[10];
    for (int i = 0; i < 10; ++i) {
        arr[i] = DualTrits(i % 3, static_cast<DualTrits::wide_t>((i+1) % 3));
    }
    DualTrits out[10];
    
    for (auto _ : state) {
        auto packed = pack10(arr);
        unpack10(packed, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_RoundTrip10);

// Benchmark: Round-trip pack and unpack for 20 DualTrits
static void BM_RoundTrip20(benchmark::State& state) {
    DualTrits arr[20];
    for (int i = 0; i < 20; ++i) {
        arr[i] = DualTrits((i*2) % 3, static_cast<DualTrits::wide_t>((i+2) % 3));
    }
    DualTrits out[20];
    
    for (auto _ : state) {
        auto packed = pack20(arr);
        unpack20(packed, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_RoundTrip20);

BENCHMARK_MAIN();
