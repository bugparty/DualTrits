# DUAL-TRITS

[![CI](https://github.com/bugparty/DualTrits/actions/workflows/ci.yml/badge.svg)](https://github.com/bugparty/DualTrits/actions/workflows/ci.yml)
[![Benchmark](https://github.com/bugparty/DualTrits/actions/workflows/benchmark.yml/badge.svg)](https://github.com/bugparty/DualTrits/actions/workflows/benchmark.yml)

need to install libmpfr-dev libmpfrc++-dev libgmp-dev

## Building and Testing

### Build the project
```bash
# Debug build
cmake -B build
cmake --build build

# Release build (recommended for benchmarks)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_MPFR=ON
cmake --build build
```

### Run tests
```bash
./run_tests.sh
```

### Run benchmarks
```bash
./run_benchmarks.sh
```

Or run benchmarks directly:
```bash
./build/packing_benchmarks
```

Benchmark options:
```bash
# Run specific benchmark with filter
./build/packing_benchmarks --benchmark_filter=Pack5

# Output results in JSON format
./build/packing_benchmarks --benchmark_format=json

# Run for minimum time
./build/packing_benchmarks --benchmark_min_time=5.0
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **CI Workflow** (`.github/workflows/ci.yml`): Runs on every push and PR
  - Builds the project in Release mode
  - Runs all tests with CTest
  - Runs quick benchmarks (with `--benchmark_min_time=0.1`)

- **Benchmark Workflow** (`.github/workflows/benchmark.yml`): Dedicated benchmark runs
  - Can be triggered manually via workflow_dispatch
  - Runs comprehensive benchmarks
  - Uploads benchmark results as artifacts (JSON format)
  - Displays results in the GitHub Actions summary

# phases

## phase 0
Implement this format in C++ and CUDA, support basic arithmetic operations, support convert to and convert from standard formats (FP32, FP16, FP4), benchmark the software implementation speed impact compared to FP4.

## phase 1

Implement a PyTorch/CUDA layer that:Stores weights in dual-trit format (compressed),Decodes on-demand to FP8/FP16 during forward pass,Caches decoded weights if memory allows.

## phase 2

Compare accuracy and memory usage of deep learning networks using both formats.
## phase 3

Quantization-Aware Training (QAT)

1. Fine-tune pre-trained models with dual-trit simulation
1. Compare convergence vs FP4-QAT
1. Comparison with SOTA Methods
