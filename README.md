# DUAL-TRITS

[![CI](https://github.com/bugparty/DualTrits/actions/workflows/ci.yml/badge.svg)](https://github.com/bugparty/DualTrits/actions/workflows/ci.yml)
[![Benchmark](https://github.com/bugparty/DualTrits/actions/workflows/benchmark.yml/badge.svg)](https://github.com/bugparty/DualTrits/actions/workflows/benchmark.yml)

need to install libmpfr-dev libmpfrc++-dev libgmp-dev

## Building and Testing

### Prerequisites

**CPU-only build:**
```bash
sudo apt-get install -y libmpfr-dev libmpfrc++-dev libgmp-dev cmake build-essential
```

**CUDA build** (requires NVIDIA GPU):
- CUDA Toolkit ≥ 11.0 (includes `nvcc`)

**ROCm/HIP build** (requires AMD GPU):
- ROCm ≥ 5.0 — see https://rocm.docs.amd.com/ for installation instructions
- `hipcc` must be on `PATH` (typically `/opt/rocm/bin/hipcc`)
- Supported GPU architectures: `gfx906` (MI50/60), `gfx908` (MI100), `gfx90a` (MI200), `gfx942` (MI300X), `gfx1030` (RX 6000), `gfx1100`/`gfx1101` (RX 7000)

---

### Build the project

#### CPU-only (default)
```bash
# Debug build
cmake -B build
cmake --build build

# Release build (recommended for benchmarks)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_MPFR=ON
cmake --build build
```

#### CUDA backend
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDUALTRITS_GPU_BACKEND=CUDA
cmake --build build
```

#### ROCm/HIP backend
```bash
export PATH=/opt/rocm/bin:$PATH   # ensure hipcc is on PATH
cmake -B build_hip -DCMAKE_BUILD_TYPE=Release -DDUALTRITS_GPU_BACKEND=HIP
cmake --build build_hip
```

> **Note:** The default backend is `AUTO`, which tries CUDA first, then HIP, then falls back to
> CPU-only. Pass `-DDUALTRITS_GPU_BACKEND=NONE` to force a CPU-only build.

---

### Run tests

#### CPU tests
```bash
./run_tests.sh
```

#### CUDA tests
```bash
./run_tests_cuda.sh
```

#### ROCm/HIP tests
```bash
./run_tests_rocm.sh
```

Or manually:
```bash
ctest --test-dir build_hip --output-on-failure -R HipPack
```

---

### Run benchmarks

#### CPU benchmarks
```bash
./run_benchmarks.sh
```

#### CUDA benchmarks
```bash
./run_benchmarks_cuda.sh
```

#### ROCm/HIP benchmarks
```bash
./run_benchmarks_rocm.sh
```

Or run the benchmark binary directly:
```bash
./build_hip/packing_hip_benchmarks
```

Benchmark options:
```bash
# Run specific benchmark with filter
./build_hip/packing_hip_benchmarks --benchmark_filter=Pack5

# Output results in JSON format
./build_hip/packing_hip_benchmarks --benchmark_format=json

# Run for minimum time
./build_hip/packing_hip_benchmarks --benchmark_min_time=5.0
```

---

### ROCm environment variables

| Variable | Default | Description |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` | ROCm installation prefix used by the helper scripts |
| `CMAKE_HIP_COMPILER` | (auto-detected) | Override the path to `hipcc` |

---

### Known limitations (ROCm)

- The unpack stride kernel grid/block dimensions are tuned for NVIDIA hardware defaults; AMD GPU users may want to adjust `blockSize` and `gridSize` in `src/hip/dual_trits_pack.hip` for their specific CU count.
- `__umulhi` and `__umul64hi` intrinsics are supported in HIP (same API as CUDA).

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

- **CUDA CI** (`.github/workflows/cuda_ci.yml`): Self-hosted runner with NVIDIA GPU
  - Builds and tests the CUDA backend

- **ROCm CI** (`.github/workflows/rocm_ci.yml`): Self-hosted runner with AMD GPU
  - Builds and tests the ROCm/HIP backend

- **ROCm Benchmark** (`.github/workflows/benchmark_rocm.yml`): Self-hosted runner with AMD GPU
  - Runs comprehensive HIP benchmarks and tracks performance over time

# phases

 ## phase 0
- [x] Implement this format in C++ and CUDA, support basic arithmetic operations, support convert to and convert from standard formats (FP32, FP16, FP4), benchmark the software implementation speed impact compared to FP4.

## phase 1

- [x] Implement a PyTorch/CUDA layer that:Stores weights in dual-trit format (compressed),Decodes on-demand to FP8/FP16 during forward pass,Caches decoded weights if memory allows.

## phase 2

 - [ ] Compare accuracy and memory usage of deep learning networks using both formats.
## phase 3

 - [ ]  Quantization-Aware Training (QAT)

   - [ ] 1. Fine-tune pre-trained models with dual-trit simulation
   - [ ] 1. Compare convergence vs FP4-QAT
   - [ ] 1. Comparison with SOTA Methods
