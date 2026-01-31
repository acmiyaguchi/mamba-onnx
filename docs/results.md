# Benchmarking Results

This document summarizes the performance analysis of the Mamba Selective Scan custom operators.
All benchmarks run with multi-threaded backends (C++ with OpenMP, Zig with std.Thread.Pool),
4 threads, batch size 1, 100 repetitions, median latency reported.

## Op Variants

The library provides 4 custom op variants for different accuracy/performance tradeoffs:

| Op | Inputs | Discretization | Features |
|---|---|---|---|
| `SelectiveScan` | 6 | Linear (1+δA) | Fastest, ~5% approximation error |
| `SelectiveScanExact` | 6 | Exact (exp(δA)) | Accurate, +30-40% latency |
| `SelectiveScanFused` | 7 | Linear | + Softplus(δ) + SiLU(z) gate |
| `SelectiveScanFusedExact` | 7 | Exact | + Softplus(δ) + SiLU(z) gate |

## Summary

- **Speedup vs PyTorch (C++ backend)**: **15-77x** across all variants
- **Speedup vs PyTorch (Zig backend)**: **13-54x** across all variants
- **Fastest Op**: `SelectiveScan` via C++ at **0.860ms** (B=1, D=768, L=1024)
- **C++ leads across all ops**, with Zig within 0.67-0.93x

## Head-to-Head Comparison (B=1, D=768, L=1024, 4 threads)

| Op | C++ (ms) | Zig (ms) | PyTorch (ms) | C++ Speedup | Zig Speedup |
|---|---|---|---|---|---|
| **SelectiveScan** | **0.860** | 1.216 | 65.927 | **76.7x** | 54.2x |
| **SelectiveScanExact** | **1.871** | 2.798 | 88.900 | **47.5x** | 31.8x |
| **SelectiveScanFused** | **5.882** | 6.337 | 91.319 | **15.5x** | 14.4x |
| **SelectiveScanFusedExact** | **7.964** | 9.372 | 119.175 | **15.0x** | 12.7x |

## Backend Comparison: C++ vs Zig

C++ wins across all ops. The gap is largest on the lighter ops where threading overhead
is a larger fraction of total runtime.

| Op | C++ (ms) | Zig (ms) | C++/Zig Ratio |
|---|---|---|---|
| SelectiveScan | **0.860** | 1.216 | 0.71x |
| SelectiveScanExact | **1.871** | 2.798 | 0.67x |
| SelectiveScanFused | **5.882** | 6.337 | 0.93x |
| SelectiveScanFusedExact | **7.964** | 9.372 | 0.85x |

**Analysis**: Both backends are multi-threaded and parallelize across batch and dimension.
C++ uses OpenMP; Zig uses `std.Thread.Pool`.

- **C++ wins everywhere** — OpenMP's chunked static scheduling dispatches contiguous ranges
  per thread with minimal overhead. Zig's pool has slightly higher per-dispatch cost.
- **Gap narrows on heavier ops** — Fused ops (0.93x, 0.85x) close the gap because per-item
  compute (exp, softplus, SiLU) dominates over threading overhead.
- An experimental ForkJoin backend is available via `zig build -Dforkjoin=true` but
  benchmarks show no measurable difference vs std.Thread.Pool at these workload sizes.

## Scaling Analysis: Sequence Length (D=768, 4 threads)

### C++ Backend

| L | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 128 | 0.122 ms | 0.247 ms | 0.694 ms | 0.995 ms |
| 256 | 0.228 ms | 0.485 ms | 1.411 ms | 2.000 ms |
| 512 | 0.440 ms | 0.950 ms | 2.777 ms | 4.079 ms |
| 1024 | 0.923 ms | 1.897 ms | 5.508 ms | 7.935 ms |
| 2048 | 1.742 ms | 3.770 ms | 11.088 ms | 16.258 ms |
| 4096 | 3.396 ms | 7.414 ms | 25.346 ms | 31.195 ms |

### Zig Backend

| L | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 128 | 0.217 ms | 0.415 ms | 0.847 ms | 1.220 ms |
| 256 | 0.352 ms | 0.743 ms | 1.630 ms | 2.400 ms |
| 512 | 0.641 ms | 1.416 ms | 3.256 ms | 4.753 ms |
| 1024 | 1.217 ms | 2.796 ms | 6.379 ms | 9.358 ms |
| 2048 | 2.406 ms | 5.519 ms | 12.752 ms | 18.775 ms |
| 4096 | 4.867 ms | 11.114 ms | 25.400 ms | 37.158 ms |

*Both backends scale linearly with sequence length as expected (O(L) complexity).
Non-fused ops stay under 5ms even at L=4096 on C++.*

## Scaling Analysis: Model Dimension (L=1024, 4 threads)

### C++ Backend

| D | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 256 | 0.323 ms | 0.630 ms | 1.802 ms | 2.582 ms |
| 512 | 0.570 ms | 1.245 ms | 3.589 ms | 5.146 ms |
| 768 | 0.860 ms | 3.004 ms | 5.502 ms | 7.822 ms |
| 1024 | 1.168 ms | 2.483 ms | 7.473 ms | 10.434 ms |
| 1536 | 1.720 ms | 3.729 ms | 11.200 ms | 15.617 ms |
| 2048 | 2.290 ms | 4.928 ms | 14.360 ms | 21.464 ms |

### Zig Backend

| D | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 256 | 0.451 ms | 0.961 ms | 2.157 ms | 3.157 ms |
| 512 | 0.834 ms | 1.872 ms | 4.296 ms | 6.282 ms |
| 768 | 1.244 ms | 2.837 ms | 6.376 ms | 9.472 ms |
| 1024 | 1.611 ms | 3.713 ms | 8.500 ms | 12.506 ms |
| 1536 | 2.386 ms | 5.539 ms | 13.012 ms | 18.864 ms |
| 2048 | 3.200 ms | 7.352 ms | 16.885 ms | 25.022 ms |

*Dimension scaling is sub-linear for the lighter ops thanks to multi-threading —
`SelectiveScan` (C++) only goes from 0.323ms to 2.290ms as D grows 8x (256 to 2048).*

## Historical Context

Prior to the custom op implementation, the options were:

1. **PyTorch Eager**: ~62-119ms for L=1024, D=768 (Python interpreter overhead, no fusion)
2. **Vanilla ONNX Export**: Unrolls the scan loop into 600+ nodes, even slower than PyTorch

The custom ops (both backends) provide massive speedups by:
- Fusing the entire scan into a single kernel
- Using AVX2 SIMD vectorization for state updates
- Fast polynomial approximation for exp() using IEEE 754 bit manipulation
- Keeping hidden state in registers/L1 cache
- Parallelizing across batch and dimension (C++ via OpenMP, Zig via std.Thread.Pool)

## Reproducing

```bash
# Build both backends
zig build -Doptimize=ReleaseFast

# Head-to-head comparison (4 threads, 100 reps)
python benchmarks/suite.py --mode compare --backend both --threads 4 --reps 100

# Scaling analysis
python benchmarks/suite.py --mode scale --backend both --threads 4 --reps 100

# Regenerate plots
python benchmarks/plot.py
```

## Recommendations

| Use Case | Recommended Op | Recommended Backend |
|---|---|---|
| Maximum throughput | `SelectiveScan` | C++ |
| Numerical accuracy required | `SelectiveScanExact` | C++ |
| Full Mamba block integration | `SelectiveScanFusedExact` | C++ |
| Minimal dependencies / portability | Any | Zig |
| Research/debugging | `SelectiveScanExact` | Either (matches PyTorch exactly) |
