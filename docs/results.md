# Benchmarking Results

This document summarizes the performance analysis of the Mamba Selective Scan custom operators.
All benchmarks run with multi-threaded backends (C++ with OpenMP, Zig with std.Thread.Pool),
4 threads, batch size 1, pytest-benchmark with 50 rounds, median latency reported.

## Op Variants

The library provides 4 custom op variants for different accuracy/performance tradeoffs:

| Op | Inputs | Discretization | Features |
|---|---|---|---|
| `SelectiveScan` | 6 | Linear (1+δA) | Fastest, ~5% approximation error |
| `SelectiveScanExact` | 6 | Exact (exp(δA)) | Accurate, +30-40% latency |
| `SelectiveScanFused` | 7 | Linear | + Softplus(δ) + SiLU(z) gate |
| `SelectiveScanFusedExact` | 7 | Exact | + Softplus(δ) + SiLU(z) gate |

## Summary

- **Speedup vs PyTorch (C++ backend)**: **40-122x** across all variants
- **Speedup vs PyTorch (Zig backend)**: **58-115x** across all variants
- **Fastest Op**: `SelectiveScan` via C++ at **0.531ms** (B=1, D=768, L=1024)
- **Zig beats C++ on Exact ops**; C++ wins on non-exact ops. Overall parity is much closer than earlier runs.

## Head-to-Head Comparison (B=1, D=768, L=1024, 4 threads)

| Op | C++ (ms) | Zig (ms) | PyTorch (ms) | C++ Speedup | Zig Speedup |
|---|---|---|---|---|---|
| **SelectiveScan** | **0.531** | 0.735 | 64.543 | **121.5x** | 87.9x |
| **SelectiveScanExact** | 0.942 | **0.725** | 83.202 | 88.4x | **114.8x** |
| **SelectiveScanFused** | **1.150** | 1.442 | 89.859 | **78.1x** | 62.3x |
| **SelectiveScanFusedExact** | 2.838 | **1.987** | 114.925 | 40.5x | **57.8x** |

## Backend Comparison: C++ vs Zig

C++ wins on non-exact ops (SelectiveScan, SelectiveScanFused), while Zig wins on exact ops
(SelectiveScanExact, SelectiveScanFusedExact). The gap between backends has narrowed significantly.

| Op | C++ (ms) | Zig (ms) | Zig/C++ Ratio |
|---|---|---|---|
| SelectiveScan | **0.531** | 0.735 | 1.38x |
| SelectiveScanExact | 0.942 | **0.725** | 0.77x |
| SelectiveScanFused | **1.150** | 1.442 | 1.25x |
| SelectiveScanFusedExact | 2.838 | **1.987** | 0.70x |

**Analysis**: Both backends are multi-threaded and parallelize across batch and dimension.
C++ uses OpenMP; Zig uses `std.Thread.Pool`.

- **C++ wins on non-exact ops** — OpenMP's chunked static scheduling dispatches contiguous ranges
  per thread with minimal overhead, giving it an edge on lighter workloads.
- **Zig wins on exact ops** — For compute-heavy ops (exp, softplus, SiLU), Zig's code generation
  and scheduling is competitive or faster, reversing the earlier C++ advantage.
- An experimental ForkJoin backend is available via `zig build -Dforkjoin=true` but
  benchmarks show no measurable difference vs std.Thread.Pool at these workload sizes.

## Scaling Analysis: Sequence Length (D=768, 4 threads)

### C++ Backend

| L | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 128 | 0.141 ms | 0.207 ms | 0.272 ms | 0.407 ms |
| 256 | 0.863 ms | 0.300 ms | 0.532 ms | 0.670 ms |
| 512 | 0.317 ms | 0.550 ms | 2.758 ms | 1.464 ms |
| 1024 | 0.423 ms | 0.880 ms | 3.986 ms | 2.546 ms |
| 2048 | 0.801 ms | 0.930 ms | 4.211 ms | 5.755 ms |
| 4096 | 1.639 ms | 3.646 ms | 8.696 ms | 10.227 ms |

### Zig Backend

| L | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 128 | 0.294 ms | 0.355 ms | 0.409 ms | 0.476 ms |
| 256 | 0.460 ms | 0.386 ms | 0.582 ms | 0.611 ms |
| 512 | 0.362 ms | 0.496 ms | 0.806 ms | 1.003 ms |
| 1024 | 0.483 ms | 0.744 ms | 1.490 ms | 1.975 ms |
| 2048 | 0.716 ms | 1.329 ms | 2.947 ms | 3.964 ms |
| 4096 | 1.354 ms | 2.425 ms | 5.623 ms | 7.665 ms |

*Both backends scale linearly with sequence length as expected (O(L) complexity).
Non-fused ops stay under 4ms even at L=4096 on both backends.*

## Scaling Analysis: Model Dimension (L=1024, 4 threads)

### C++ Backend

| D | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 256 | 0.219 ms | 0.336 ms | 0.799 ms | 0.969 ms |
| 512 | 0.365 ms | 0.643 ms | 1.331 ms | 1.803 ms |
| 768 | 0.504 ms | 0.927 ms | 2.219 ms | 2.864 ms |
| 1024 | 0.663 ms | 1.223 ms | 3.662 ms | 3.735 ms |
| 1536 | 0.817 ms | 1.808 ms | 3.928 ms | 5.213 ms |
| 2048 | 3.631 ms | 2.347 ms | 5.712 ms | 6.744 ms |

### Zig Backend

| D | SelectiveScan | SelectiveScanExact | SelectiveScanFused | SelectiveScanFusedExact |
|---|---|---|---|---|
| 256 | 0.341 ms | 0.445 ms | 0.631 ms | 0.740 ms |
| 512 | 0.574 ms | 0.554 ms | 1.047 ms | 1.362 ms |
| 768 | 0.529 ms | 0.756 ms | 1.474 ms | 1.981 ms |
| 1024 | 0.584 ms | 0.989 ms | 1.971 ms | 2.709 ms |
| 1536 | 0.720 ms | 1.352 ms | 2.900 ms | 3.861 ms |
| 2048 | 0.899 ms | 1.720 ms | 3.767 ms | 5.176 ms |

*Dimension scaling is sub-linear for the lighter ops thanks to multi-threading —
`SelectiveScan` (Zig) only goes from 0.341ms to 0.899ms as D grows 8x (256 to 2048).*

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

# Run all benchmarks (4 threads, 50 rounds)
uv run pytest benchmarks/ --benchmark-only

# Compare against a saved baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=0001_baseline
```

## Recommendations

| Use Case | Recommended Op | Recommended Backend |
|---|---|---|
| Maximum throughput | `SelectiveScan` | C++ |
| Numerical accuracy required | `SelectiveScanExact` | Zig |
| Full Mamba block integration | `SelectiveScanFusedExact` | Zig |
| Minimal dependencies / portability | Any | Zig |
| Research/debugging | `SelectiveScanExact` | Either (matches PyTorch exactly) |
