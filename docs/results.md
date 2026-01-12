# Benchmarking Results

This document summarizes the performance gains achieved by using the fused Mamba Selective Scan custom operator in ONNX Runtime compared to standard PyTorch and unrolled ONNX baselines.

## Experimental Setup

- **Sequence Length ($L$):** 128 (limited by baseline unrolling) and 2048 (for fused performance).
- **Model Dimension ($D$):** 768.
- **State Dimension ($N$):** 16.
- **Hardware:** CPU (AVX2 supported).

## Performance Comparison ($L=128, D=768$)

At a sequence length of 128, we can still export the unrolled baseline (Standard ONNX ops) for a complete comparison.

| Method | Latency | Speedup vs PyTorch |
| :--- | :--- | :--- |
| **PyTorch Eager** | 9.59 ms | 1.0x |
| **Baseline ONNX** (Standard Ops) | 6.61 ms | ~1.45x |
| **Fused ONNX** (Custom Kernel) | **0.08 ms** | **~122x** |

## Performance at Scale ($L=2048, D=768$)

At longer sequences, the unrolled baseline fails to export or run efficiently due to graph size explosion. The Fused ONNX kernel maintains exceptional efficiency.

| Method | Latency | Speedup vs PyTorch |
| :--- | :--- | :--- |
| **PyTorch Eager** | ~210 ms | 1.0x |
| **Fused ONNX** (Custom Kernel) | **~2.00 ms** | **~105x** |

## Analysis

### 1. The Bottleneck in Standard ONNX
Standard ONNX Runtime provides a modest improvement (~1.5x) over PyTorch by optimizing individual mathematical operations. However, for sequential recurrences like Mamba's Selective Scan, the "unrolled loop" approach results in thousands of small nodes. This creates massive overhead in:
- **Instruction dispatch**: The engine must schedule and execute each small node.
- **Memory Bandwidth**: Data is repeatedly written to and read from memory between individual Add/Mul ops.

### 2. Why Fusion Works
The custom C++ kernel achieves **100x+ speedups** by addressing these bottlenecks:
- **Zero Interpreter Overhead**: The entire sequence loop is executed in a single native call.
- **L1 Cache Efficiency**: Intermediate state values are kept in CPU registers and L1 cache, never hitting main memory during the sequence pass.
- **SIMD (AVX2)**: Hand-optimized intrinsics process 8 channels in parallel within each step.
- **OpenMP**: Parallelizes across the model dimension ($D$) and batch ($B$) for multi-core scaling.
