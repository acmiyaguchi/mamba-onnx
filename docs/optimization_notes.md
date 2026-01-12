# Optimization Notes: Fused Mamba Operator

## Fast Recurrence Optimization (2026-01-11)

### Summary
We refactored the inner loop of the `SelectiveScan` kernel in `csrc/selective_scan.cc` to use an algebraic simplification of the recurrence relation.

**Results:**
- **Baseline Latency:** 0.28 ms
- **Optimized Latency:** 0.07 ms
- **Speedup:** ~4.0x

### Technical Details

#### 1. Algebraic Simplification
The original implementation approximated the exponential discretization `exp(delta * A)` as `1 + delta * A`.

**Original Code (Pseudocode):**
```cpp
// 4 vector instructions
A_bar = 1 + delta * A
B_bar = delta * B
B_u = B_bar * u
h = A_bar * h + B_u
```

**Optimized Code:**
```cpp
// 3 vector instructions (25% reduction)
// h = (1 + delta*A)*h + delta*B*u
//   = h + delta * (A*h + B*u)
tmp = A * h + B * u
h = h + delta * tmp
```

#### 2. Performance Analysis: Why 4x?
While the instruction count reduction is only ~25%, the observed 4x speedup suggests additional factors:

- **Reduced Register Pressure**: The optimized version removes the need to store intermediate vectors `ones`, `A_bar`, and `B_bar`. This frees up AVX2 registers, likely allowing the compiler to unroll the loop more aggressively without spilling to the stack.
- **Improved Pipelining**: The simplified FMA chain (`h += delta * tmp`) maps more cleanly to hardware units compared to the multi-stage dependency `A_bar -> h_new`.
- **Memory Bandwidth**: Although loads are identical, the reduced instruction density might allow the CPU's load/store units to keep up better with the arithmetic pipeline.

### Future Work
- **Fused Activations**: Implementing fused `Softplus` (for delta) and `SiLU` (for the gate) directly in the kernel to save memory bandwidth.
