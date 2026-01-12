# Project Tasks

## Phase 1: The Control (The "Slow" Path)
- [x] **Isolate MambaBlock**: Create `export_baseline.py` that instantiates a single `MambaBlock` (`d_model=768`, `d_state=16`).
- [x] **Export Baseline**: Run `torch.onnx.export` to generate the unrolled loop ONNX model.
- [x] **Benchmark Baseline**: Measure `latency_per_token` using ONNX Runtime.

## Phase 2: The Kernel (The "Fast" Path)
- [x] **Generate Kernel Source**: Create `src/selective_scan.cc` implementing the recurrence $h_t = A \cdot h_{t-1} + B \cdot x_t$.
- [x] **Optimize**: Ensure usage of `_mm256_fmadd_ps` (AVX2 FMA) for vector updates.
- [x] **Build System**: Create `CMakeLists.txt` to compile the source into `libmamba_ops.so`.
- [x] **Compile**: Build the shared library.

## Phase 3: Integration
- [x] **Create Shim**: Implement `MambaScanOp` in PyTorch (`torch.autograd.Function`).
- [x] **Register Symbolic**: Add the symbolic function so export emits a `CustomOp`.
- [x] **Export Optimized**: Export the model using the custom op.
- [x] **Run & Benchmark**: Load `libmamba_ops.so` in ONNX Runtime, run the optimized model, and compare performance.

## Phase 4: Optimization
- [x] **Algebraic Simplification**: Refactor inner loop to use $h = h + \Delta(Ah + Bu)$ identity, reducing vector ops by 25%.
    - Result: **4x speedup** (0.28ms -> 0.07ms). See [docs/optimization_notes.md](optimization_notes.md).
- [ ] **Fused Softplus and Z-Gate**: Fuse `Softplus(delta)` and `SiLU(z) * y` into the kernel to save memory bandwidth.
    - Reference: `vendor/mamba/csrc/selective_scan/selective_scan_fwd_kernel.cuh` (lines 159-161 for Softplus, 285-303 for Z-Gate).
    - Goal: Update ONNX schema and C++ kernel to handle raw inputs.

## Phase 5: Comprehensive Benchmarking & Analysis
- [x] **Extended Metrics**: Implement measurement of Throughput (tokens/s) and Peak Memory Usage.
- [x] **Scaling Analysis**: Benchmark across varying Sequence Lengths (`L`) and Model Dimensions (`D`).
- [x] **Visualizations**: Generate plots (bar charts, scaling curves) to visualize relative improvements vs PyTorch/Baseline.
- [x] **Automated Suite**: Create a unified runner to execute sweeping benchmarks and report results.
