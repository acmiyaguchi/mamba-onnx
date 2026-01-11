# Project Tasks

## Phase 1: The Control (The "Slow" Path)
- [ ] **Isolate MambaBlock**: Create `export_baseline.py` that instantiates a single `MambaBlock` (`d_model=768`, `d_state=16`).
- [ ] **Export Baseline**: Run `torch.onnx.export` to generate the unrolled loop ONNX model.
- [ ] **Benchmark Baseline**: Measure `latency_per_token` using ONNX Runtime.

## Phase 2: The Kernel (The "Fast" Path)
- [ ] **Generate Kernel Source**: Create `src/selective_scan.cc` implementing the recurrence $h_t = A \cdot h_{t-1} + B \cdot x_t$.
- [ ] **Optimize**: Ensure usage of `_mm256_fmadd_ps` (AVX2 FMA) for vector updates.
- [ ] **Build System**: Create `CMakeLists.txt` to compile the source into `libmamba_ops.so`.
- [ ] **Compile**: Build the shared library.

## Phase 3: Integration
- [ ] **Create Shim**: Implement `MambaScanOp` in PyTorch (`torch.autograd.Function`).
- [ ] **Register Symbolic**: Add the symbolic function so export emits a `CustomOp`.
- [ ] **Export Optimized**: Export the model using the custom op.
- [ ] **Run & Benchmark**: Load `libmamba_ops.so` in ONNX Runtime, run the optimized model, and compare performance.
