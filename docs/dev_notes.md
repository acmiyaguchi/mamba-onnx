# Development Notes

## Progress Log

### Phase 1: Baseline
- Successfully created `export_baseline.py` using `transformers.MambaModel`.
- Identified that `torch.onnx.export` unrolls the selective scan loop into hundreds of nodes (600+ for seqlen=64).
- Confirmed the "Slow Path" hypothesis.

### Phase 2: Kernel Implementation
- Implemented `src/selective_scan.cc` using AVX2 intrinsics (`_mm256_fmadd_ps`).
- **Challenge**: `onnxruntime` pip package does not include C++ headers.
  - **Solution**: Vendored `onnxruntime` (v1.17.1) as a submodule.
- **Challenge**: C++ Wrapper (`Ort::CustomOpApi`) availability.
  - **Solution**: Rewrote the kernel to use the raw C API (`OrtApi` struct) via `api_` member.
- **Challenge**: `undefined symbol: OrtGetApiBase` when loading the shared library.
  - **Solution**: Added a dummy `extern "C" const OrtApiBase* OrtGetApiBase(void) noexcept { return nullptr; }` to satisfy the linker, as the shared library is a plugin and shouldn't link against ORT core.

### Phase 3: Integration & Benchmarking
- Created `experiments/benchmark_scan.py` to compare Reference (Python Loop) vs Fused (Custom Op).
- **Challenge**: Exporting the Reference model with `L=1024` timed out due to massive graph generation.
  - **Solution**: Reduced `L` to 128 for benchmarking.
- **Current Status**: The benchmark runs but crashes with **Exit Code 139 (Segfault)** when loading/executing the custom op.
- **Next Step**: Debug the segfault with a minimal unit test.
