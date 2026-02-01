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

### Benchmarking Tool Evaluation: asv vs pytest-benchmark

#### Outcome

The project evaluated `asv` and `pytest-benchmark` as replacements for the custom `benchmarks/suite.py` timing harness. **pytest-benchmark was adopted** and `suite.py` was retired.

#### Why pytest-benchmark won

- **Minimal migration effort**: Existing bench calls wrapped in `benchmark.pedantic()` with minimal boilerplate.
- **Statistical rigor**: Automatic stddev, outlier detection, and configurable rounds — a major upgrade over single-mean `time.perf_counter()` timing.
- **Fits existing workflow**: Runs via `uv run pytest benchmarks/`, integrates with the existing pytest runner.
- **JSON export**: `--benchmark-save` produces machine-readable results for diffing across runs (`--benchmark-compare`).

#### Why asv was rejected

- Requires rewriting benchmarks into asv's class-based format.
- Rebuilds the project per commit via `pip install`, which is slow and fragile with compiled Zig/C++ backends and a custom build system.
- Value is maximized in CI, which the project doesn't have yet.

#### Current setup

Benchmarks live in `benchmarks/` and run via:
```bash
uv run pytest benchmarks/ --benchmark-only
```

Results are saved to `.benchmarks/` in JSON format. Use `--benchmark-compare` to diff against previous runs.
