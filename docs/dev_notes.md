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

#### Current Setup

The project uses a custom `benchmarks/suite.py` that directly times ONNX Runtime sessions and PyTorch reference kernels via `time.perf_counter()`. It supports two modes (compare across backends/ops, scale across L/D dimensions), outputs tabular results to stdout, and saves CSV files. Benchmarks are run manually from the CLI.

#### asv (airspeed velocity)

**What it provides:** Historical results DB keyed by git commit, automatic regression detection, `asv continuous` for branch comparison, HTML dashboards with time-series plots, git-bisect-style binary search for regressions.

**Pros:** Tracks performance across kernel evolution; commit-pinned results verify refactors don't regress.

**Cons:** Requires rewriting `suite.py` into asv's class-based format. Rebuilds the project per commit via `pip install`, which is slow and fragile with compiled Zig/C++ backends and a custom build system. Heavier infrastructure (`asv.conf.json`, machine-specific results DB). Value is maximized in CI, which the project doesn't have yet.

#### pytest-benchmark

**What it provides:** `benchmark` fixture for timing, statistical analysis (stddev, outlier detection), histogram output, JSON export, `pytest-benchmark compare` for diffing saved results.

**Pros:** Minimal migration effort (wrap existing bench calls in `benchmark.pedantic()`). Statistical rigor is an upgrade over single-mean timing, important for noisy SIMD/threading benchmarks. Fits into existing pytest runner. JSON export enables lightweight CI regression checks.

**Cons:** No built-in historical tracking across commits. No HTML dashboard. No git-bisect integration. Comparing runs requires manually saving/diffing JSON files.

#### Recommendation

**Keep `suite.py` as the primary benchmarking tool.** It already covers comparison and scaling scenarios and outputs CSV for plotting. For a project without CI, adopting asv or pytest-benchmark doesn't pay for itself yet.

If adding one improvement: add pytest-benchmark as a secondary check via a `tests/test_bench.py` that wraps ONNX session timing in `benchmark.pedantic()`. This gives statistical rigor and outlier detection for free, and JSON output can later feed CI regression gates.

Avoid asv until the project has stable CI and a `pip install` that works at every commit. The rebuild-per-commit model is a poor fit for compiled Zig/C++ backends.
