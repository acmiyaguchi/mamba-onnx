# Mamba ONNX CPU Optimization

Optimized implementations of the Mamba Selective Scan operator for ONNX Runtime (CPU). Fuses the sequential recurrence into a single AVX2-accelerated kernel for significant speedups over standard PyTorch or naive ONNX implementations.

## Features

- **Dual Backend**: C++ (OpenMP) and Zig (std.Thread.Pool) implementations, both multi-threaded.
- **AVX2 Acceleration**: Hand-optimized SIMD for fast state updates.
- **Parallel Execution**: Both backends parallelize across batch and dimension. C++ uses OpenMP; Zig uses std.Thread.Pool (or optional ForkJoin with `-Dforkjoin=true`).
- **Operator Fusion**: Replaces thousands of ONNX nodes with a single optimized kernel.
- **4 Op Variants**: SelectiveScan, SelectiveScanExact, SelectiveScanFused, SelectiveScanFusedExact.

## Prerequisites

- **Python**: 3.10+
- **Zig**: 0.14+ (build system and Zig backend)
- **ONNX Runtime**: 1.16+ (installed via pip)
- **C++ compiler** (optional, only for C++ backend): GCC 9+ or Clang with AVX2 support
- **libomp-dev** (for C++ backend OpenMP): `apt install libomp-dev`

## Installation

Install directly from the latest [GitHub release](https://github.com/acmiyaguchi/mamba-onnx/releases/latest):

```bash
# Linux x86_64
uv pip install "mamba-onnx @ https://github.com/acmiyaguchi/mamba-onnx/releases/latest/download/mamba_onnx-0.1.0-py3-none-manylinux_2_17_x86_64.whl"

# macOS x86_64
uv pip install "mamba-onnx @ https://github.com/acmiyaguchi/mamba-onnx/releases/latest/download/mamba_onnx-0.1.0-py3-none-macosx_11_0_x86_64.whl"
```

Or with pip:

```bash
pip install "https://github.com/acmiyaguchi/mamba-onnx/releases/latest/download/mamba_onnx-0.1.0-py3-none-manylinux_2_17_x86_64.whl"
```

The C++ backend requires `libomp` at runtime (`apt install libomp-dev` on Debian/Ubuntu). The Zig backend has no extra runtime dependencies.

### From source

```bash
git clone https://github.com/acmiyaguchi/mamba-onnx.git
cd mamba-onnx
pip install -e ".[dev]"
```

This requires Zig 0.14+ and (optionally) `libomp-dev` for the C++ backend.

## Usage

```python
import onnxruntime as ort
import mamba_onnx

# Create session options and register custom ops
sess_options = ort.SessionOptions()
mamba_onnx.register_custom_ops(sess_options, backend="zig")  # or "cpp", "auto"

# Load your model
session = ort.InferenceSession("model.onnx", sess_options, providers=['CPUExecutionProvider'])
```

### Selecting a backend

```python
# List available backends
print(mamba_onnx.available_backends())  # e.g. ['cpp', 'zig']

# Use a specific backend
mamba_onnx.register_custom_ops(sess_options, backend="zig")
```

## Testing

```bash
# Run tests with pytest
pytest tests/

# Or run standalone
python tests/test_ops.py
```

## Benchmarking

```bash
# Run compare-mode benchmarks
uv run pytest benchmarks/ --benchmark-only -k "not scale"

# Run all benchmarks (compare + scale sweeps)
uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against a saved baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=0001_baseline

# Pin thread count
uv run pytest benchmarks/ --benchmark-only --threads=4

# Profile MambaBlock components (unchanged)
uv run python benchmarks/profiler.py

# Generate plots from saved results
uv run python benchmarks/plot.py
```

## Building Wheels

The Docker build cross-compiles wheels for Linux and macOS (both x86_64, AVX2 required):

```bash
# Build both wheels into dist/
docker build --target artifacts -o dist .

# Inspect contents
unzip -l dist/*.whl
```

Wheels are tagged `manylinux_2_17_x86_64` (Linux) and `macosx_11_0_x86_64` (macOS).

To change the Zig version:

```bash
docker build --build-arg ZIG_VERSION=0.15.1 --target artifacts -o dist .
```

## Releasing

Releases are automated via GitHub Actions (`.github/workflows/release.yml`). The workflow builds wheels, runs tests, and uploads them to GitHub Releases.

**To make a release:**

```bash
# 1. Update the version in pyproject.toml
# 2. Commit and tag
git add pyproject.toml
git commit -m "release: v0.1.0"
git tag v0.1.0

# 3. Push the tag — this triggers the release workflow
git push origin v0.1.0
```

The workflow will:
1. Build Linux + macOS wheels via Docker
2. Install the Linux wheel and run `pytest tests/`
3. If tests pass and the trigger is a `v*` tag, upload wheels to a GitHub Release

**To test the workflow without releasing**, use the manual trigger:
Actions → Build and Release → Run workflow. This runs build + test but skips the release step.

## Project Structure

- `src/native/zig/`: Zig implementation of the selective scan kernel.
- `src/native/cpp/`: C++ implementation of the selective scan kernel.
- `src/mamba_onnx/`: Python package wrapper and library loader.
- `vendor/onnxruntime/`: Vendored ONNX Runtime C API headers.
- `benchmarks/`: Performance comparison scripts.
- `tests/`: Correctness verification tests.
- `build.zig`: Zig build system configuration.
- `build_backend.py`: Python build backend for `pip install`.

## License

[MIT](LICENSE)
