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

```bash
# Clone the repository
git clone https://github.com/your-repo/mamba-onnx.git
cd mamba-onnx

# Build both backends (C++ and Zig)
zig build -Doptimize=ReleaseFast

# Install the Python package
pip install -e ".[dev]"

# Copy the built library into the package
zig build install-py
```

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
# Compare all ops x both backends vs PyTorch reference
python benchmarks/suite.py --mode compare --backend both --reps 100

# Scaling analysis (L and D sweeps) with both backends
python benchmarks/suite.py --mode scale --backend both --reps 100

# Regenerate plots from CSV data
python benchmarks/plot.py

# Profile MambaBlock component breakdown
python benchmarks/profiler.py
```

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
