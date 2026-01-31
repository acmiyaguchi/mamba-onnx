# Mamba ONNX CPU Optimization

Optimized implementations of the Mamba Selective Scan operator for ONNX Runtime (CPU). Fuses the sequential recurrence into a single AVX2-accelerated kernel for significant speedups over standard PyTorch or naive ONNX implementations.

## Features

- **Dual Backend**: Choose between C++ and Zig implementations.
- **AVX2 Acceleration**: Hand-optimized SIMD for fast state updates.
- **Parallel Execution**: Zig backend uses a warm thread pool for multi-core scaling.
- **Operator Fusion**: Replaces thousands of ONNX nodes with a single optimized kernel.
- **4 Op Variants**: SelectiveScan, SelectiveScanExact, SelectiveScanFused, SelectiveScanFusedExact.

## Prerequisites

- **Python**: 3.10+
- **Zig**: 0.14+ (build system and Zig backend)
- **ONNX Runtime**: 1.16+ (installed via pip)
- **C++ compiler** (optional, only for C++ backend): GCC 9+ or Clang with AVX2 support

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mamba-onnx.git
cd mamba-onnx

# Build the Zig backend (recommended)
zig build -Dbackend=zig -Doptimize=ReleaseFast

# Install the Python package
pip install -e ".[dev]"

# Copy the built library into the package
zig build install-py
```

### Building both backends

To build both the C++ and Zig backends for comparison:

```bash
zig build both -Doptimize=ReleaseFast
```

This produces `libmamba_ops_cpp` and `libmamba_ops_zig` in `zig-out/lib/`.

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

Compare C++ and Zig backends (requires both to be built):

```bash
python benchmarks/compare_backends.py
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
