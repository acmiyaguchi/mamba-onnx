# Mamba ONNX CPU Optimization

This project provides a highly optimized C++ implementation of the Mamba Selective Scan operator, specifically designed for ONNX Runtime (CPU). By fusing the sequential recurrence into a single AVX2-accelerated kernel, it achieves significant speedups compared to standard PyTorch or naive ONNX implementations.

## Features

- **AVX2 Acceleration**: Hand-optimized intrinsics for fast state updates.
- **Operator Fusion**: Replaces the expensive $O(L)$ loop in Python/ONNX with a single C++ node.
- **Easy Integration**: Package-based installation with automatic ONNX Runtime registration.
- **Verified Correctness**: Includes a benchmark suite that validates outputs against PyTorch's reference implementation.

## Prerequisites

- **Python**: 3.10+
- **Compiler**: GCC 9+ or Clang (supporting AVX2 and OpenMP)
- **CMake**: 3.18+

## Installation

The project uses `scikit-build-core` to manage the C++ compilation. You can install it in editable mode for development:

```bash
# Clone the repository
git clone https://github.com/your-repo/mamba-onnx.git
cd mamba-onnx

# Install in editable mode (triggers CMake build)
pip install -e .
```

## Usage

Registering the custom operator with ONNX Runtime is handled automatically by the `mamba_onnx` package:

```python
import onnxruntime as ort
import mamba_onnx

# Create session options
sess_options = ort.SessionOptions()

# Register the custom op
mamba_onnx.register_custom_ops(sess_options)

# Load your model (e.g., exported with the custom op)
session = ort.InferenceSession("mamba_fused.onnx", sess_options, providers=['CPUExecutionProvider'])
```

## Benchmarking Results

By fusing the sequential recurrence into a single AVX2-accelerated kernel, we achieve over **100x speedup** compared to standard PyTorch and unrolled ONNX implementations.

### Performance Comparison ($L=128, D=768$)

| Method | Latency | Speedup vs PyTorch |
| :--- | :--- | :--- |
| **PyTorch Eager** | 9.59 ms | 1.0x |
| **Baseline ONNX** (Standard Ops) | 6.61 ms | ~1.45x |
| **Fused ONNX** (Custom Kernel) | **0.08 ms** | **~122x** |

### Why it's faster
1. **Operator Fusion**: Replaces thousands of small ONNX nodes with one optimized kernel, eliminating dispatch overhead.
2. **Register-Level State Updates**: Keeps intermediate states in CPU registers rather than writing back to main memory.
3. **AVX2 SIMD**: Processes 8 channels simultaneously using 256-bit vector instructions.

For detailed analysis, see [docs/results.md](docs/results.md).

## Installation

Run the test suite to ensure everything is working correctly:

```bash
pytest tests/
```

## Project Structure

- `csrc/`: C++ source code for the optimized Selective Scan kernel.
- `src/mamba_onnx/`: Python package wrapper and library loader.
- `benchmarks/`: Performance measurement scripts.
- `tests/`: Correctness verification tests.
- `vendor/`: Vendored headers for ONNX Runtime.

## License

[MIT](LICENSE)