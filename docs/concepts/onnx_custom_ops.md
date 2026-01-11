# ONNX Custom Operations

## Concept
Standard ONNX models are composed of standard operators (MatMul, Add, Softmax, etc.). When a model uses an operation not supported by the standard set, or when performance requires fusing multiple operations into one, **Custom Operators** are used.

## Architecture

1.  **Schema (The Definition)**: 
    - Defines the operator name (e.g., `mamba.selective_scan`), inputs, outputs, and attributes.
    - Included in the ONNX model file (`.onnx`).
    - Handled by `vendor/onnx` specs.

2.  **Kernel (The Implementation)**:
    - The executable code that performs the math.
    - Implemented in C++/CUDA for the specific inference engine (e.g., ONNX Runtime).
    - compiled into a shared library (e.g., `libmamba_ops.so`).

## Workflow for this Project

### 1. PyTorch Shim
We define a `torch.autograd.Function` in Python. This acts as the "Op" inside PyTorch.
```python
class MambaScanOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, u, delta, A, B, C, D):
        return g.op("mamba::selective_scan", u, delta, A, B, C, D)
```

### 2. Export
When `torch.onnx.export` encounters this function, it calls the `symbolic` method. Instead of tracing the inner Python loop, it emits a single node:
`Node: mamba::selective_scan`

### 3. Runtime (ONNX Runtime)
We must provide a C++ implementation that registers itself with ONNX Runtime.
- **CustomOp Class**: Inherits from `Ort::CustomOpBase`. Implements `Compute()`.
- **Register**: A `RegisterCustomOps` function creates a domain/registry and adds the op.

### 4. Execution
The ONNX Runtime loads the model. When it sees `mamba::selective_scan`, it looks up the kernel in the loaded shared library and executes our optimized C++ code.
