# Technical Design

## Architecture

The project consists of two main execution paths for the MambaBlock: the Control (Slow) path and the Kernel (Fast) path.

### 1. The Control (Baseline)
- **Component**: Single `MambaBlock` from Hugging Face `transformers` (config: `d_model=768`, `d_state=16`).
- **Export Mechanism**: `torch.onnx.export` on the native PyTorch module.
- **Result**: A large ONNX graph where the scan loop is unrolled into ~250+ individual nodes.
- **Runtime**: Standard ONNX Runtime execution, suffering from interpreter overhead for many small ops.

### 2. The Kernel (Optimized)
- **Core Logic**: Implement the recurrence $h_t = A \cdot h_{t-1} + B \cdot x_t$.
- **Implementation**: C++ using AVX2 intrinsics (`_mm256_fmadd_ps`) for vectorized FMA (Fused Multiply-Add) updates.
- **Compilation**: Built as a shared library (`libmamba_ops.so`) using CMake.

### 3. Integration Strategy
- **PyTorch Shim**: A `torch.autograd.Function` named `MambaScanOp` will act as a placeholder in the PyTorch model.
- **ONNX Export**: A symbolic function will be registered to tell `torch.onnx.export` to emit a single `CustomOp` node instead of tracing the internal loop.
- **Inference**: ONNX Runtime will load `libmamba_ops.so` to execute the custom node efficiently.
