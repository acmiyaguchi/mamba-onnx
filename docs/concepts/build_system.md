# Build System Architecture

## Overview
The project requires a build system to compile the C++ custom operator into a shared library (`.so` or `.dll`) that can be loaded by ONNX Runtime. We will use **CMake**.

## Components

### 1. CMake (The Builder)
- **Tool**: `CMake` (version 3.18+ recommended).
- **Config**: `CMakeLists.txt` at the root (or `src/`).
- **Dependencies**: 
    - `OnnxRuntime` headers (often provided via a pip install or downloaded).
    - `PyTorch` (optional, strictly for ABI compatibility if mixing, but for pure ONNX Runtime custom ops, we just need ORT headers).
    - `AVX2` compiler flags (`-mavx2 -mfma`).

### 2. The Artifact
- **Target**: `libmamba_ops.so` (Linux) or `mamba_ops.dll` (Windows).
- **Type**: Shared Library (`SHARED`).

### 3. Source Structure
```
src/
  ├── selective_scan.cc    # The Kernel Logic & ORT Wrapper
  └── CMakeLists.txt       # Build instructions
```

### 4. Integration with Python
We do not need a complex `setup.py` extension build (like PyBind11) because ONNX Runtime has a native mechanism to load custom ops from a raw shared library path.

**Python Usage:**
```python
import onnxruntime as ort

so = ort.SessionOptions()
# Load the compiled library directly
so.register_custom_ops_library("./lib/libmamba_ops.so")

sess = ort.InferenceSession("model.onnx", so)
```

## Compilation Steps (Manual)
1. `mkdir build && cd build`
2. `cmake ..`
3. `make`
4. `cp libmamba_ops.so ../lib/`

## Compilation Steps (Automated)
We can add a simple `build.sh` script or a Python helper to run these commands for the user.
