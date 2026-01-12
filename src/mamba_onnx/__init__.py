"""
Mamba Selective Scan Custom Op for ONNX Runtime.

Provides optimized implementations of the Mamba selective scan operation
with 4 variants for different accuracy/performance tradeoffs:
  - SelectiveScan: Linear discretization (fastest)
  - SelectiveScanExact: Exact exp() discretization
  - SelectiveScanFused: Linear + Softplus/SiLU fusion
  - SelectiveScanFusedExact: Exact + Softplus/SiLU fusion

Build with: zig build -Doptimize=ReleaseFast
"""
import os
import sys
from pathlib import Path

__version__ = "0.1.0"

_LIB_DIR = Path(__file__).parent

# Library names by platform
_LIB_NAMES = {
    "win32": ["mamba_ops.dll", "libmamba_ops.dll"],
    "darwin": ["libmamba_ops.dylib", "libmamba_ops.so"],
    "linux": ["libmamba_ops.so"],
}


def available_backends() -> list[str]:
    """List available backends (cpp, zig)."""
    backends = []
    for name in _LIB_DIR.glob("libmamba_ops*.so"):
        if "cpp" in name.name:
            backends.append("cpp")
        elif "zig" in name.name:
            backends.append("zig")
        elif name.name == "libmamba_ops.so":
            # Default library - could be either
            if "cpp" not in backends and "zig" not in backends:
                backends.append("default")
    return backends


def get_library_path(backend: str = "auto") -> str:
    """
    Get path to the compiled shared library.

    Args:
        backend: "cpp", "zig", or "auto" (tries default, then cpp, then zig)

    Returns:
        Path to the shared library

    Raises:
        FileNotFoundError: If no matching library is found
    """
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    lib_names = _LIB_NAMES.get(platform, _LIB_NAMES["linux"])

    # Backend-specific library names
    if backend == "cpp":
        candidates = ["libmamba_ops_cpp.so", "libmamba_ops.so"]
    elif backend == "zig":
        candidates = ["libmamba_ops_zig.so", "libmamba_ops.so"]
    else:  # auto
        candidates = ["libmamba_ops.so", "libmamba_ops_cpp.so", "libmamba_ops_zig.so"]

    # Adjust for platform
    if platform == "darwin":
        candidates = [c.replace(".so", ".dylib") for c in candidates] + candidates
    elif platform == "win32":
        candidates = [c.replace("lib", "").replace(".so", ".dll") for c in candidates]

    # Search in package directory
    for name in candidates:
        path = _LIB_DIR / name
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"No mamba_ops library found in {_LIB_DIR}. "
        f"Build with: zig build -Doptimize=ReleaseFast"
    )


def register_custom_ops(session_options, backend: str = "auto") -> str:
    """
    Register Mamba custom ops with ONNX Runtime session options.

    Args:
        session_options: An onnxruntime.SessionOptions instance
        backend: "cpp", "zig", or "auto"

    Returns:
        Path to the library that was loaded (for logging)

    Example:
        >>> import onnxruntime as ort
        >>> import mamba_onnx
        >>> opts = ort.SessionOptions()
        >>> lib = mamba_onnx.register_custom_ops(opts, backend="zig")
        >>> print(f"Using: {lib}")
        >>> session = ort.InferenceSession("model.onnx", opts)
    """
    lib_path = get_library_path(backend)
    session_options.register_custom_ops_library(lib_path)
    return lib_path


__all__ = [
    "register_custom_ops",
    "get_library_path",
    "available_backends",
    "__version__",
]
