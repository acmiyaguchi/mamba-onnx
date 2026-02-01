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
import sys
from pathlib import Path

__version__ = "0.1.0"

_PKG_DIR = Path(__file__).parent
_BUILD_LIB_DIR = _PKG_DIR.parent.parent / "build" / "lib"

# Search directories: build/lib/ first (dev/editable), then package dir (wheel)
_SEARCH_DIRS = [_BUILD_LIB_DIR, _PKG_DIR]


def available_backends() -> list[str]:
    """List available backends (cpp, zig)."""
    backends = []
    patterns = ["libmamba_ops*.so", "libmamba_ops*.dylib", "libmamba_ops*.dll", "mamba_ops*.dll"]
    seen = set()
    for search_dir in _SEARCH_DIRS:
        if not search_dir.is_dir():
            continue
        for pattern in patterns:
            for path in search_dir.glob(pattern):
                if path.name in seen:
                    continue
                seen.add(path.name)
                if "cpp" in path.name:
                    backends.append("cpp")
                elif "zig" in path.name:
                    backends.append("zig")
    return backends


def get_library_path(backend: str = "auto") -> str:
    """
    Get path to the compiled shared library.

    Args:
        backend: "cpp", "zig", or "auto" (tries zig first, then cpp)

    Returns:
        Path to the shared library

    Raises:
        FileNotFoundError: If no matching library is found
    """
    platform = "linux" if sys.platform.startswith("linux") else sys.platform

    if backend == "cpp":
        candidates = ["libmamba_ops_cpp.so"]
    elif backend == "zig":
        candidates = ["libmamba_ops_zig.so"]
    else:  # auto — prefer zig, then cpp
        candidates = ["libmamba_ops_zig.so", "libmamba_ops_cpp.so"]

    # Adjust for platform
    if platform == "darwin":
        candidates = [c.replace(".so", ".dylib") for c in candidates] + candidates
    elif platform == "win32":
        candidates = [c.replace("lib", "").replace(".so", ".dll") for c in candidates]

    for search_dir in _SEARCH_DIRS:
        for name in candidates:
            path = search_dir / name
            if path.exists():
                return str(path)

    searched = ", ".join(str(d) for d in _SEARCH_DIRS)
    raise FileNotFoundError(
        f"No mamba_ops library found. Searched: {searched}\n"
        f"Build with: zig build install-py -Doptimize=ReleaseFast"
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
