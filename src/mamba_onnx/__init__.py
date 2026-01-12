"""
Mamba Selective Scan Custom Op for ONNX Runtime.

Provides optimized AVX2 implementations of the Mamba selective scan operation
with 4 variants for different accuracy/performance tradeoffs:
  - SelectiveScan: Linear discretization (fastest)
  - SelectiveScanExact: Exact exp() discretization
  - SelectiveScanFused: Linear + Softplus/SiLU fusion
  - SelectiveScanFusedExact: Exact + Softplus/SiLU fusion
"""
import os
import sys

__version__ = "0.1.0"

def get_library_path():
    """Returns the path to the compiled shared library."""
    base_path = os.path.dirname(__file__)

    # Platform-specific library names
    if sys.platform == "win32":
        lib_names = ["mamba_ops.dll", "libmamba_ops.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libmamba_ops.dylib", "libmamba_ops.so"]
    else:
        lib_names = ["libmamba_ops.so"]

    # Search paths in order of priority
    search_paths = [base_path]

    # For editable installs, the .so is in site-packages/mamba_onnx/
    # while __file__ points to the source directory
    try:
        import importlib.util
        spec = importlib.util.find_spec("mamba_onnx")
        if spec and spec.submodule_search_locations:
            search_paths.extend(spec.submodule_search_locations)
    except (ImportError, AttributeError):
        pass

    # Also check site-packages directly
    for site_path in sys.path:
        candidate = os.path.join(site_path, "mamba_onnx")
        if os.path.isdir(candidate) and candidate not in search_paths:
            search_paths.append(candidate)

    # Search all paths for the library
    for search_path in search_paths:
        for name in lib_names:
            lib_path = os.path.join(search_path, name)
            if os.path.exists(lib_path):
                return lib_path

    # Return expected path for error message
    return os.path.join(base_path, lib_names[0])


def register_custom_ops(session_options):
    """
    Registers the Mamba custom ops library with ONNX Runtime session options.

    Args:
        session_options: An onnxruntime.SessionOptions instance

    Example:
        >>> import onnxruntime as ort
        >>> import mamba_onnx
        >>> sess_options = ort.SessionOptions()
        >>> mamba_onnx.register_custom_ops(sess_options)
        >>> session = ort.InferenceSession("model.onnx", sess_options)
    """
    lib_path = get_library_path()
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Custom ops library not found at {lib_path}. "
            f"Please ensure the package is properly installed with: pip install ."
        )

    session_options.register_custom_ops_library(lib_path)


# Convenience exports
__all__ = ["register_custom_ops", "get_library_path", "__version__"]
