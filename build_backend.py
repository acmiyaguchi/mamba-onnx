"""Custom build backend that invokes Zig before setuptools."""

import subprocess
import shutil
from pathlib import Path

# Re-export everything from setuptools.build_meta
from setuptools.build_meta import (
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_wheel,
    build_sdist,
    build_wheel as _build_wheel,
    build_editable as _build_editable,
)

__all__ = [
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
    "build_sdist",
    "build_wheel",
    "build_editable",
]

PROJECT_ROOT = Path(__file__).parent
LIB_DEST = PROJECT_ROOT / "src" / "mamba_onnx"


def _get_lib_name() -> str:
    """Get platform-specific library name."""
    import sys
    if sys.platform == "win32":
        return "mamba_ops.dll"
    elif sys.platform == "darwin":
        return "libmamba_ops.dylib"
    else:
        return "libmamba_ops.so"


def _build_native(config_settings: dict | None = None) -> None:
    """Build native code with Zig."""
    config_settings = config_settings or {}

    # Get backend from config settings (default: zig)
    backend = config_settings.get("backend", "zig")
    optimize = config_settings.get("optimize", "ReleaseFast")

    cmd = ["zig", "build", f"-Dbackend={backend}", f"-Doptimize={optimize}"]

    print(f"Building native code: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Zig build failed:\n{result.stderr}")
        raise RuntimeError(f"Zig build failed with exit code {result.returncode}")

    # Copy built library to package directory
    lib_name = _get_lib_name()
    src_lib = PROJECT_ROOT / "zig-out" / "lib" / lib_name
    dst_lib = LIB_DEST / lib_name

    if not src_lib.exists():
        raise RuntimeError(f"Built library not found at {src_lib}")

    print(f"Copying {src_lib} -> {dst_lib}")
    shutil.copy2(src_lib, dst_lib)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel, compiling native code first."""
    _build_native(config_settings)
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install, compiling native code first."""
    _build_native(config_settings)
    return _build_editable(wheel_directory, config_settings, metadata_directory)
