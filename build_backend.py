"""Custom build backend that invokes Zig before setuptools."""

import subprocess
import shutil
import sys
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
BUILD_LIB = PROJECT_ROOT / "build" / "lib"
PKG_DIR = PROJECT_ROOT / "src" / "mamba_onnx"


def _lib_ext() -> str:
    if sys.platform == "win32":
        return ".dll"
    elif sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _lib_prefix() -> str:
    return "" if sys.platform == "win32" else "lib"


def _build_native(config_settings: dict | None = None) -> None:
    """Build native code with Zig and install to build/lib/."""
    config_settings = config_settings or {}
    optimize = config_settings.get("optimize", "ReleaseFast")

    cmd = ["zig", "build", "install-py", f"-Doptimize={optimize}"]

    print(f"Building native code: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Zig build failed:\n{result.stderr}")
        raise RuntimeError(f"Zig build failed with exit code {result.returncode}")

    # Copy libs from build/lib/ into src/mamba_onnx/ for wheel packaging
    ext = _lib_ext()
    prefix = _lib_prefix()
    for suffix in ("cpp", "zig"):
        name = f"{prefix}mamba_ops_{suffix}{ext}"
        src = BUILD_LIB / name
        if src.exists():
            dst = PKG_DIR / name
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel, compiling native code first."""
    _build_native(config_settings)
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install, compiling native code first."""
    _build_native(config_settings)
    return _build_editable(wheel_directory, config_settings, metadata_directory)
