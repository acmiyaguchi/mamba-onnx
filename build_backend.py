"""Custom build backend that invokes Zig before setuptools."""

import os
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
BUILD_LIB = PROJECT_ROOT / "build" / "native"
PKG_DIR = PROJECT_ROOT / "src" / "mamba_onnx"


def _target_os(config_settings: dict | None = None) -> str:
    """Determine the target OS from config settings or sys.platform."""
    config_settings = config_settings or {}
    target = config_settings.get("target")
    if target:
        if "macos" in target:
            return "darwin"
        elif "windows" in target:
            return "win32"
        return "linux"
    return sys.platform


def _lib_ext(config_settings: dict | None = None) -> str:
    target_os = _target_os(config_settings)
    if target_os == "win32":
        return ".dll"
    elif target_os == "darwin":
        return ".dylib"
    return ".so"


def _lib_prefix(config_settings: dict | None = None) -> str:
    target_os = _target_os(config_settings)
    return "" if target_os == "win32" else "lib"


def _build_native(config_settings: dict | None = None) -> None:
    """Build native code with Zig and install to build/lib/."""
    config_settings = config_settings or {}
    optimize = config_settings.get("optimize", "ReleaseFast")
    target = config_settings.get("target")
    cpu = config_settings.get("cpu")

    cmd = ["zig", "build", "install-py", f"-Doptimize={optimize}"]
    if target:
        cmd.append(f"-Dtarget={target}")
    if cpu:
        cmd.append(f"-Dcpu={cpu}")

    print(f"Building native code: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Zig build failed:\n{result.stderr}")
        raise RuntimeError(f"Zig build failed with exit code {result.returncode}")

    # Copy libs from build/lib/ into src/mamba_onnx/ for wheel packaging
    ext = _lib_ext(config_settings)
    prefix = _lib_prefix(config_settings)
    for suffix in ("cpp", "zig"):
        name = f"{prefix}mamba_ops_{suffix}{ext}"
        src = BUILD_LIB / name
        if src.exists():
            dst = PKG_DIR / name
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)


def _retag_wheel(wheel_dir: str, orig_name: str, plat_name: str) -> str:
    """Retag a wheel file with the given platform using `wheel tags`."""
    wheel_path = Path(wheel_dir) / orig_name
    plat_tag = plat_name.replace("-", "_").replace(".", "_")
    result = subprocess.run(
        ["wheel", "tags", "--platform-tag", plat_tag,
         "--remove", str(wheel_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wheel tags failed:\n{result.stderr}")

    # Parse new filename from stdout (e.g. "path/to/new.whl")
    new_path = Path(result.stdout.strip())
    return new_path.name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel, compiling native code first."""
    config_settings = config_settings or {}
    _build_native(config_settings)
    whl_name = _build_wheel(wheel_directory, config_settings, metadata_directory)

    plat_name = config_settings.get("plat-name")
    if plat_name:
        whl_name = _retag_wheel(wheel_directory, whl_name, plat_name)

    return whl_name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install, compiling native code first."""
    _build_native(config_settings)
    return _build_editable(wheel_directory, config_settings, metadata_directory)
