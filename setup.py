"""Build the ghee C engine (libghee.so) into the package's lib/.

A failed build fails the install. $GHEE_LIB (render with an engine already on
the host) and GHEE_SKIP_ENGINE=1 (install without rendering) opt out.
"""

import glob
import os
import shutil
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT = os.path.abspath(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "ghee")  # the importable package dir
ENGINE = os.path.join(PKG, "_engine")  # C source + CMakeLists

APT = (
    "sudo apt install cmake pkg-config libdrm-dev libgbm-dev "
    "libegl1-mesa-dev libgles2-mesa-dev"
)


def _build_failed(e):
    return f"""
ghee: C engine build failed: {type(e).__name__}: {e}

libghee.so is what renders, so this install would not be able to.
Install the toolchain and the GLES/DRM headers, then install ghee
again. On Raspberry Pi OS or Debian:

    {APT}

To install without building the engine:

    GHEE_LIB=/path/to/libghee.so   render with an engine already on this host
    GHEE_SKIP_ENGINE=1             no engine, no rendering (ghee.scene, ghee.isf)
"""


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        if os.environ.get("GHEE_SKIP_ENGINE"):
            print(
                "ghee: GHEE_SKIP_ENGINE set, no C engine, this install cannot render."
            )
            return
        for ext in self.extensions:
            try:
                self._build(ext)
            except Exception as e:
                lib = os.environ.get("GHEE_LIB")
                if not lib:
                    raise SystemExit(_build_failed(e)) from e
                print(
                    f"ghee: C engine build failed ({e}), rendering with $GHEE_LIB={lib}."
                )

    def _build(self, ext):
        build_tmp = os.path.abspath(self.build_temp)
        os.makedirs(build_tmp, exist_ok=True)
        # Install into ghee/lib/ so an editable install finds the engine in place.
        subprocess.check_call(
            [
                "cmake",
                ext.sourcedir,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={PKG}",
            ],
            cwd=build_tmp,
        )
        # Parallel -O3 compiles exhaust a 512MB board, so the level stays
        # overridable through CMake's own env var.
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or os.cpu_count() or 1
        subprocess.check_call(
            ["cmake", "--build", ".", "--", "-j", str(jobs), "install"], cwd=build_tmp
        )
        self._stage(os.path.join(PKG, "lib"))

    def _stage(self, libdir):
        """build_py runs before build_ext, so package_data collected lib/*.so when
        it was still empty. Put the engine in the wheel tree here instead."""
        dest = os.path.abspath(os.path.join(self.build_lib, "ghee", "lib"))
        if dest == os.path.abspath(libdir):
            return
        os.makedirs(dest, exist_ok=True)
        for so in glob.glob(os.path.join(libdir, "libghee*.so")):
            shutil.copy2(so, dest)


setup(
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "ghee": [
            "console/web/*",
            "examples/*.scene.json",
            "examples/*.glsl",
            "examples/*.py",
            "filters/*.filter.json",
            "lib/*.so",
        ]
    },
    ext_modules=[CMakeExtension("ghee._engine.engine", ENGINE)],
    cmdclass={"build_ext": CMakeBuild},
)
