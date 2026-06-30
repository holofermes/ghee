"""Build the ghee C engine (libghee.so) into the package's lib/.

The build is best effort: on a host without CMake or the GLES/DRM
stack it just prints a warning, so the pure-Python package still installs.
"""

import os
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT = os.path.abspath(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "ghee")  # the importable package dir
ENGINE = os.path.join(PKG, "_engine")  # C source + CMakeLists


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            try:
                self._build(ext)
            except Exception as e:
                print(
                    f"ghee: skipping C engine build ({e}). Rendering will need a "
                    f"prebuilt libghee.so via $GHEE_LIB."
                )

    def _build(self, ext):
        build_tmp = os.path.abspath(self.build_temp)
        os.makedirs(build_tmp, exist_ok=True)
        # Install into ghee/lib/ so the package ships its own .so in-place
        # (editable install) or in the wheel (package_data lib/*.so).
        subprocess.check_call(
            [
                "cmake",
                ext.sourcedir,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={PKG}",
            ],
            cwd=build_tmp,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--", "-j", "4", "install"], cwd=build_tmp
        )


setup(
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "ghee": ["console/web/*", "examples/*.scene.json", "examples/*.py", "lib/*.so"]
    },
    ext_modules=[CMakeExtension("ghee._engine.engine", ENGINE)],
    cmdclass={"build_ext": CMakeBuild},
)
