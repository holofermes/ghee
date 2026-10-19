"""The `ghee` command that the wheel installs."""

import pathlib
import subprocess
import sys

import pytest
from conftest import REPO

GHEE = pathlib.Path(sys.executable).parent / "ghee"
CLI = REPO / "target" / "release" / "ghee"


def run(command, *argv):
    return subprocess.run(
        [str(command), *argv], capture_output=True, text=True, check=False, timeout=60
    )


def test_the_wheel_installs_the_command_line(tmp_path):
    """The wheel's ghee answers as the build's binary does, status included."""
    calls = [["--help"], ["render", str(tmp_path / "missing.scene.json")]]
    got = [run(GHEE, *argv) for argv in calls]
    assert [r.returncode for r in got] == [0, 1]
    if not CLI.exists():
        pytest.skip(f"no {CLI} to compare against")
    for argv, ran in zip(calls, got):
        binary = run(CLI, *argv)
        assert (ran.returncode, ran.stdout, ran.stderr) == (
            binary.returncode,
            binary.stdout,
            binary.stderr,
        )


def test_the_command_imports_nothing_past_the_native_module():
    """Every module that `ghee` imports loads again at each start: the scene
    shim's imports and the signal module took 230 ms on a Pi Zero 2 W."""

    def modules(code, *argv):
        probe = f"import sys; {code}print(*sys.modules, file=sys.stderr)"
        ran = subprocess.run(
            [sys.executable, "-c", probe, *argv],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return set(ran.stderr.split())

    bare = modules("")
    ran = modules("import ghee._cli; ghee._cli.main(); ", "--version")
    assert ran - bare == {"ghee", "ghee._cli", "ghee._native"}
