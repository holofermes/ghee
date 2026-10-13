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
