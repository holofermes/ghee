"""Packaging tests: what the sdist actually carries, what the metadata
promises, and what setup.py does when the engine cannot be built. No GL, and
the engine build here only ever runs far enough to fail."""

import os
import subprocess
import sys
import tarfile
import tomllib
from importlib.metadata import entry_points, version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIST = "ghee-render"


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def sdist_names(tmp_path_factory):
    """Roll a real sdist and return its member paths, relative to the archive root."""
    if not (ROOT / "setup.py").exists():
        pytest.skip("not a source checkout")
    pytest.importorskip("build", reason="pip install -e '.[test]'")
    out = tmp_path_factory.mktemp("sdist")
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation", "-o", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tars = list(out.glob("*.tar.gz"))
    assert len(tars) == 1, tars
    with tarfile.open(tars[0]) as t:
        names = t.getnames()
    prefix = names[0].split("/")[0] + "/"
    return sorted(n[len(prefix) :] for n in names if n.startswith(prefix))


def test_sdist_excludes_the_test_suite(sdist_names):
    """Not useful to someone installing from PyPI, and not usable either: the
    suite needs conftest.py, helpers.py and the goldens, and the sdist default
    carries the test modules alone, which cannot even be collected."""
    assert not [n for n in sdist_names if n == "tests" or n.startswith("tests/")]


def test_sdist_has_no_prebuilt_engine(sdist_names):
    """A .so from the build host is the wrong architecture for whoever unpacks it."""
    assert not [n for n in sdist_names if n.endswith(".so")]


@pytest.mark.parametrize(
    "member",
    [
        "LICENSE",
        "README.md",
        "MANIFEST.in",
        "pyproject.toml",
        "setup.py",
        "ghee/console/web/index.html",
        "ghee/console/web/h264client.js",
        "ghee/console/web/sw.js",
        "ghee/examples/bars.scene.json",
        "ghee/examples/bars.glsl",
        "ghee/examples/text.py",
        "ghee/filters/dither.filter.json",
        "ghee/_engine/engine.c",
        "ghee/_engine/engine.h",
        "ghee/_engine/CMakeLists.txt",
    ],
)
def test_sdist_carries(sdist_names, member):
    assert member in sdist_names


def test_sdist_carries_every_bundled_example_and_filter(sdist_names):
    """Whatever is in the tree ships, so a new example cannot be silently dropped."""
    for d, suffixes in (
        ("ghee/examples", (".scene.json", ".glsl", ".py")),
        ("ghee/filters", (".filter.json",)),
        ("ghee/console/web", ()),
    ):
        on_disk = {
            f"{d}/{p.name}"
            for p in (ROOT / d).iterdir()
            if p.is_file() and (not suffixes or p.name.endswith(suffixes))
        }
        assert on_disk <= set(sdist_names), on_disk - set(sdist_names)


def test_sdist_carries_every_engine_source(sdist_names):
    """setup.py compiles ghee/_engine from the sdist, so a new .c cannot be left out."""
    on_disk = {
        f"ghee/_engine/{p.name}"
        for p in (ROOT / "ghee" / "_engine").iterdir()
        if p.suffix in (".c", ".h") or p.name == "CMakeLists.txt"
    }
    assert on_disk <= set(sdist_names), on_disk - set(sdist_names)


def test_console_scripts_resolve():
    """Both names the README tells people to run, pointing at a real callable."""
    declared = _pyproject()["project"]["scripts"]
    assert set(declared) == {"ghee", "ghee-render"}
    for name, target in declared.items():
        eps = [e for e in entry_points(group="console_scripts") if e.name == name]
        assert eps, (
            f"pyproject declares '{name}' but the installed {DIST} does not "
            f"provide it, so the resolved metadata predates the declaration"
        )
        assert eps[0].value.replace(" ", "") == target.replace(" ", "")
        assert callable(eps[0].load())


def test_bundled_drivers_are_listed_by_name():
    """`ghee displays` reads this; third-party panels join through the
    ghee.drivers entry point group."""
    from ghee.drivers import available_drivers

    names = available_drivers()
    assert {"st7789", "ssd1306"} <= set(names)
    entry_points(group="ghee.drivers")


def test_declared_extras_match_the_readme():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert set(extras) == {"drivers", "test"}
    assert any(d.lower().startswith("spidev") for d in extras["drivers"])


def test_runtime_deps_have_floors():
    for dep in _pyproject()["project"]["dependencies"]:
        assert any(op in dep for op in (">=", "==", "~=")), dep


def test_version_matches_the_installed_distribution():
    import ghee

    assert ghee.__version__ == version(DIST) == _pyproject()["project"]["version"]


def _build_ext(tmp_path, **env):
    """setup.py's build_ext with an empty PATH, so cmake is missing the way it
    is on a host without the toolchain. Build products go to tmp_path."""
    if not (ROOT / "setup.py").exists():
        pytest.skip("not a source checkout")
    (tmp_path / "bin").mkdir(exist_ok=True)
    e = {
        k: v for k, v in os.environ.items() if k not in ("GHEE_LIB", "GHEE_SKIP_ENGINE")
    }
    e["PATH"] = str(tmp_path / "bin")
    e.update(env)
    return subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "-b",
            str(tmp_path / "lib"),
            "-t",
            str(tmp_path / "temp"),
        ],
        cwd=ROOT,
        env=e,
        capture_output=True,
        text=True,
    )


def test_a_failed_engine_build_fails_the_install(tmp_path):
    """A package that installs without libghee.so cannot render, and the user
    only finds out at the first render."""
    r = _build_ext(tmp_path)
    assert r.returncode != 0
    assert "C engine build failed" in r.stderr


def test_the_failure_names_both_escapes_and_the_headers(tmp_path):
    """Fixable from the message alone, without the README."""
    err = _build_ext(tmp_path).stderr
    for expected in (
        "GHEE_LIB=",
        "GHEE_SKIP_ENGINE=1",
        "sudo apt install cmake pkg-config libdrm-dev libgbm-dev "
        "libegl1-mesa-dev libgles2-mesa-dev",
    ):
        assert expected in err, err


def test_skipping_the_engine_installs(tmp_path):
    """Installing for ghee.scene / ghee.isf alone, with no rendering."""
    r = _build_ext(tmp_path, GHEE_SKIP_ENGINE="1")
    assert r.returncode == 0, r.stderr
    assert "GHEE_SKIP_ENGINE set" in r.stdout


def test_a_prebuilt_engine_installs(tmp_path):
    """$GHEE_LIB is the fleet's flow: the Python side installs anywhere and
    renders with the engine already on the box."""
    r = _build_ext(tmp_path, GHEE_LIB="/opt/libghee.so")
    assert r.returncode == 0, r.stderr
    assert "/opt/libghee.so" in r.stdout
