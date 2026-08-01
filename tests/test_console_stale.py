"""The console says when the files under the running scene have moved on.

A scene compiles from more than the file it is named after: the shader files a
pass or a common block points at, and a .py scene's own libraries. All of them
are what "moved on" is measured against. A values sidecar is not one of them,
and neither is a copy that rewrote mtimes without changing a byte. Applying a
working copy leaves a stale scene stale; reloading it is what clears it.
"""

import contextlib
import json
import os
import threading
import time

import pytest

from ghee import sources
from ghee.console import server
from ghee.console.core import ConsoleAPI
from ghee.console.store import SceneStore

CODE = "void main(){ gl_FragColor = vec4(1.0); }"
JSON_CT = {"Content-Type": "application/json"}


def _write(path, text):
    """Write, and push the mtime past whatever a snapshot may already hold: a
    rewrite inside one timestamp tick is otherwise invisible to a stat."""
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def _touch(path):
    """What a file copy leaves behind: a new mtime over identical content."""
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


# what a load records


def test_a_scene_and_the_shader_file_it_names_are_both_recorded(tmp_path):
    (tmp_path / "one.glsl").write_text(CODE, encoding="utf-8")
    scene = tmp_path / "one.scene.json"
    scene.write_text(
        json.dumps({"passes": [{"name": "main", "shader": "one.glsl"}]}),
        encoding="utf-8",
    )
    store = SceneStore(str(tmp_path))
    with sources.collect() as paths:
        store.resolve("one")
    assert paths == {str(scene), str(tmp_path / "one.glsl")}


def test_a_common_block_shader_file_is_recorded(tmp_path):
    (tmp_path / "shared.glsl").write_text("float half(float x){return x*0.5;}", "utf-8")
    scene = tmp_path / "c.scene.json"
    scene.write_text(
        json.dumps(
            {
                "common": {"shader": "shared.glsl"},
                "passes": [{"name": "main", "code": CODE}],
            }
        ),
        encoding="utf-8",
    )
    store = SceneStore(str(tmp_path))
    with sources.collect() as paths:
        store.resolve("c")
    assert paths == {str(scene), str(tmp_path / "shared.glsl")}


PROC = """
from libsnap import common

from ghee import Pass, Scene, SceneController


class C(SceneController):
    def build(self, base):
        return Scene(passes=[Pass(name="main", code=common.CODE)])
"""
LIB = f'CODE = "{CODE}"\n'


def _procedural(tmp_path, package, module="proc"):
    (tmp_path / package).mkdir()
    (tmp_path / package / "common.py").write_text(LIB, encoding="utf-8")
    (tmp_path / f"{module}.py").write_text(
        PROC.replace("libsnap", package), encoding="utf-8"
    )
    return tmp_path / package / "common.py"


def test_a_procedural_scene_records_its_module_and_its_library(tmp_path):
    lib = _procedural(tmp_path, "libsnap")
    store = SceneStore(str(tmp_path))
    with sources.collect() as paths:
        store.resolve("proc")
    assert paths == {str(tmp_path / "proc.py"), str(lib)}


def test_a_values_sidecar_is_not_one_of_a_scenes_sources(tmp_path):
    _procedural(tmp_path, "libside")
    (tmp_path / "modified").mkdir()
    sidecar = tmp_path / "modified" / "proc.values.json"
    sidecar.write_text(json.dumps({"passes": {}}), encoding="utf-8")
    store = SceneStore(str(tmp_path))
    with sources.collect() as paths:
        store.resolve("proc")
    assert str(sidecar) not in paths


def test_nothing_is_recorded_outside_a_collect(tmp_path):
    scene = tmp_path / "one.scene.json"
    scene.write_text(json.dumps({"passes": [{"name": "main", "code": CODE}]}), "utf-8")
    SceneStore(str(tmp_path)).resolve("one")  # no collector: record() is a no-op
    with sources.collect() as paths:
        pass
    assert paths == set()


# what counts as moved on


@pytest.fixture
def watched(tmp_path):
    p = tmp_path / "watched.glsl"
    p.write_text(CODE, encoding="utf-8")
    return p


def test_untouched_files_are_not_stale(watched):
    assert sources.Snapshot([str(watched)], interval=0).stale() is False


def test_changed_content_is_stale(watched):
    snap = sources.Snapshot([str(watched)], interval=0)
    _write(watched, CODE.replace("1.0", "0.5"))
    assert snap.stale() is True


def test_a_copy_that_only_rewrote_the_mtime_is_not_stale(watched):
    snap = sources.Snapshot([str(watched)], interval=0)
    _touch(watched)
    assert snap.stale() is False


def test_a_copied_file_is_not_hashed_twice(watched, monkeypatch):
    """The new mtime is adopted, so a redeployed tree costs one hash, not one
    per poll for as long as the console runs."""
    snap = sources.Snapshot([str(watched)], interval=0)
    _touch(watched)
    assert snap.stale() is False
    calls = []
    monkeypatch.setattr(sources, "_digest", lambda path: calls.append(path))
    assert snap.stale() is False
    assert calls == []


def test_a_file_that_disappeared_is_stale(watched):
    snap = sources.Snapshot([str(watched)], interval=0)
    watched.unlink()
    assert snap.stale() is True


def test_a_scene_with_no_files_behind_it_is_never_stale():
    assert sources.Snapshot([], interval=0).stale() is False


def test_an_unchanged_file_is_never_hashed_again(watched, monkeypatch):
    snap = sources.Snapshot([str(watched)], interval=0)
    calls = []
    monkeypatch.setattr(sources, "_digest", lambda path: calls.append(path))
    for _ in range(5):
        assert snap.stale() is False
    assert calls == []


def test_disk_is_read_at_most_once_an_interval(watched):
    now = [1000.0]
    snap = sources.Snapshot([str(watched)], interval=10.0, clock=lambda: now[0])
    _write(watched, "void main(){ discard; }")
    assert snap.stale() is False  # inside the interval, the last answer stands
    now[0] += 11.0
    assert snap.stale() is True


# the running console


class StubRenderer:
    """A scene whose code says BROKEN is refused the way a compile failure
    refuses one; everything else is a no-op."""

    width, height, components = 64, 48, 4

    error_line = 1
    error_pass = 0

    def setup(self, scene):
        if any("BROKEN" in p.code for p in scene.passes):
            return "0:1(1): error: BROKEN"
        return ""

    def display_vrefresh(self):
        return 0

    def draw(self, provider=None):
        pass

    def render(self, provider=None, out=None, out_size=None):
        return out

    def set_param(self, *a):
        pass

    def set_scene_param(self, *a):
        pass

    def set_dt(self, value):
        pass


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@contextlib.contextmanager
def _console(monkeypatch, scenes_dir):
    monkeypatch.setattr(server, "STORE", SceneStore(str(scenes_dir), read_only=True))
    monkeypatch.setattr(server, "W", 64)
    monkeypatch.setattr(server, "H", 48)
    monkeypatch.setattr(server, "_make_renderer", StubRenderer)
    monkeypatch.setattr(server, "_nclients", 0)
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    # every /stats in a test answers from disk, instead of the poll-rate cache
    monkeypatch.setattr(server, "Snapshot", lambda p: sources.Snapshot(p, interval=0))
    monkeypatch.setitem(server._running, "cur", None)
    monkeypatch.setitem(server._sources, "cur", None)
    server._stopping.clear()
    t = threading.Thread(target=server.render_loop, daemon=True)
    t.start()
    try:
        assert _wait(lambda: server._running["cur"] is not None), "nothing loaded"
        yield ConsoleAPI(server._RenderLoopHost())
    finally:
        server._stopping.set()
        server._q.put(("stop",))
        t.join(timeout=5)
        server._stopping.clear()
        while not server._q.empty():
            server._q.get_nowait()


@pytest.fixture
def scenes(tmp_path):
    (tmp_path / "one.glsl").write_text(CODE, encoding="utf-8")
    (tmp_path / "one.scene.json").write_text(
        json.dumps({"version": 1, "passes": [{"name": "main", "shader": "one.glsl"}]}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def api(monkeypatch, scenes):
    with _console(monkeypatch, scenes) as a:
        yield a


@pytest.fixture
def proc_api(monkeypatch, tmp_path):
    _procedural(tmp_path, "libwatch")
    with _console(monkeypatch, tmp_path) as a:
        yield a


def _stats(api):
    r = api.dispatch("GET", "/stats")
    assert r.status == 200, r.body
    return json.loads(r.body)


def _stale(api):
    return _stats(api)["stale"]


def _scene(api):
    return json.loads(api.dispatch("GET", "/scene", {}).body)


def _apply(api, doc):
    return api.dispatch("POST", "/scene", {}, JSON_CT, json.dumps(doc).encode())


def _reload(api):
    name = server._stat["active"]
    assert api.dispatch("POST", "/switch", {"name": name}).status == 200


def test_stale_is_in_the_stats_payload(api):
    assert "stale" in _stats(api)


def test_a_freshly_loaded_scene_is_not_stale(api):
    assert _stale(api) is False


def test_an_edited_scene_file_turns_the_running_scene_stale(api, scenes):
    _write(
        scenes / "one.scene.json",
        json.dumps({"version": 1, "passes": [{"name": "main", "code": CODE}]}),
    )
    assert _stale(api) is True


def test_an_edited_shader_file_turns_the_running_scene_stale(api, scenes):
    _write(scenes / "one.glsl", "void main(){ discard; }")
    assert _stale(api) is True


def test_a_rewritten_but_identical_scene_directory_stays_current(api, scenes):
    _touch(scenes / "one.scene.json")
    _touch(scenes / "one.glsl")
    assert _stale(api) is False


def test_an_edited_library_turns_a_procedural_scene_stale(proc_api, tmp_path):
    assert _stale(proc_api) is False
    _write(tmp_path / "libwatch" / "common.py", 'CODE = "void main(){ discard; }"\n')
    assert _stale(proc_api) is True


def test_a_written_values_sidecar_does_not_turn_a_scene_stale(proc_api, tmp_path):
    (tmp_path / "modified").mkdir()
    _write(
        tmp_path / "modified" / "proc.values.json",
        json.dumps({"filters": [], "scene": {}, "passes": {}}),
    )
    assert _stale(proc_api) is False


def test_applying_a_working_copy_leaves_a_stale_scene_stale(api, scenes):
    doc = _scene(api)
    _write(scenes / "one.glsl", "void main(){ discard; }")
    assert _stale(api) is True
    doc["passes"][0]["code"] = "void main(){ gl_FragColor = vec4(0.25); }"
    assert _apply(api, doc).status == 200
    assert _wait(lambda: _scene(api)["passes"][0]["code"] == doc["passes"][0]["code"])
    assert _stale(api) is True


def test_reloading_the_scene_clears_it(api, scenes):
    edited = "void main(){ gl_FragColor = vec4(0.25); }"
    _write(scenes / "one.glsl", edited)
    assert _stale(api) is True
    _reload(api)
    assert _wait(lambda: _stale(api) is False)
    assert _scene(api)["passes"][0]["code"] == edited


def test_a_reload_that_does_not_compile_stays_stale(api, scenes):
    _write(scenes / "one.glsl", "BROKEN")
    assert _stale(api) is True
    _reload(api)
    assert _wait(lambda: server._stat["error"])
    assert _stale(api) is True
    assert _scene(api)["passes"][0]["code"] == CODE  # the last good scene is rendering
