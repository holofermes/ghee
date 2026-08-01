"""GET /scene answers for the active scene with what is rendering.

A switch loads it from disk, an applied scene that compiled replaces it, a /set
patches the value in it. Any other name still answers from disk, and none of it
writes anything back. The render loop runs for real against a stub renderer:
the queue and the published document are what is under test.
"""

import contextlib
import json
import threading
import time

import pytest

from ghee.console import server
from ghee.console.core import ConsoleAPI
from ghee.console.store import SceneStore

CODE = "void main(){ gl_FragColor = vec4(1.0); }"
ONE = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "code": CODE,
            "params": {
                "amt": {"type": "float", "value": 0.25, "min": 0.0, "max": 1.0},
                "steps": {"type": "int", "value": 2},
                "on": {"type": "bool", "value": False},
            },
        }
    ],
    "params": {"tint": {"type": "float", "value": 0.5}},
}
TWO = {"version": 1, "passes": [{"name": "main", "code": "void main(){ discard; }"}]}
COMMON = "float half(float x){ return x * 0.5; }"
THREE = {
    "version": 1,
    "common": {"code": COMMON},
    "passes": [{"name": "main", "code": CODE}],
}
PROC = """
from ghee import Param, Pass, Scene, SceneController


class C(SceneController):
    def build(self, base):
        return Scene(
            passes=[
                Pass(
                    name="main",
                    code="void main(){ gl_FragColor = vec4(1.0); }",
                    params={"amt": Param("amt", "float", 0.25, 0.0, 1.0, 0.01)},
                )
            ]
        )
"""
JSON_CT = {"Content-Type": "application/json"}


class StubRenderer:
    """A scene whose code says BROKEN is refused the way a compile failure
    refuses one; everything else is a no-op."""

    width, height, components = 64, 48, 4

    error_line = 1
    error_pass = 0

    def setup(self, scene):
        # the common compiles into every pass, so it refuses the same way
        sources = [p.code for p in scene.passes] + [scene.common or ""]
        if any("BROKEN" in s for s in sources):
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
    monkeypatch.setitem(server._running, "cur", None)
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
    (tmp_path / "one.scene.json").write_text(json.dumps(ONE), encoding="utf-8")
    (tmp_path / "two.scene.json").write_text(json.dumps(TWO), encoding="utf-8")
    (tmp_path / "three.scene.json").write_text(json.dumps(THREE), encoding="utf-8")
    return tmp_path


@pytest.fixture
def api(monkeypatch, scenes):
    with _console(monkeypatch, scenes) as a:
        yield a


@pytest.fixture
def proc_api(monkeypatch, tmp_path):
    (tmp_path / "procdemo.py").write_text(PROC, encoding="utf-8")
    (tmp_path / "modified").mkdir()
    (tmp_path / "modified" / "procdemo.values.json").write_text(
        json.dumps({"passes": {"main": {"amt": 0.9}}}), encoding="utf-8"
    )
    with _console(monkeypatch, tmp_path) as a:
        yield a


def _scene(api, name=None):
    r = api.dispatch("GET", "/scene", {"name": name} if name else {})
    assert r.status == 200, r.body
    return json.loads(r.body)


def _amt(doc, name="amt"):
    return doc["passes"][0]["params"][name]["value"]


def _apply(api, doc):
    return api.dispatch("POST", "/scene", {}, JSON_CT, json.dumps(doc).encode())


def _set(api, scope, name, value, ptype="float"):
    r = api.dispatch(
        "POST", "/set", {"pass": scope, "name": name, "type": ptype, "value": value}
    )
    assert r.status == 200, r.body


def _switch(api, name):
    assert api.dispatch("POST", "/switch", {"name": name}).status == 200
    assert _wait(lambda: server._stat["active"] == name)


def _edited(doc, code="void main(){ gl_FragColor = vec4(0.5); }"):
    d = json.loads(json.dumps(doc))
    d["passes"][0]["code"] = code
    return d


# a load


def test_a_freshly_loaded_scene_answers_with_disk(api, scenes):
    assert _scene(api) == json.loads(
        (scenes / "one.scene.json").read_text(encoding="utf-8")
    )


def test_the_no_name_form_answers_for_the_active_scene(api):
    assert _scene(api) == _scene(api, "one")


def test_a_disk_edit_shows_up_on_the_next_switch(api, scenes):
    _switch(api, "two")
    edited = _edited(ONE, "void main(){ gl_FragColor = vec4(0.25); }")
    edited["passes"][0]["params"]["amt"]["value"] = 0.9
    (scenes / "one.scene.json").write_text(json.dumps(edited), encoding="utf-8")
    _switch(api, "one")
    doc = _scene(api)
    assert _amt(doc) == 0.9
    assert doc["passes"][0]["code"] == edited["passes"][0]["code"]


# an applied scene


def test_an_applied_scene_is_what_the_active_name_answers_with(api):
    want = _edited(ONE)
    want["passes"][0]["params"]["amt"]["value"] = 0.6
    assert _apply(api, want).status == 200
    assert _wait(lambda: _scene(api) == want)


def test_a_set_after_an_apply_patches_the_applied_document(api):
    applied = _edited(ONE)
    assert _apply(api, applied).status == 200
    assert _wait(
        lambda: _scene(api)["passes"][0]["code"] == applied["passes"][0]["code"]
    )
    _set(api, "0", "amt", "0.75")
    assert _wait(lambda: _amt(_scene(api)) == 0.75)
    assert _scene(api)["passes"][0]["code"] == applied["passes"][0]["code"]


def test_an_applied_scene_that_fails_to_compile_leaves_the_running_document(api):
    running = _scene(api)
    assert _apply(api, _edited(ONE, "void main(){ BROKEN }")).status == 200
    assert _wait(lambda: server._stat["error"])
    assert _scene(api) == running


# the common block


def test_a_scene_with_a_common_serves_it(api, scenes):
    assert _scene(api, "three") == json.loads(
        (scenes / "three.scene.json").read_text(encoding="utf-8")
    )


def test_an_applied_common_is_what_the_active_name_answers_with(api):
    _switch(api, "three")
    want = json.loads(json.dumps(THREE))
    want["common"] = {"code": "float half(float x){ return x * 0.25; }"}
    assert _apply(api, want).status == 200
    assert _wait(lambda: _scene(api) == want)


def test_a_common_added_to_the_running_scene_shows_up(api):
    want = json.loads(json.dumps(ONE))
    want["common"] = {"code": COMMON}
    assert _apply(api, want).status == 200
    assert _wait(lambda: _scene(api).get("common") == {"code": COMMON})


def test_an_uncompilable_common_leaves_the_running_document(api):
    _switch(api, "three")
    running = _scene(api)
    broken = json.loads(json.dumps(THREE))
    broken["common"] = {"code": "BROKEN"}
    assert _apply(api, broken).status == 200
    assert _wait(lambda: server._stat["error"])
    assert _scene(api) == running


def test_a_deleted_common_leaves_the_document_without_one(api):
    _switch(api, "three")
    assert _wait(lambda: "common" in _scene(api))
    without = {k: v for k, v in THREE.items() if k != "common"}
    assert _apply(api, without).status == 200
    assert _wait(lambda: "common" not in _scene(api))


# a live param


def test_a_set_lands_where_the_editor_reads_the_value(api):
    _set(api, "0", "amt", "0.75")
    assert _wait(lambda: _amt(_scene(api)) == 0.75)
    doc = _scene(api)
    assert doc["passes"][0]["params"]["amt"]["max"] == 1.0  # metadata survives
    assert doc["passes"][0]["code"] == CODE


def test_a_scene_scoped_set_lands_in_the_scene_params(api):
    _set(api, "scene", "tint", "0.125")
    assert _wait(lambda: _scene(api)["params"]["tint"]["value"] == 0.125)


@pytest.mark.parametrize(
    "ptype,name,raw,want",
    [("int", "steps", "2.6", 3), ("bool", "on", "true", True)],
)
def test_a_set_records_the_coerced_value_not_the_wire_value(
    api, ptype, name, raw, want
):
    _set(api, "0", name, raw, ptype)
    assert _wait(lambda: _amt(_scene(api), name) == want)
    assert isinstance(_amt(_scene(api), name), type(want))


def test_a_set_the_scene_has_no_param_for_changes_nothing(api):
    running = _scene(api)
    _set(api, "0", "nosuch", "0.75")
    assert not _wait(lambda: _scene(api) != running, timeout=0.5)


# every other name


def test_another_scene_still_answers_from_disk(api, scenes):
    _set(api, "0", "amt", "0.75")
    assert _wait(lambda: _amt(_scene(api)) == 0.75)
    assert _scene(api, "two") == json.loads(
        (scenes / "two.scene.json").read_text(encoding="utf-8")
    )


def test_switching_away_and_back_answers_from_disk_again(api):
    _set(api, "0", "amt", "0.75")
    assert _wait(lambda: _amt(_scene(api)) == 0.75)
    _switch(api, "two")
    _switch(api, "one")
    assert _amt(_scene(api)) == 0.25


def test_an_unknown_name_is_still_404(api):
    assert api.dispatch("GET", "/scene", {"name": "nope"}).status == 404


def test_nothing_that_edits_the_running_scene_writes_to_disk(api, scenes):
    before = (scenes / "one.scene.json").read_text(encoding="utf-8")
    assert _apply(api, _edited(ONE)).status == 200
    _set(api, "0", "amt", "0.75")
    assert _wait(lambda: _amt(_scene(api)) == 0.75)
    assert (scenes / "one.scene.json").read_text(encoding="utf-8") == before
    assert not (scenes / "modified").exists()


# a procedural scene


def test_a_procedural_scene_serves_what_its_load_built(proc_api):
    assert _amt(_scene(proc_api)) == 0.9  # the sidecar over the code's 0.25


def test_a_procedural_scene_takes_a_set_into_its_document(proc_api):
    _set(proc_api, "0", "amt", "0.4")
    assert _wait(lambda: _amt(_scene(proc_api)) == 0.4)


def test_a_scene_that_refuses_to_save_still_serves(proc_api):
    body = json.dumps({"name": "procdemo", "scene": ONE}).encode()
    assert proc_api.dispatch("POST", "/save", {}, JSON_CT, body).status == 403
    assert _scene(proc_api)["passes"][0]["params"]["amt"]["value"] == 0.9
