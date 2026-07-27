"""The console core with no server under it: what ConsoleAPI routes onto a
host, what it refuses before a host is ever reached, and what the scene store
reads and writes. No sockets, no GL, no threads."""

import json
import os

import pytest

from ghee import Param, Pass, Scene
from ghee.console.core import (
    MAX_BODY,
    ConsoleAPI,
    ConsoleHost,
    NotFound,
    ServerError,
    origin_allowed,
    parse_param_value,
    parse_query,
    parse_scope,
    static_path,
)
from ghee.console.store import SceneStore

SCENE = {
    "version": 1,
    "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}],
}
JSON = {"Content-Type": "application/json"}


class FakeHost(ConsoleHost):
    """A host that records instead of rendering."""

    writable = False
    save_refused = "nope"

    def __init__(self, names=("a", "b", "c"), active="a"):
        self.names = list(names)
        self.active = active
        self.calls = []

    def stats(self):
        return {"fps": 12.0, "active": self.active}

    def scene_names(self):
        return list(self.names)

    def active_name(self):
        return self.active

    def scene_def(self, name=None):
        self.calls.append(("scene_def", name))
        if name is not None and name not in self.names:
            raise NotFound("no scene")
        return dict(SCENE, name=name or self.active)

    def filters(self):
        return {"dither": {"code": "x"}}

    def hwinfo(self):
        return {"model": "fake"}

    def set_param(self, scope, name, value):
        self.calls.append(("set_param", scope, name, value))

    def switch(self, name):
        if name not in self.names:
            raise NotFound("no scene")
        self.active = name
        self.calls.append(("switch", name))
        return name

    def apply_scene(self, scene):
        self.calls.append(("apply_scene", scene))

    def save_scene(self, name, scene):
        self.calls.append(("save_scene", name, scene))


@pytest.fixture
def host():
    return FakeHost()


@pytest.fixture
def api(host):
    return ConsoleAPI(host)


def _body(resp):
    return resp.body.decode("utf-8")


def _json(resp):
    return json.loads(_body(resp))


# reads


def test_stats_comes_from_the_host(api, host):
    r = api.dispatch("GET", "/stats")
    assert (r.status, r.content_type) == (200, "application/json")
    assert _json(r) == host.stats()


def test_scenes_lists_names_and_active(api):
    assert _json(api.dispatch("GET", "/scenes")) == {
        "scenes": ["a", "b", "c"],
        "active": "a",
    }


def test_scene_without_a_name_asks_for_the_active_one(api, host):
    api.dispatch("GET", "/scene")
    assert host.calls == [("scene_def", None)]


def test_scene_passes_the_name_through(api, host):
    assert _json(api.dispatch("GET", "/scene", {"name": "b"}))["name"] == "b"
    assert host.calls == [("scene_def", "b")]


def test_an_unknown_scene_is_404(api):
    assert api.dispatch("GET", "/scene", {"name": "nope"}).status == 404


def test_a_host_error_keeps_its_status(api, host):
    def boom(name=None):
        raise ServerError("build x: kaboom")

    host.scene_def = boom
    r = api.dispatch("GET", "/scene")
    assert r.status == 500
    assert "kaboom" in _body(r)


def test_filters_and_hwinfo_come_from_the_host(api):
    assert _json(api.dispatch("GET", "/filters")) == {"dither": {"code": "x"}}
    assert _json(api.dispatch("GET", "/hwinfo")) == {"model": "fake"}


def test_defaults_supply_the_registry_and_no_hwinfo():
    class Bare(ConsoleHost):
        pass

    assert Bare().hwinfo() == {}
    assert isinstance(Bare().filters(), dict)  # ghee's own filter registry


# writes


@pytest.mark.parametrize(
    "ptype,raw,want",
    [
        ("float", "0.5", 0.5),
        ("int", "5", 5.0),
        ("color", "1,0,0.5", [1.0, 0.0, 0.5]),
        ("vec2", "0.1,0.2", [0.1, 0.2]),
        ("string", "GHEE", "GHEE"),
        ("bool", "true", True),
        ("bool", "false", False),
    ],
)
def test_set_reaches_the_host_with_a_typed_value(api, host, ptype, raw, want):
    r = api.dispatch(
        "POST", "/set", {"pass": "0", "name": "k", "type": ptype, "value": raw}
    )
    assert (r.status, _body(r)) == (200, "ok")
    assert host.calls == [("set_param", 0, "k", want)]


def test_set_routes_the_scene_scope(api, host):
    api.dispatch("POST", "/set", {"pass": "scene", "name": "tint", "value": "0.5"})
    assert host.calls == [("set_param", "scene", "tint", 0.5)]


@pytest.mark.parametrize(
    "query",
    [
        {"pass": "0", "name": "k"},  # no value
        {"pass": "0", "value": "1"},  # no name
        {"name": "k", "value": "1"},  # no pass
        {"pass": "nope", "name": "k", "value": "1"},
        {"pass": "0", "name": "k", "type": "color", "value": "1,0"},
        {"pass": "0", "name": "k", "type": "bool", "value": "yes"},
        {"pass": "0", "name": "k", "value": "notanumber"},
    ],
)
def test_a_malformed_set_never_reaches_the_host(api, host, query):
    r = api.dispatch("POST", "/set", query)
    assert r.status == 400
    assert "bad /set" in _body(r)
    assert host.calls == []


def test_switch_returns_the_new_active(api, host):
    assert _json(api.dispatch("POST", "/switch", {"name": "c"})) == {"active": "c"}
    assert host.calls == [("switch", "c")]


def test_switching_to_nothing_is_404(api, host):
    assert api.dispatch("POST", "/switch", {}).status == 404
    assert host.calls == []


def test_next_and_prev_wrap_through_the_host_names(api):
    assert _json(api.dispatch("POST", "/next")) == {"active": "b"}
    assert _json(api.dispatch("POST", "/next")) == {"active": "c"}
    assert _json(api.dispatch("POST", "/next")) == {"active": "a"}  # wraps
    assert _json(api.dispatch("POST", "/prev")) == {"active": "c"}


def test_step_with_no_scenes_is_404(api, host):
    host.names = []
    assert api.dispatch("POST", "/next").status == 404


def test_a_posted_scene_arrives_validated(api, host):
    r = api.dispatch("POST", "/scene", {}, JSON, json.dumps(SCENE).encode())
    assert (r.status, _body(r)) == (200, "ok")
    kind, scene = host.calls[0]
    assert kind == "apply_scene"
    assert isinstance(scene, Scene) and scene.passes[0].name == "main"


def test_an_invalid_scene_is_400_and_never_reaches_the_host(api, host):
    r = api.dispatch("POST", "/scene", {}, JSON, b'{"passes": "not-a-list"}')
    assert r.status == 400
    assert "passes" in _body(r)
    assert host.calls == []


def test_a_scene_naming_a_shader_file_is_400(api, host):
    """A posted scene has no base directory, so a "shader" path is a read of
    something off disk: the model refuses it and the API surfaces that."""
    body = json.dumps(
        {"passes": [{"name": "main", "shader": "/etc/hostname"}]}
    ).encode()
    r = api.dispatch("POST", "/scene", {}, JSON, body)
    assert r.status == 400
    assert "shader" in _body(r)
    assert host.calls == []


def test_unparseable_json_is_400(api, host):
    assert api.dispatch("POST", "/scene", {}, JSON, b"{oops").status == 400
    assert host.calls == []


@pytest.mark.parametrize("path", ["/scene", "/save"])
def test_a_body_without_a_json_content_type_is_415(api, host, path):
    r = api.dispatch("POST", path, {}, {}, json.dumps(SCENE).encode())
    assert r.status == 415
    assert "application/json" in _body(r)
    assert host.calls == []


def test_save_is_refused_unless_the_host_opts_in(api, host):
    body = json.dumps({"name": "a", "scene": SCENE}).encode()
    r = api.dispatch("POST", "/save", {}, JSON, body)
    assert (r.status, _body(r)) == (403, "nope")
    assert host.calls == []


def test_a_writable_host_gets_the_name_and_a_scene(api, host):
    host.writable = True
    body = json.dumps({"name": "a", "scene": SCENE}).encode()
    assert api.dispatch("POST", "/save", {}, JSON, body).status == 200
    kind, name, scene = host.calls[0]
    assert (kind, name) == ("save_scene", "a")
    assert isinstance(scene, Scene)


def test_a_save_without_a_scene_is_400(api, host):
    host.writable = True
    r = api.dispatch("POST", "/save", {}, JSON, b'{"name": "a"}')
    assert r.status == 400
    assert host.calls == []


def test_an_unrouted_post_is_404(api):
    assert api.dispatch("POST", "/nope", {}, JSON, b"{}").status == 404


def test_an_unknown_method_is_405(api):
    assert api.dispatch("DELETE", "/stats").status == 405


# refusals: precheck decides them, and dispatch enforces them again


def test_state_changing_paths_are_post_only(api, host):
    for path in ("/set", "/switch", "/next", "/prev"):
        r = api.dispatch("GET", path, {"pass": "0", "name": "k", "value": "1"})
        assert (r.status, path) == (405, path)
        assert "POST" in _body(r)
    assert host.calls == []


def test_a_host_declaring_its_own_get_routes_still_mutates_through_post():
    """spikerer keeps GET /set for its scripts and buttons: it re-labels the
    request as a POST rather than weakening the core."""
    host = FakeHost()
    api = ConsoleAPI(host, post_only=())
    assert api.precheck("GET", "/set", {}) is None
    api.dispatch("POST", "/set", {"pass": "0", "name": "k", "value": "1"})
    assert host.calls == [("set_param", 0, "k", 1.0)]


def test_a_cross_origin_post_is_refused(api, host):
    headers = dict(JSON, Host="127.0.0.1:8090", Origin="http://evil.example")
    r = api.dispatch("POST", "/scene", {}, headers, json.dumps(SCENE).encode())
    assert r.status == 403
    assert "cross-origin" in _body(r)
    assert host.calls == []


def test_a_same_origin_post_is_accepted(api, host):
    headers = dict(JSON, Host="127.0.0.1:8090", Origin="http://127.0.0.1:8090")
    assert (
        api.dispatch("POST", "/scene", {}, headers, json.dumps(SCENE).encode()).status
        == 200
    )


def test_a_get_carrying_an_origin_is_not_refused(api):
    """Reads are safe; only the mutating side gets the Origin check."""
    assert api.precheck("GET", "/stats", {"Origin": "http://evil.example"}) is None


def test_an_oversized_body_is_413_before_it_is_read(api, host):
    headers = dict(JSON, **{"Content-Length": str(MAX_BODY + 1)})
    r = api.precheck("POST", "/scene", headers)
    assert r.status == 413
    assert str(MAX_BODY) in _body(r)
    assert api.dispatch("POST", "/scene", {}, headers, b"{}").status == 413
    assert host.calls == []


def test_a_bad_content_length_is_400(api):
    assert (
        api.precheck("POST", "/scene", {"Content-Length": "not-a-number"}).status == 400
    )


def test_the_body_cap_is_configurable(host):
    api = ConsoleAPI(host, max_body=10)
    assert api.precheck("POST", "/scene", {"Content-Length": "11"}).status == 413
    assert api.precheck("POST", "/scene", {"Content-Length": "10"}) is None


# static files


def test_static_serves_the_bundled_page():
    from ghee.console import web_dir

    api = ConsoleAPI(FakeHost(), web=web_dir())
    r = api.dispatch("GET", "/")
    assert r.status == 200
    assert b"<title>ghee</title>" in r.body
    assert r.content_type == "text/html"


def test_no_web_root_serves_nothing(api):
    assert api.dispatch("GET", "/index.html").status == 404


def test_static_path_refuses_traversal_and_prefix_siblings(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("ok", encoding="utf-8")
    sibling = tmp_path / "webXYZ"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret", encoding="utf-8")
    assert static_path(str(web), "/") == str(web / "index.html")
    assert static_path(str(web), "/../webXYZ/secret.txt") is None
    assert static_path(str(web), "/../../etc/passwd") is None
    assert static_path(str(web), "/missing.js") is None


# the pure helpers


def test_origin_helper():
    assert origin_allowed("", "h")  # curl, a script, the CLI
    assert origin_allowed("http://127.0.0.1:1", "127.0.0.1:1")
    assert not origin_allowed("http://evil.example", "127.0.0.1:1")
    assert not origin_allowed("null", "127.0.0.1:1")
    assert not origin_allowed("http://evil.example", None)  # no Host to match


def test_parse_query_takes_the_first_value():
    assert parse_query("a=1&b=x&a=2") == {"a": "1", "b": "x"}
    assert parse_query("") == {}


def test_parse_scope():
    assert parse_scope("scene") == "scene"
    assert parse_scope("3") == 3
    with pytest.raises(ValueError):
        parse_scope("notanint")


def test_bool_parses_both_spellings_and_refuses_anything_else():
    for raw in ("true", "TRUE", " true ", "1"):
        assert parse_param_value("bool", raw) is True
    for raw in ("false", "False", "0"):
        assert parse_param_value("bool", raw) is False
    for raw in ("yes", "", "2", "0.0"):
        with pytest.raises(ValueError):
            parse_param_value("bool", raw)


# the scene store


@pytest.fixture
def scenes(tmp_path):
    (tmp_path / "one.scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
    (tmp_path / "two.scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
    return SceneStore(str(tmp_path))


def test_the_store_scans_a_directory(scenes):
    assert scenes.names == ["one", "two"]
    assert "one" in scenes and "three" not in scenes


def test_the_store_resolves_a_scene(scenes):
    scene, controller = scenes.resolve("one")
    assert scene.passes[0].name == "main"
    assert controller is None


def test_the_store_refuses_an_unknown_name(scenes):
    with pytest.raises(NotFound):
        scenes.path("three")
    with pytest.raises(NotFound):
        scenes.save("three", Scene.from_dict(SCENE))


def test_a_save_shadows_the_original_without_touching_it(scenes, tmp_path):
    before = (tmp_path / "one.scene.json").read_text(encoding="utf-8")
    edited = dict(SCENE)
    edited["passes"] = [dict(SCENE["passes"][0], code="void main(){ discard; }")]
    scenes.save("one", Scene.from_dict(edited))
    assert (tmp_path / "one.scene.json").read_text(encoding="utf-8") == before
    saved = tmp_path / "modified" / "one.scene.json"
    assert saved.is_file()
    assert "discard" in scenes.resolve("one").__getitem__(0).passes[0].code


PROC = """
from ghee import Param, Pass, Scene, SceneController


class C(SceneController):
    def build(self, base):
        return Scene(
            passes=[
                Pass(
                    name="main",
                    code="void main(){ gl_FragColor = vec4(1.0); }",
                    params={
                        "amt": Param("amt", "float", 0.25, 0.0, 1.0, 0.01),
                        "on": Param("on", "bool", False),
                    },
                )
            ]
        )
"""


@pytest.fixture
def procedural(tmp_path):
    (tmp_path / "proc.py").write_text(PROC, encoding="utf-8")
    return SceneStore(str(tmp_path))


def test_a_procedural_scene_saves_a_values_sidecar(procedural, tmp_path):
    scene, _ = procedural.resolve("proc")
    scene.passes[0].params["amt"] = Param("amt", "float", 0.75, 0.0, 1.0, 0.01)
    scene.passes[0].params["on"] = Param("on", "bool", True)
    procedural.save("proc", scene)
    vals = json.loads((tmp_path / "modified" / "proc.values.json").read_text())
    assert vals["passes"]["main"] == {"amt": 0.75, "on": True}
    assert not (tmp_path / "modified" / "proc.scene.json").exists()


def test_the_sidecar_is_applied_over_the_code_defaults(procedural, tmp_path):
    assert procedural.resolve("proc")[0].passes[0].params["amt"].value == 0.25
    os.makedirs(tmp_path / "modified", exist_ok=True)
    (tmp_path / "modified" / "proc.values.json").write_text(
        json.dumps({"passes": {"main": {"amt": 0.9, "on": True}}}), encoding="utf-8"
    )
    params = procedural.resolve("proc")[0].passes[0].params
    assert params["amt"].value == 0.9
    assert params["on"].value is True  # a bool round-trips as a real bool


def test_a_sidecar_bool_written_as_a_number_is_skipped(procedural, tmp_path, capsys):
    os.makedirs(tmp_path / "modified", exist_ok=True)
    (tmp_path / "modified" / "proc.values.json").write_text(
        json.dumps({"passes": {"main": {"on": 1}}}), encoding="utf-8"
    )
    assert procedural.resolve("proc")[0].passes[0].params["on"].value is False
    assert "bad 'main.on'" in capsys.readouterr().err


def test_a_scene_json_scene_ignores_a_stray_sidecar(scenes, tmp_path):
    """Only a procedural scene reads one; a JSON scene saves a whole copy."""
    os.makedirs(tmp_path / "modified", exist_ok=True)
    (tmp_path / "modified" / "one.values.json").write_text(
        json.dumps({"passes": {"main": {"nope": 1}}}), encoding="utf-8"
    )
    assert scenes.resolve("one")[0].passes[0].name == "main"


def test_a_declared_filter_comes_back_as_a_pass(tmp_path):
    (tmp_path / "f.scene.json").write_text(
        json.dumps(dict(SCENE, filters=["dither"])), encoding="utf-8"
    )
    store = SceneStore(str(tmp_path))
    scene, _ = store.resolve("f")
    assert [p.filter for p in scene.passes] == ["", "dither"]
    store.save("f", scene)  # the expanded pass must not be written back
    written = json.loads((tmp_path / "modified" / "f.scene.json").read_text())
    assert written["filters"] == ["dither"]
    assert len(written["passes"]) == 1


def test_the_store_reports_whether_it_may_be_written():
    assert SceneStore(".", read_only=True).read_only is True
    assert SceneStore(".").read_only is False


def test_a_pass_keeps_its_params_through_a_round_trip(tmp_path):
    scene = Scene(
        passes=[
            Pass(
                name="main",
                code="void main(){}",
                params={"on": Param("on", "bool", True)},
            )
        ]
    )
    (tmp_path / "b.scene.json").write_text(
        json.dumps(scene.to_dict()), encoding="utf-8"
    )
    store = SceneStore(str(tmp_path))
    assert store.resolve("b")[0].passes[0].params["on"].value is True
