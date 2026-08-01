"""The feeder shim (shim/ghee.py): the scene it emits, the probe, the wire.

The shim ships beside the bundled demos, so it is loaded from its file here
rather than imported.

    python3 -m pytest tests/test_shim.py
"""

import importlib.util
import io
import itertools
import json
import os
import struct
import sys

import pytest

SHIM = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shim", "ghee.py")
_names = itertools.count()


def load():
    """A fresh shim: the feed and hook registries are module state."""
    name = f"ghee_shim_{next(_names)}"
    spec = importlib.util.spec_from_file_location(name, SHIM)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves the postponed annotations through sys.modules
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Out:
    """sys.stdout as run() uses it: the binary buffer under it."""

    def __init__(self):
        self.buffer = io.BytesIO()


def drive(shim, scene, ticks, monkeypatch):
    """Run the loop over `ticks` -> (handshake dict, message bytes)."""
    lines = "".join(json.dumps(t) + "\n" for t in ticks)
    out = Out()
    monkeypatch.setattr(sys, "stdin", io.StringIO(lines))
    monkeypatch.setattr(sys, "stdout", out)
    shim.run(scene)
    head, _, rest = out.buffer.getvalue().partition(b"\n")
    return json.loads(head), rest


def messages(buf):
    """The wire back to (kind, name, values)."""
    out, i = [], 0
    while i < len(buf):
        kind = buf[i]
        i += 1
        if kind == 2:
            out.append((2, "", []))
            continue
        nlen = buf[i]
        i += 1
        name = buf[i : i + nlen].decode()
        i += nlen
        (count,) = struct.unpack_from("<I", buf, i)
        i += 4
        out.append((kind, name, list(struct.unpack_from(f"<{count}f", buf, i))))
        i += 4 * count
    return out


def probe(shim, argv, monkeypatch, specs=()):
    """`args()` under a command line -> (result or SystemExit code, stdout)."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        return shim.args(specs), buf.getvalue()
    except SystemExit as e:
        return e.code, buf.getvalue()


# ---------------------------------------------------------------- scene model


RECORDED = {
    "version": 1,
    "passes": [
        {
            "name": "grid",
            "code": "void main(){}",
            "size": [64, 32],
            "arrays": [{"name": "bins", "size": 16}, {"name": "mask", "size": [8, 4]}],
        },
        {
            "name": "main",
            "code": "void main(){}",
            "channels": [{"buffer": "grid"}],
            "params": {
                "gain": {
                    "type": "float",
                    "value": 1.0,
                    "min": 0,
                    "max": 2,
                    "step": 0.1,
                },
                "steps": {"type": "int", "value": 3},
                "on": {"type": "bool", "value": True},
                "tint": {"type": "color", "value": [1.0, 0.5, 0.0]},
                "pan": {"type": "vec2", "value": [0.0, 1.0]},
                "word": {"type": "string", "value": "hi", "options": ["hi", "ho"]},
            },
        },
    ],
    "common": {"code": "float k(){ return 1.0; }"},
    "filters": ["dither"],
    "params": {"speed": {"type": "float", "value": 0.5}},
}


def test_the_scene_json_is_the_recorded_json():
    """The recorded scene document, built by the shim's own model."""
    shim = load()
    built = shim.Scene(
        passes=[
            shim.Pass(
                name="grid",
                code="void main(){}",
                size=(64, 32),
                arrays=[shim.ArrayInput("bins", 16), shim.ArrayInput("mask", (8, 4))],
            ),
            shim.Pass(
                name="main",
                code="void main(){}",
                channels=[shim.Channel(buffer="grid")],
                params={
                    "gain": shim.Param("gain", "float", 1, min=0, max=2, step=0.1),
                    "steps": shim.Param("steps", "int", 3),
                    "on": shim.Param("on", "bool", True),
                    "tint": shim.Param("tint", "color", (1, 0.5, 0)),
                    "pan": shim.Param("pan", "vec2", [0, 1]),
                    "word": shim.Param("word", "string", "hi", options=["hi", "ho"]),
                },
            ),
        ],
        common="float k(){ return 1.0; }",
        filters=["dither"],
        params={"speed": shim.Param("speed", "float", 0.5)},
    ).to_dict()
    assert built == RECORDED
    assert json.dumps(built) == json.dumps(RECORDED)  # key order too


def test_an_array_declares_its_size_and_its_count():
    shim = load()
    flat, grid = shim.ArrayInput("bins", 16), shim.ArrayInput("mask", [8, 4])
    assert flat.to_dict() == {"name": "bins", "size": 16} and flat.count == 16
    assert grid.to_dict() == {"name": "mask", "size": [8, 4]} and grid.count == 32


def test_a_param_of_an_unknown_type_says_so():
    shim = load()
    with pytest.raises(ValueError, match="unknown type"):
        shim.Param("gain", "double", 1.0)
    with pytest.raises(ValueError, match="needs 3 numbers"):
        shim.Param("tint", "color", [1.0, 0.0])


# ---------------------------------------------------------------------- args


def test_the_probe_answers_one_json_line_and_stops(monkeypatch):
    shim = load()
    specs = [
        shim.Arg("emit", "size", default=(480, 300), help="draw at WxH"),
        shim.Arg("bins", "int", default=16),
        shim.Arg("dither", "bool", default=True),
        shim.Arg("wash", "size"),
    ]
    code, out = probe(shim, ["feeder.py", "--ghee-args"], monkeypatch, specs)
    assert code == 0
    assert out.endswith("\n") and out.count("\n") == 1
    assert json.loads(out) == {
        "args": [
            {
                "name": "emit",
                "type": "size",
                "default": "480x300",
                "help": "draw at WxH",
            },
            {"name": "bins", "type": "int", "default": 16, "help": ""},
            {"name": "dither", "type": "bool", "default": True, "help": ""},
            {"name": "wash", "type": "size", "default": None, "help": ""},
        ]
    }


def test_help_lists_the_args(monkeypatch):
    shim = load()
    specs = [shim.Arg("emit", "size", default=(480, 300), help="draw at WxH")]
    code, out = probe(shim, ["feeder.py", "--help"], monkeypatch, specs)
    assert code == 0
    assert "--emit WxH" in out and "draw at WxH (default: 480x300)" in " ".join(
        out.split()
    )


def test_the_command_line_lands_on_the_declared_names(monkeypatch):
    shim = load()
    specs = [
        shim.Arg("emit", "size"),
        shim.Arg("near-seg", "int", default=3),
        shim.Arg("dither", "bool", default=True),
        shim.Arg("gain", "float", default=1.0),
    ]
    argv = ["feeder.py", "--emit", "480x300", "--near-seg", "1", "--no-dither"]
    got, _ = probe(shim, argv, monkeypatch, specs)
    assert (got.emit, got.near_seg, got.dither, got.gain) == ((480, 300), 1, False, 1.0)
    plain, _ = probe(shim, ["feeder.py"], monkeypatch, specs)
    assert (plain.emit, plain.near_seg, plain.dither) == (None, 3, True)


def test_a_bad_size_is_rejected(monkeypatch):
    shim = load()
    monkeypatch.setattr(sys, "argv", ["feeder.py", "--emit", "480"])
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit):
        shim.args([shim.Arg("emit", "size")])


# ------------------------------------------------------------------ the loop


def scene_of(shim, size=4):
    return shim.Scene(
        passes=[shim.Pass("main", "void main(){}", arrays=[shim.ArrayInput("v", size)])]
    )


def test_the_handshake_is_the_first_line_and_the_data_follows(monkeypatch):
    shim = load()

    @shim.feed("v")
    def v(tick):
        return [tick.frame, tick.time, tick.dt, len(tick.params)]

    ticks = [
        {"frame": 0, "time": 0.0, "dt": 0.5, "params": {"word": "hi"}},
        {"frame": 1, "time": 0.5, "dt": 0.5, "params": {}},
    ]
    head, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert head == scene_of(shim).to_dict()
    assert messages(rest) == [
        (0, "v", [0.0, 0.0, 0.5, 1.0]),
        (0, "v", [1.0, 0.5, 0.5, 0.0]),
    ]


def test_a_feed_that_returns_none_sends_nothing(monkeypatch):
    shim = load()

    @shim.feed("v")
    def v(tick):
        return [1, 2, 3, 4] if tick.frame == 1 else None

    ticks = [{"frame": n, "time": 0.0, "dt": 0.0, "params": {}} for n in range(3)]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [(0, "v", [1.0, 2.0, 3.0, 4.0])]


def test_a_param_and_a_clean_tick(monkeypatch):
    shim = load()

    @shim.tick
    def hook(tick):
        if tick.frame == 0:
            shim.set_param("gain", 0.5)
            shim.set_param("tint", [1.0, 0.0, 0.25])
        else:
            shim.skip()

    ticks = [{"frame": n, "time": 0.0, "dt": 0.0, "params": {}} for n in range(2)]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [
        (1, "gain", [0.5]),
        (1, "tint", [1.0, 0.0, 0.25]),
        (2, "", []),
    ]
    assert rest.endswith(b"\x02")


def test_a_feed_of_the_wrong_length_names_the_array(monkeypatch):
    shim = load()

    @shim.feed("v")
    def v(tick):
        return [1.0, 2.0]

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    with pytest.raises(ValueError, match="array 'v' takes 4 values, got 2"):
        drive(shim, scene_of(shim), ticks, monkeypatch)


def test_a_feed_the_scene_never_declared_is_caught_at_the_handshake(monkeypatch):
    shim = load()

    @shim.feed("nope")
    def nope(tick):
        return [0.0]

    with pytest.raises(ValueError, match="feed 'nope' is not an array"):
        drive(shim, scene_of(shim), [], monkeypatch)


def test_a_float32_buffer_goes_out_as_it_lies(monkeypatch):
    """A numpy array reaches the wire through the buffer protocol, so the
    values must survive a source that is not a list."""
    shim = load()
    np = pytest.importorskip("numpy")

    @shim.feed("v")
    def v(tick):
        return np.array([[0.25, 0.5], [0.75, 1.0]], np.float32)

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [(0, "v", [0.25, 0.5, 0.75, 1.0])]


def test_the_tick_line_carries_the_string_params(monkeypatch):
    shim = load()
    seen = []

    @shim.tick
    def hook(tick):
        seen.append((tick.frame, tick.time, tick.dt, tick.params))

    ticks = [
        {"frame": 3, "time": 0.25, "dt": 0.016, "params": {"text": "GHEE"}},
        {"frame": 4, "time": 0.5, "dt": 0.016},
    ]
    drive(shim, scene_of(shim), ticks, monkeypatch)
    assert seen == [(3, 0.25, 0.016, {"text": "GHEE"}), (4, 0.5, 0.016, {})]
