"""The feeder shim (ghee._shim): the scene it emits, the probe, the wire.

The shim is the Python half of the ghee-render wheel. Each test loads a copy
of it from the installed file, because the feed and hook registries are module
state.

    pytest python/tests/test_shim.py
"""

import array
import glob
import importlib.util
import io
import itertools
import json
import pathlib
import re
import struct
import subprocess
import sys

import pytest
from conftest import numpy
from test_errors import CLI

SHIM = importlib.util.find_spec("ghee._shim").origin
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


def drive(shim, scene, ticks, monkeypatch, lockstep=True):
    """Run the loop over `ticks` -> (handshake dict, message bytes).

    The mark is what puts run() on the wire. test_direct.py drives the rest.
    `lockstep` False stands for a ghee that reads no kind 5.
    """
    lines = "".join(json.dumps(t) + "\n" for t in ticks)
    out = Out()
    monkeypatch.setenv(shim.WIRE, "1")
    if lockstep:
        monkeypatch.setenv(shim.LOCKSTEP, "1")
    else:
        monkeypatch.delenv(shim.LOCKSTEP, raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(lines))
    monkeypatch.setattr(sys, "stdout", out)
    shim.run(scene)
    head, _, rest = out.buffer.getvalue().partition(b"\n")
    return json.loads(head), rest


# kind 5, the end of one tick's answer
END = (5, "", [])


def messages(buf):
    """The wire back to (kind, name, values).

    Kind 4 carries one byte for each value, kinds 3 and 9 carry text, and
    kinds 6, 7, 8 and 10 carry JSON, read back. Kinds 2 and 5 are the single
    byte.
    """
    out, i = [], 0
    while i < len(buf):
        kind = buf[i]
        i += 1
        if kind in (2, 5):
            out.append((kind, "", []))
            continue
        nlen = buf[i]
        i += 1
        name = buf[i : i + nlen].decode()
        i += nlen
        (count,) = struct.unpack_from("<I", buf, i)
        i += 4
        if kind in (3, 6, 7, 8, 9, 10):
            text = buf[i : i + count].decode()
            out.append((kind, name, text if kind in (3, 9) else json.loads(text)))
            i += count
            continue
        width = 1 if kind == 4 else 4
        fmt = f"<{count}B" if kind == 4 else f"<{count}f"
        out.append((kind, name, list(struct.unpack_from(fmt, buf, i))))
        i += width * count
    return out


def probe(shim, argv, monkeypatch, specs=()):
    """`args()` under a command line -> (result or SystemExit code, stdout)."""
    buf = io.StringIO()
    monkeypatch.setenv(shim.WIRE, "1")
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
            "visible": False,
            "fragment": "void main(){}",
            "size": [64, 32],
            "sampling": "trilinear",
            "arrays": [{"name": "bins", "size": 16}, {"name": "mask", "size": [8, 4]}],
        },
        {
            "name": "main",
            "fragment": "void main(){}",
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
    "common": "float k(){ return 1.0; }",
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
                fragment="void main(){}",
                size=(64, 32),
                sampling="trilinear",
                visible=False,
                arrays=[shim.ArrayInput("bins", 16), shim.ArrayInput("mask", (8, 4))],
            ),
            shim.Pass(
                name="main",
                fragment="void main(){}",
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


def test_params_take_a_list_or_a_dict_keyed_by_name():
    shim = load()
    gain, level = shim.Param("gain", "float", 1.0), shim.Param("level", "int", 2)
    keyed = {"gain": gain.to_dict(), "level": level.to_dict()}
    for params in ([gain, level], {"gain": gain, "level": level}):
        doc = shim.Scene(
            passes=[shim.Pass("main", "void main(){}", params=params)], params=params
        ).to_dict()
        assert doc["params"] == keyed
        assert doc["passes"][0]["params"] == keyed


@pytest.mark.parametrize(
    "owner,params,error",
    [
        ("scene", "dict", "scene param 'level' sits under the key 'gain'"),
        ("pass", "dict", "param 'level' in pass 'main' sits under the key 'gain'"),
        ("scene", "list", "scene param 'level' is declared twice"),
        ("pass", "list", "param 'level' in pass 'main' is declared twice"),
    ],
)
def test_a_param_under_another_name_or_twice_is_refused(owner, params, error):
    shim = load()
    level = shim.Param("level", "float", 1.0)
    given = {"gain": level} if params == "dict" else [level, level]
    with pytest.raises(ValueError) as raised:
        if owner == "scene":
            shim.Scene(params=given)
        else:
            shim.Pass("main", "void main(){}", params=given)
    assert str(raised.value) == error


def test_an_array_declares_its_size_and_its_count():
    shim = load()
    flat, grid = shim.ArrayInput("bins", 16), shim.ArrayInput("mask", [8, 4])
    assert flat.to_dict() == {"name": "bins", "size": 16} and flat.count == 16
    assert grid.to_dict() == {"name": "mask", "size": [8, 4]} and grid.count == 32


def test_a_layer_carries_its_words():
    shim = load()
    layer = shim.Layer(name="words", text="hi", font_family="sans", text_align="center")
    built = shim.Scene(
        passes=[shim.Pass("main", "void main(){}", layers=[layer])],
        assets=[shim.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
    ).to_dict()
    assert built == {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){}",
                "layers": [
                    {
                        "name": "words",
                        "text": "hi",
                        "font_family": "sans",
                        "text_align": "center",
                    }
                ],
            }
        ],
        "assets": {"sans": {"file": "fonts/DejaVuSans-Latin.ttf"}},
    }


def test_an_asset_is_its_file():
    shim = load()
    scene = shim.Scene(
        passes=[shim.Pass("main", "void main(){}")],
        assets=[shim.Asset("logo", "logo.png")],
    )
    assert scene.to_dict()["assets"] == {"logo": {"file": "logo.png"}}
    assert "assets" not in shim.Scene(passes=[]).to_dict()


def test_an_asset_takes_the_keys_of_the_scene_format():
    shim = load()
    raw = shim.Asset(
        "mask",
        "mask.raw",
        width=4,
        height=2,
        format="u8",
        components=1,
        sampling="linear",
    )
    icon = shim.Asset("icon", "icon.png", sampling="nearest")
    recorded = [
        {
            "file": "mask.raw",
            "width": 4,
            "height": 2,
            "format": "u8",
            "components": 1,
            "sampling": "linear",
        },
        {"file": "icon.png", "sampling": "nearest"},
    ]
    built = [raw.to_dict(), icon.to_dict()]
    assert built == recorded
    assert json.dumps(built) == json.dumps(recorded)  # key order too


def test_a_gradient_is_the_gradient_object_of_the_format():
    shim = load()
    assert shim.gradient("red", "yellow", angle=90) == {
        "from": "red",
        "to": "yellow",
        "angle": 90,
    }
    assert shim.gradient("red", [0, 0, 1])["angle"] == 180


def test_two_assets_of_one_name_are_refused():
    """The document keys them by name, where the second would hide the first."""
    shim = load()
    with pytest.raises(ValueError) as raised:
        shim.Scene(assets=[shim.Asset("logo", "a.png"), shim.Asset("logo", "b.png")])
    assert str(raised.value) == "asset 'logo' is declared twice"


def test_numpy_numbers_build_a_document_that_json_takes():
    shim = load()
    np = pytest.importorskip("numpy")
    scene = shim.Scene(
        passes=[
            shim.Pass(
                "main",
                "void main(){}",
                size=(np.int64(64), np.int64(32)),
                arrays=[shim.ArrayInput("mask", np.array([8, 4]))],
                params={"tint": shim.Param("tint", "color", np.array([1, 0.5, 0]))},
            )
        ],
        params={"gain": shim.Param("gain", "float", np.float32(0.5), max=np.int64(2))},
    )
    assert json.loads(json.dumps(scene.to_dict())) == {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){}",
                "size": [64, 32],
                "arrays": [{"name": "mask", "size": [8, 4]}],
                "params": {"tint": {"type": "color", "value": [1.0, 0.5, 0.0]}},
            }
        ],
        "params": {"gain": {"type": "float", "value": 0.5, "max": 2}},
    }


def test_a_layer_takes_the_scene_json_keys():
    shim = load()
    layer = shim.Layer(
        name="title",
        text="hi",
        visible=False,
        x=4,
        y=8,
        color={"from": "#ff0000", "to": [1, 1, 0], "angle": 90},
        opacity=0.5,
        blend="add",
        font_size=48,
        font_family="sans",
        text_align="right",
        render="slug",
        text_shadow={"x": 0, "y": 1},
        glow={},
        text_stroke={"visible": False},
        karaoke={"sung": 3},
        wave={},
        scroll={"speed": 4},
        blink={"rate": 2},
    )
    recorded = {
        "name": "title",
        "visible": False,
        "x": 4.0,
        "y": 8.0,
        "color": {"from": "#ff0000", "to": [1.0, 1.0, 0.0], "angle": 90.0},
        "opacity": 0.5,
        "blend": "add",
        "text": "hi",
        "font_size": 48.0,
        "font_family": "sans",
        "text_align": "right",
        "render": "slug",
        "text_shadow": {"x": 0.0, "y": 1.0},
        "glow": {},
        "text_stroke": {"visible": False},
        "karaoke": {"sung": 3.0},
        "wave": {},
        "scroll": {"speed": 4.0},
        "blink": {"rate": 2.0},
    }
    built = layer.to_dict()
    assert built == recorded
    assert json.dumps(built) == json.dumps(recorded)  # key order too


def test_a_layer_takes_a_list_of_fonts():
    shim = load()
    fonts = ("Pixel Operator 8", pathlib.Path("noto.ttf"))
    assert shim.Layer("t", text="hi", font_family=fonts).to_dict() == {
        "name": "t",
        "text": "hi",
        "font_family": ["Pixel Operator 8", "noto.ttf"],
    }


def test_a_layer_leaves_render_to_ghee_unless_it_names_one():
    shim = load()
    assert (
        "render"
        not in shim.Layer("t", text="hi", font_family="Pixel Operator 8").to_dict()
    )
    named = shim.Layer("t", text="hi", font_family="Pixel Operator 8", render="mtsdf")
    assert named.to_dict()["render"] == "mtsdf"


def test_an_image_layer_takes_the_scene_json_keys():
    shim = load()
    layer = shim.Layer(
        name="icon",
        image="wifi.png",
        visible=False,
        x=110,
        y=0,
        color="#ff8000",
        opacity=0.5,
        blend="screen",
        width=16,
        height=8,
    )
    recorded = {
        "name": "icon",
        "visible": False,
        "x": 110.0,
        "y": 0.0,
        "color": "#ff8000",
        "opacity": 0.5,
        "blend": "screen",
        "image": "wifi.png",
        "width": 16.0,
        "height": 8.0,
    }
    built = layer.to_dict()
    assert built == recorded
    assert json.dumps(built) == json.dumps(recorded)  # key order too


def test_a_shape_layer_takes_the_scene_json_keys():
    shim = load()
    rect = shim.Layer(
        "frame",
        shape="rect",
        x=0,
        y=24,
        width=128,
        height=8,
        color="transparent",
        border={"width": 1, "color": "white"},
        border_radius=2,
    )
    line = shim.Layer("scope", shape="line", points=[(0, 16), (8, 4)], stroke_width=2)
    recorded = [
        {
            "name": "frame",
            "x": 0.0,
            "y": 24.0,
            "color": "transparent",
            "shape": "rect",
            "width": 128.0,
            "height": 8.0,
            "border": {"width": 1.0, "color": "white"},
            "border_radius": 2.0,
        },
        {
            "name": "scope",
            "shape": "line",
            "points": [[0.0, 16.0], [8.0, 4.0]],
            "stroke_width": 2.0,
        },
    ]
    built = [rect.to_dict(), line.to_dict()]
    assert built == recorded
    assert json.dumps(built) == json.dumps(recorded)  # key order too


def test_a_layer_leaves_its_defaults_out():
    shim = load()
    assert shim.Layer(name="words").to_dict() == {"name": "words"}
    assert shim.Layer("words", "hi").to_dict() == {"name": "words", "text": "hi"}
    assert "layers" not in shim.Pass("main", "void main(){}").to_dict()


def test_a_pass_of_layers_alone_has_no_fragment():
    shim = load()
    alone = shim.Pass("ink", layers=[shim.Layer("words", "hi")])
    assert alone.to_dict() == {
        "name": "ink",
        "layers": [{"name": "words", "text": "hi"}],
    }
    # a filter pass takes its fragment from its filter
    assert shim.Pass("dither", filter="dither").to_dict()["fragment"] == ""


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
        params={
            "gain": shim.Param("gain", "float", 0.0),
            "tint": shim.Param("tint", "color", [0.0, 0.0, 0.0]),
            "words": shim.Param("words", "string", ""),
        },
        passes=[
            shim.Pass(
                "main",
                "void main(){}",
                arrays=[shim.ArrayInput("v", size)],
                params={"on": shim.Param("on", "bool", False)},
            )
        ],
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
    assert head == {**scene_of(shim).to_dict(), "lockstep": True}
    assert messages(rest) == [
        (0, "v", [0.0, 0.0, 0.5, 1.0]),
        END,
        (0, "v", [1.0, 0.5, 0.5, 0.0]),
        END,
    ]


def test_under_a_ghee_that_reads_no_kind_5_the_wire_goes_on_without_it(monkeypatch):
    """No "lockstep" in the handshake, no kind 5, and a tick with nothing to
    send writes nothing."""
    shim = load()

    @shim.feed("v")
    def v(tick):
        return [1, 2, 3, 4] if tick.frame == 1 else None

    @shim.tick
    def hook(tick):
        if tick.frame == 2:
            shim.skip()

    ticks = [{"frame": n, "time": 0.0, "dt": 0.0, "params": {}} for n in range(3)]
    head, rest = drive(shim, scene_of(shim), ticks, monkeypatch, lockstep=False)
    assert head == scene_of(shim).to_dict()
    assert messages(rest) == [(0, "v", [1.0, 2.0, 3.0, 4.0]), (2, "", [])]


def test_a_feed_that_returns_none_sends_nothing(monkeypatch):
    shim = load()

    @shim.feed("v")
    def v(tick):
        return [1, 2, 3, 4] if tick.frame == 1 else None

    ticks = [{"frame": n, "time": 0.0, "dt": 0.0, "params": {}} for n in range(3)]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    # a tick with nothing to send still ends its answer
    assert messages(rest) == [END, (0, "v", [1.0, 2.0, 3.0, 4.0]), END, END]


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
        END,
        (2, "", []),
        END,
    ]
    assert rest.endswith(b"\x02\x05")


def test_a_color_written_hex_goes_out_as_three_floats(monkeypatch):
    """kind 1: each hex pair is a byte, and the channel is byte / 255."""
    shim = load()

    @shim.tick
    def hook(tick):
        shim.set_param("tint", "#ff8000")

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    f32 = struct.unpack("<3f", struct.pack("<3f", 1.0, 128 / 255, 0.0))
    assert messages(rest) == [(1, "tint", list(f32)), END]


def test_a_color_written_as_a_name_goes_out_as_text(monkeypatch):
    """kind 3: the name as written, which ghee keeps."""
    shim = load()

    @shim.tick
    def hook(tick):
        shim.set_param("tint", "Navy")

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [(3, "tint", "Navy"), END]


def test_a_string_param_goes_out_as_text(monkeypatch):
    """kind 3: the UTF-8 bytes, and the count is their length."""
    shim = load()

    @shim.tick
    def hook(tick):
        shim.set_param("words", "café")

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [(3, "words", "café"), END]
    assert rest == b"\x03\x05words\x05\x00\x00\x00caf\xc3\xa9\x05"


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
    assert messages(rest) == [(0, "v", [0.25, 0.5, 0.75, 1.0]), END]


def test_a_uint8_buffer_goes_out_as_bytes(monkeypatch):
    """kind 4: one byte for each value, and the engine divides by 255."""
    shim = load()
    np = pytest.importorskip("numpy")

    @shim.feed("v")
    def v(tick):
        return np.array([[0, 64], [128, 255]], np.uint8)

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [(4, "v", [0, 64, 128, 255]), END]
    # kind, name_len, name, count, bytes, then the end of the answer
    assert len(rest) == 1 + 1 + 1 + 4 + 4 + 1


QUARTERS = [0.25, 0.5, 0.75, 1.0]

# (what a feed returns, and the floats that go out)
NUMBERS = {
    "float64 2D": (lambda: numpy().array([[0.25, 0.5], [0.75, 1.0]]), QUARTERS),
    "int64": (lambda: numpy().arange(4, dtype="int64"), [0.0, 1.0, 2.0, 3.0]),
    "float16": (lambda: numpy().array(QUARTERS, "float16"), QUARTERS),
    "bool": (lambda: numpy().array([True, False, True, True]), [1.0, 0.0, 1.0, 1.0]),
    "float64 slice": (lambda: numpy().arange(8.0)[::2], [0.0, 2.0, 4.0, 6.0]),
    "float32 slice": (lambda: numpy().arange(8, dtype="float32")[1::2], [1, 3, 5, 7]),
    "array d": (lambda: array.array("d", QUARTERS), QUARTERS),
    "list of ints": (lambda: [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0]),
    "generator": (lambda: (n / 4 for n in range(1, 5)), QUARTERS),
}


@pytest.mark.parametrize("kind", NUMBERS)
def test_a_feed_of_any_numbers_goes_out_as_float32(kind, monkeypatch):
    shim = load()
    make, want = NUMBERS[kind]

    @shim.feed("v")
    def v(tick):
        return make()

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [(0, "v", want), END]


# (what a feed returns, and the error that it raises)
NOT_NUMBERS = {
    "str": (
        lambda: "four",
        "feed 'v' takes a buffer or an iterable of numbers, got str",
    ),
    "float": (
        lambda: 0.5,
        "feed 'v' takes a buffer or an iterable of numbers, got float",
    ),
    "list holding str": (
        lambda: [0.25, "half", 0.75, 1.0],
        "feed 'v' takes numbers, got list holding str",
    ),
    "complex": (
        lambda: numpy().zeros(4, "complex128"),
        "feed 'v' takes numbers, got a buffer of format 'Zd'",
    ),
}


@pytest.mark.parametrize("kind", NOT_NUMBERS)
def test_a_feed_of_anything_but_numbers_says_what_it_got(kind, monkeypatch):
    shim = load()
    make, error = NOT_NUMBERS[kind]

    @shim.feed("v")
    def v(tick):
        return make()

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    with pytest.raises(ValueError) as raised:
        drive(shim, scene_of(shim), ticks, monkeypatch)
    assert str(raised.value) == error


def test_bytes_go_out_as_bytes_and_floats_stay_floats(monkeypatch):
    shim = load()

    @shim.feed("v")
    def v(tick):
        return bytes([1, 2, 3, 4]) if tick.frame == 0 else [1.0, 2.0, 3.0, 4.0]

    ticks = [{"frame": n, "time": 0.0, "dt": 0.0, "params": {}} for n in range(2)]
    _, rest = drive(shim, scene_of(shim), ticks, monkeypatch)
    assert messages(rest) == [
        (4, "v", [1, 2, 3, 4]),
        END,
        (0, "v", [1.0, 2.0, 3.0, 4.0]),
        END,
    ]


def test_a_short_byte_feed_names_the_array(monkeypatch):
    shim = load()

    @shim.feed("v")
    def v(tick):
        return bytearray([1, 2])

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    with pytest.raises(ValueError, match="array 'v' takes 4 values, got 2"):
        drive(shim, scene_of(shim), ticks, monkeypatch)


def test_a_filter_param_goes_out_unchecked(monkeypatch):
    """ghee appends a filter's pass and holds its params."""
    shim = load()

    @shim.tick
    def hook(tick):
        shim.set_param("levels", 4.0, pass_="dither")
        shim.set_param("levels", 2.0)

    scene = scene_of(shim)
    scene.filters = ["dither"]
    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene, ticks, monkeypatch)
    assert messages(rest) == [(1, "dither.levels", [4.0]), (1, "levels", [2.0]), END]


# set_param's refusals, each the same under ghee and in direct mode
REFUSED = [
    (
        ("nope", 1.0),
        (
            "param 'nope' is not a param this scene declares "
            "(gain, tint, words, main.on, on)"
        ),
    ),
    (("on", 1.0, "other"), "param 'other.on' is not a param this scene declares"),
    (("gain", "x"), "float param 'gain' needs a number, got 'x'"),
    (("gain", True), "float param 'gain' needs a number, got True"),
    (
        ("tint", [1.0, 0.0]),
        (
            "color param 'tint' needs '#rrggbb', a CSS colour name or three numbers "
            "[r, g, b], got [1.0, 0.0]"
        ),
    ),
    (
        ("tint", "#ff80"),
        (
            "color param 'tint' needs '#rrggbb', a CSS colour name or three numbers "
            "[r, g, b], got '#ff80'"
        ),
    ),
    (
        ("tint", "navyy"),
        (
            "color param 'tint' needs '#rrggbb', a CSS colour name or three numbers "
            "[r, g, b], got 'navyy'"
        ),
    ),
    (("words", 3), "string param 'words' needs a string, got 3"),
    (("on", "yes", "main"), "bool param 'main.on' needs true or false, got 'yes'"),
]


@pytest.mark.parametrize("call,error", REFUSED)
def test_a_param_the_scene_cannot_take_raises_on_the_wire(call, error, monkeypatch):
    shim = load()

    @shim.tick
    def hook(tick):
        shim.set_param(*call)

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    with pytest.raises(ValueError, match=re.escape(error)):
        drive(shim, scene_of(shim), ticks, monkeypatch)


def layered(shim):
    """scene_of with a text layer `title` that sings, and an image layer
    `icon` of the image asset a, beside the image asset b."""
    scene = scene_of(shim)
    scene.assets = [shim.Asset("a", "a.png"), shim.Asset("b", "b.png")]
    scene.passes[0].layers = [
        shim.Layer("title", "hi", karaoke={}),
        shim.Layer("icon", image="a"),
    ]
    return scene


def test_a_layer_set_goes_out_in_the_form_of_its_field(monkeypatch):
    shim = load()

    @shim.tick
    def hook(tick):
        shim.layers.title.x = 10
        shim.layers.title.y = None
        shim.layers["title"]["karaoke"]["sung"] = 2.5
        shim.layers.title.karaoke.visible = False
        shim.layers.title.color = [1, 0, 0]
        shim.layers.title.color = "#ff0000"
        shim.layers.title.color = "Transparent"
        shim.layers.title.color = {"from": [0, 0, 1], "to": "navy"}
        shim.layers.title.text = "ho"
        shim.layers.icon.image = "b"

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, layered(shim), ticks, monkeypatch)
    # a colour written as text, a gradient as its JSON, keeps its form
    assert messages(rest) == [
        (1, "title.x", [10.0]),
        (1, "title.y", []),
        (1, "title.karaoke.sung", [2.5]),
        (1, "title.karaoke.visible", [0.0]),
        (1, "title.color", [1.0, 0.0, 0.0]),
        (3, "title.color", "#ff0000"),
        (3, "title.color", "Transparent"),
        (3, "title.color", '{"from":[0,0,1],"to":"navy"}'),
        (3, "title.text", "ho"),
        (3, "icon.image", "b"),
        END,
    ]


def test_a_shape_set_goes_out_in_the_form_of_its_field(monkeypatch):
    shim = load()
    scene = scene_of(shim)
    scene.passes[0].layers = [
        shim.Layer("frame", shape="rect"),
        shim.Layer("scope", shape="line", points=[[0, 0], [1, 1]]),
    ]

    @shim.tick
    def hook(tick):
        shim.layers.scope.points = [(0, 16), (8, 4.5)]
        shim.layers.frame.border.color = "Red"
        shim.layers.frame.border.width = 2
        shim.layers.frame.width = None

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene, ticks, monkeypatch)
    # points go as an array, x and y in turn
    assert messages(rest) == [
        (0, "scope.points", [0.0, 16.0, 8.0, 4.5]),
        (3, "frame.border.color", "Red"),
        (1, "frame.border.width", [2.0]),
        (1, "frame.width", []),
        END,
    ]


# ghee.layers' refusals: (layer, field, value, effect), and the error's key
LAYER_REFUSED = [
    (("nope", "x", 1, None), "layer 'nope' is not a layer this scene declares"),
    (("title", "sung", 1, "glow"), "layer 'title' has no effect 'glow'"),
    (("title", "colour", "#ff0000", None), "layer 'title' has no field 'colour'"),
    (
        ("title", "font_size", 12, None),
        "layer 'title' font_size changes with a rebuild",
    ),
    (("title", "x", "left", None), "layer 'title' x needs a number"),
]


@pytest.mark.parametrize("call,error", LAYER_REFUSED)
def test_a_layer_set_the_scene_cannot_take_raises_on_the_wire(call, error, monkeypatch):
    shim = load()
    name, field, value, effect = call

    @shim.tick
    def hook(tick):
        layer = shim.layers[name]
        (layer[effect] if effect else layer)[field] = value

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    with pytest.raises(ValueError, match=re.escape(error)):
        drive(shim, layered(shim), ticks, monkeypatch)


def test_a_layer_set_outside_run_says_where_it_belongs():
    shim = load()
    with pytest.raises(RuntimeError) as raised:
        shim.layers.title.x = 10
    assert str(raised.value) == (
        "ghee.layers sets a layer of the scene that run() runs: set it in a tick "
        "or a feed"
    )


LANDS = """#!{python}
import ghee


@ghee.tick
def hook(t):
    ghee.layers.t.color = "#00ff00"


ghee.run(
    ghee.Scene(
        passes=[
            ghee.Pass(
                "main",
                "void main(){{ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }}",
                layers=[ghee.Layer("t", "I", font_size=24, color="#ff0000")],
            )
        ]
    )
)
"""


def test_a_layer_set_of_a_scene_that_ghee_runs_lands_in_the_frame(tmp_path):
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    if not glob.glob("/dev/dri/renderD*"):
        pytest.skip("no GPU render node to run the scene")
    path = tmp_path / "lands.py"
    path.write_text(LANDS.format(python=sys.executable))
    path.chmod(0o755)
    ran = subprocess.run(
        [str(CLI), "render", str(path), "-o", "-", "--size", "32x32"]
        + ["--frames", "1", "--timestep", "0"],
        capture_output=True,
        check=True,
    )
    px = ran.stdout
    n = len(px) // (32 * 32)
    # the words drawn green, where the scene drew them red
    assert [max(px[c::n]) for c in range(3)] == [0, 255, 0]


def test_measure_outside_run_says_where_it_belongs():
    shim = load()
    with pytest.raises(RuntimeError) as raised:
        shim.measure("hi")
    assert str(raised.value) == (
        "ghee.measure lays words out in the scene that run() runs: call it in a "
        "tick or a feed"
    )


def test_a_scene_under_ghee_renders_at_ghees_size(monkeypatch):
    shim = load()
    monkeypatch.setenv(shim.SIZE, "96x32")
    assert shim.render_size() == (96, 32)
    monkeypatch.setenv(shim.SIZE, "wide")
    with pytest.raises(RuntimeError) as raised:
        shim.render_size()
    assert str(raised.value) == "$GHEE_SIZE names no WxH render size"


def test_measure_under_ghee_lays_words_out_in_the_render_size(monkeypatch):
    """A scene that ghee runs measures on its last pass, at the render size
    that ghee gives it, and reads ghee's refusal of a font."""
    shim = load()
    got, listed = [], []

    @shim.tick
    def hook(tick):
        got.append(shim.measure("Hi there", font_size=12))
        got.append(shim.measure("Hi there", font_size=24))
        got.append(shim.measure("Hi there"))
        with pytest.raises(ValueError) as raised:
            shim.measure("Hi", font_family="serif")
        got.append(str(raised.value))
        fonts = ["Pixel Operator 8", "DejaVu Sans"]
        listed.append(shim.measure("Hi\u00a7", font_size=16, font_family=fonts))
        listed.append(shim.measure("Hi", font_size=16, font_family=fonts[0]))
        listed.append(shim.measure("\u00a7", font_size=16, font_family=fonts[1]))
        with pytest.raises(ValueError) as raised:
            shim.measure("Hi", font_family=[])
        listed.append(str(raised.value))

    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    for size in ("96x32", "192x64"):
        monkeypatch.setenv(shim.SIZE, size)
        drive(shim, scene_of(shim), ticks, monkeypatch)
    small, large, fit, refused = got[:4]
    assert large == pytest.approx((2 * small[0], 2 * small[1]))
    # the words fit a 96x32 pass, wider than tall, and twice the pass twice
    assert 32 < fit[0] <= 96 and fit[1] <= 32
    assert got[6] == pytest.approx((2 * fit[0], 2 * fit[1]))
    assert refused.startswith("font_family 'serif' is no asset of the scene")
    # each character of a list draws from the first font that holds it, on
    # the first font's line
    whole, pixel, dejavu, empty = listed[:4]
    assert whole == pytest.approx((pixel[0] + dejavu[0], pixel[1]))
    assert empty.endswith("or a list of them, got []")


def test_the_tick_line_carries_the_params(monkeypatch):
    shim = load()
    seen = []

    @shim.tick
    def hook(tick):
        seen.append((tick.frame, tick.time, tick.dt, tick.params))

    params = {"text": "GHEE", "gain": 0.5, "tint": [1.0, 0.0, 0.5]}
    ticks = [
        {"frame": 3, "time": 0.25, "dt": 0.016, "params": params},
        {"frame": 4, "time": 0.5, "dt": 0.016},
    ]
    drive(shim, scene_of(shim), ticks, monkeypatch)
    assert seen == [(3, 0.25, 0.016, params), (4, 0.5, 0.016, {})]


# --------------------------------------------------------------- the package


SURFACE = [
    "Arg",
    "ArrayInput",
    "Asset",
    "Channel",
    "Layer",
    "Param",
    "Pass",
    "Scene",
    "Tick",
    "args",
    "draw",
    "feed",
    "gradient",
    "layers",
    "measure",
    "render_size",
    "run",
    "set_param",
    "skip",
    "tick",
]


def test_the_package_is_the_shim_plus_the_native_half():
    """`import ghee` is the whole surface. pip is the only way to it."""
    import ghee
    from ghee import _shim

    for name in SURFACE:
        assert getattr(ghee, name) is getattr(_shim, name), name
    assert set(ghee.__all__) == set(SURFACE) | {"Handle", "open", "__version__"}
    assert ghee.__version__


def test_the_stubs_name_what_the_native_module_takes():
    """The wheel marks itself typed, and _native.pyi gives open, each Handle
    method and each Drawing call the parameters that the module itself
    takes."""
    import ast
    import inspect
    import pathlib

    import ghee._native as native

    here = pathlib.Path(native.__file__).parent
    assert (here / "py.typed").is_file()
    stub = ast.parse((here / "_native.pyi").read_text())

    def params(fn):
        a = fn.args
        return [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs if p.arg != "self"]

    def runtime(fn):
        return [p for p in inspect.signature(fn).parameters if p != "self"]

    funcs = {f.name: f for f in stub.body if isinstance(f, ast.FunctionDef)}
    assert params(funcs["open"]) == runtime(native.open)
    (handle,) = [
        c for c in stub.body if isinstance(c, ast.ClassDef) and c.name == "Handle"
    ]
    methods = {f.name: f for f in handle.body if isinstance(f, ast.FunctionDef)}
    public = {n for n in dir(native.Handle) if not n.startswith("_")}
    assert {n for n in methods if not n.startswith("_")} == public
    for name in public - {"size", "components", "layers", "brightness"}:
        assert params(methods[name]) == runtime(getattr(native.Handle, name)), name
    (drawing,) = [
        c for c in stub.body if isinstance(c, ast.ClassDef) and c.name == "Drawing"
    ]
    calls = {f.name: f for f in drawing.body if isinstance(f, ast.FunctionDef)}
    public = {n for n in dir(native.Drawing) if not n.startswith("_")}
    assert {n for n in calls if not n.startswith("_")} == public
    for name in public:
        assert params(calls[name]) == runtime(getattr(native.Drawing, name)), name


def test_open_without_the_native_module_names_it(monkeypatch):
    """A shim run under ghee needs no native module until it sets a layer.
    open() needs it."""
    import ghee

    class Missing:
        def find_spec(self, name, path=None, target=None):
            if name == "ghee._native":
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.delitem(sys.modules, "ghee._native", raising=False)
    monkeypatch.delattr(ghee, "_native", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Missing(), *sys.meta_path])
    with pytest.raises(ImportError, match="ghee._native is missing") as caught:
        assert ghee.open
    assert "ghee-render" in str(caught.value)
    # the shim itself is still there
    assert ghee.Scene(passes=[]).to_dict() == {"version": 1, "passes": []}
