"""The feeder shim (ghee._shim): the scene it emits, the probe, the wire.

The shim is the Python half of the ghee-render wheel. Each test loads a copy
of it from the installed file, because the feed and hook registries are module
state.

    pytest python/tests/test_shim.py
"""

import importlib.util
import io
import itertools
import json
import re
import struct
import sys

import pytest

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

    Kind 4 carries one byte for each value, and kind 3 carries text. Kinds 2
    and 5 are the single byte.
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
        if kind == 3:
            out.append((3, name, buf[i : i + count].decode()))
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
            "code": "void main(){}",
            "size": [64, 32],
            "mipmaps": True,
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
                mipmaps=True,
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


def words_scene(shim, texts):
    return shim.Scene(
        passes=[shim.Pass("main", "void main(){}")],
        assets=[shim.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
        texts=texts,
    )


def test_a_text_carries_its_words():
    shim = load()
    text = shim.Text("words", "sans", "hi", align="center")
    built = words_scene(shim, [text]).to_dict()
    assert built == {
        "version": 1,
        "passes": [{"name": "main", "code": "void main(){}"}],
        "assets": {"sans": {"file": "fonts/DejaVuSans-Latin.ttf"}},
        "texts": {"words": {"value": "hi", "font": "sans", "align": "center"}},
    }
    # left is the default, and the document leaves it out
    assert shim.Text("words", "sans").to_dict() == {"value": "", "font": "sans"}
    assert "texts" not in words_scene(shim, []).to_dict()
    # a font file needs no asset, and a list mixes both
    for font in ("DejaVuSans.ttf", ["sans", "Noto.OTF"]):
        doc = words_scene(shim, [shim.Text("words", font)]).to_dict()
        assert doc["texts"]["words"]["font"] == font


def test_an_asset_is_its_file():
    shim = load()
    scene = shim.Scene(
        passes=[shim.Pass("main", "void main(){}")],
        assets=[shim.Asset("logo", "logo.png")],
    )
    assert scene.to_dict()["assets"] == {"logo": {"file": "logo.png"}}
    assert "assets" not in shim.Scene(passes=[]).to_dict()
    with pytest.raises(ValueError, match="asset 'logo' 'file' needs a filename"):
        shim.Asset("logo", "")
    with pytest.raises(ValueError, match="asset 'logo' is declared twice"):
        shim.Scene(assets=[shim.Asset("logo", "a.png"), shim.Asset("logo", "b.png")])


def test_a_text_says_what_it_needs():
    shim = load()
    with pytest.raises(ValueError, match="align needs 'left', 'center' or 'right'"):
        shim.Text("words", "sans", align="middle")
    with pytest.raises(ValueError, match="'font' needs the name of a font asset"):
        shim.Text("words", "")
    with pytest.raises(ValueError, match="value needs words in a string, got 3"):
        shim.Text("words", "sans", 3)
    for ty, value in (("float", 1.0), ("string", "hi")):
        with pytest.raises(
            ValueError,
            match="scene param 'gain' has the name of text 'gain', which holds its "
            "words in its 'value'",
        ):
            shim.Scene(
                params={"gain": shim.Param("gain", ty, value)},
                texts=[shim.Text("gain", "sans.ttf")],
            )
    with pytest.raises(
        ValueError,
        match="param 'gain' in pass 'main' has the name of text 'gain', which holds",
    ):
        shim.Scene(
            passes=[
                shim.Pass(
                    "main",
                    "void main(){}",
                    params={"gain": shim.Param("gain", "float", 1.0)},
                )
            ],
            texts=[shim.Text("gain", "sans.ttf")],
        )
    with pytest.raises(ValueError, match="text 'words' is declared twice"):
        words_scene(shim, [shim.Text("words", "sans"), shim.Text("words", "sans")])
    with pytest.raises(
        ValueError,
        match="text 'words' font 'serif' is no asset of the scene and no .ttf or .otf file",
    ):
        words_scene(shim, [shim.Text("words", "serif")])


INK = "void main(){ gl_FragColor = vec4(glyph_cover()); }"
PLACE = "vec2 glyph_place(vec2 em) { return em * 24.0; }"


def drawn_scene(shim, passes, text=None):
    return shim.Scene(
        passes=passes,
        assets=[shim.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
        texts=[text or shim.Text("words", "sans", "hi")],
    )


def test_a_draw_and_its_text_take_the_scene_json_keys():
    shim = load()
    built = drawn_scene(
        shim,
        [
            shim.Pass(
                "main",
                "void main(){}",
                draws=[shim.Draw("words", INK, PLACE, blend="add", pad=0.25)],
            )
        ],
        shim.Text("words", "sans", "hi", fill="slug"),
    ).to_dict()
    recorded = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "code": "void main(){}",
                "draws": [
                    {
                        "text": "words",
                        "vertex": PLACE,
                        "fragment": INK,
                        "blend": "add",
                        "pad": 0.25,
                    }
                ],
            }
        ],
        "assets": {"sans": {"file": "fonts/DejaVuSans-Latin.ttf"}},
        "texts": {"words": {"value": "hi", "font": "sans", "fill": "slug"}},
    }
    assert built == recorded
    assert json.dumps(built) == json.dumps(recorded)  # key order too


def test_a_draw_and_a_text_leave_their_defaults_out():
    shim = load()
    draw = shim.Draw("words", INK, PLACE)
    assert (draw.blend, draw.pad, draw.emoji) == ("over", 0.0, None)
    assert draw.to_dict() == {"text": "words", "vertex": PLACE, "fragment": INK}
    text = shim.Text("words", "sans")
    assert (text.value, text.fill, text.atlas) == ("", "mtsdf", None)
    assert text.to_dict() == {"value": "", "font": "sans"}
    assert "draws" not in shim.Pass("main", "void main(){}").to_dict()


def test_a_draw_with_no_vertex_sends_px_per_em_and_with_no_fragment_none():
    shim = load()
    draw = shim.Draw("words", px_per_em=48)
    assert draw.to_dict() == {"text": "words", "px_per_em": 48.0}
    assert json.dumps(draw.to_dict()) == '{"text": "words", "px_per_em": 48.0}'
    assert shim.Draw("words", vertex=PLACE).to_dict() == {
        "text": "words",
        "vertex": PLACE,
    }


def test_a_draw_sends_emoji_when_it_is_set():
    shim = load()
    for emoji in (True, False):
        assert shim.Draw("words", INK, PLACE, emoji=emoji).to_dict() == {
            "text": "words",
            "vertex": PLACE,
            "fragment": INK,
            "emoji": emoji,
        }
    with pytest.raises(ValueError, match="emoji needs True, False or None, got 1"):
        shim.Draw("words", INK, PLACE, emoji=1)


def test_a_pass_of_draws_alone_has_no_code():
    shim = load()
    layer = shim.Pass("ink", draws=[shim.Draw("words", INK, PLACE)])
    assert layer.to_dict() == {
        "name": "ink",
        "draws": [{"text": "words", "vertex": PLACE, "fragment": INK}],
    }
    assert "code" not in drawn_scene(shim, [layer]).to_dict()["passes"][0]
    with pytest.raises(ValueError, match="pass 'main' has no code and no draws"):
        shim.Pass("main")
    # a filter pass takes its code from its filter, and no draws
    assert shim.Pass("dither", filter="dither").to_dict()["code"] == ""
    with pytest.raises(ValueError, match="a filter pass takes no draws"):
        shim.Pass("dither", filter="dither", draws=[shim.Draw("words", INK, PLACE)])


def test_a_draw_says_what_it_needs():
    shim = load()
    with pytest.raises(ValueError, match="blend needs 'over', 'add' or 'replace'"):
        shim.Draw("words", INK, PLACE, blend="mix")
    for pad in (-0.1, float("nan"), True, "0.1"):
        with pytest.raises(ValueError, match="pad needs a number of em, 0 or more"):
            shim.Draw("words", INK, PLACE, pad=pad)
    with pytest.raises(ValueError, match="'fragment' needs fragment GLSL"):
        shim.Draw("words", "  ", PLACE)
    with pytest.raises(ValueError, match="'vertex' needs vertex GLSL"):
        shim.Draw("words", INK, " ")
    with pytest.raises(
        ValueError,
        match="needs 'px_per_em', the pixels per em of the words centred in the "
        "pass, or 'vertex', GLSL that defines",
    ):
        shim.Draw("words", INK)
    with pytest.raises(ValueError, match="takes 'vertex' or 'px_per_em', got both"):
        shim.Draw("words", INK, PLACE, px_per_em=32)
    for px in (0, -1.0, float("inf"), True, "32"):
        with pytest.raises(
            ValueError, match="px_per_em needs a number of pixels above 0"
        ):
            shim.Draw("words", px_per_em=px)
    with pytest.raises(ValueError, match="'text' needs the name of a scene text"):
        shim.Draw("", INK, PLACE)
    with pytest.raises(
        ValueError,
        match="pass 'main' draw 0 names text 'title', which the scene does not declare",
    ):
        drawn_scene(shim, [shim.Pass("main", draws=[shim.Draw("title", INK, PLACE)])])


def test_a_text_fill_and_atlas_say_what_they_need():
    shim = load()
    for fill in ("sdf", "outline"):
        with pytest.raises(
            ValueError,
            match=f"fill needs 'mtsdf', fast and with glyph_dist and glyph_soft, "
            f"or 'slug', exact edges at any size, got '{fill}'",
        ):
            shim.Text("words", "sans", fill=fill)
    with pytest.raises(
        ValueError,
        match="""takes no 'atlas' with "fill": "slug", which reads the curves""",
    ):
        shim.Text("words", "sans", fill="slug", atlas={"px_per_em": 32})
    with pytest.raises(ValueError, match="atlas takes 'px_per_em', got 'px'"):
        shim.Text("words", "sans", atlas={"px": 32})
    with pytest.raises(
        ValueError,
        match="atlas takes no 'range': the field reaches the largest 'pad' of the "
        "text's draws",
    ):
        shim.Text("words", "sans", atlas={"px_per_em": 32, "range": 6})
    for n in (0, 1.5, True):
        with pytest.raises(ValueError, match="atlas px_per_em needs a whole number"):
            shim.Text("words", "sans", atlas={"px_per_em": n})
    with pytest.raises(ValueError, match="atlas needs a dict with 'px_per_em'"):
        shim.Text("words", "sans", atlas=32)


def test_a_font_list_goes_out_as_given():
    shim = load()
    scene = shim.Scene(
        passes=[shim.Pass("main", "void main(){}")],
        assets=[shim.Asset("sans", "sans.ttf"), shim.Asset("emoji", "emoji.ttf")],
        texts=[shim.Text("words", ("sans", "emoji"), "hi")],
    )
    assert scene.to_dict()["texts"] == {
        "words": {"value": "hi", "font": ["sans", "emoji"]}
    }
    with pytest.raises(ValueError, match="text 'words' font 'serif' is no asset"):
        drawn_scene(shim, [], shim.Text("words", ["sans", "serif"]))
    with pytest.raises(ValueError, match="'font' needs the name of a font asset"):
        shim.Text("words", ["sans", ""])


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


def test_a_text_takes_its_words_through_set_param(monkeypatch):
    """ghee declares the string param of a text's name."""
    shim = load()

    @shim.tick
    def hook(tick):
        shim.set_param("title", "ho")

    scene = scene_of(shim)
    scene.texts = [shim.Text("title", "sans.ttf", "hi")]
    ticks = [{"frame": 0, "time": 0.0, "dt": 0.0, "params": {}}]
    _, rest = drive(shim, scene, ticks, monkeypatch)
    assert messages(rest) == [(3, "title", "ho"), END]


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
    (("tint", [1.0, 0.0]), "color param 'tint' needs three numbers [r, g, b]"),
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


# --------------------------------------------------------------- the package


SURFACE = [
    "Arg",
    "ArrayInput",
    "Asset",
    "Channel",
    "Draw",
    "Param",
    "Pass",
    "Scene",
    "Text",
    "Tick",
    "args",
    "feed",
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


def test_open_without_the_native_module_names_it(monkeypatch):
    """A shim run needs no native module. open() is what needs it."""
    import ghee

    class Missing:
        def find_spec(self, name, path=None, target=None):
            if name == "ghee._native":
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    monkeypatch.delitem(sys.modules, "ghee._native", raising=False)
    monkeypatch.delattr(ghee, "_native", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Missing(), *sys.meta_path])
    with pytest.raises(ImportError, match="ghee._native is missing") as caught:
        assert ghee.open
    assert "ghee-render" in str(caught.value)
    # the shim itself is still there
    assert ghee.Scene(passes=[]).to_dict() == {"version": 1, "passes": []}
