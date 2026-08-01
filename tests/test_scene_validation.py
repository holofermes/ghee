"""What a malformed or hostile scene does: rejected at parse time, with the
scene, the pass and the field in the message. Plus the sinks and drivers that
fail on the same input. All pure Python, no GL device."""

import ctypes
import importlib
import io
import json
import queue
import subprocess
import sys
import types

import numpy as np
import pytest

from ghee import Param, Pass, Scene

GOOD = "void main(){ gl_FragColor = vec4(1.0); }"
COMMON_TEXT = "float half(float x){ return x * 0.5; }"


def _scene(*passes, **kw):
    return {"version": 1, "passes": list(passes), **kw}


# task 1: a scene file reference never escapes its scene directory


def test_shader_key_without_base_dir_raises():
    with pytest.raises(ValueError) as e:
        Scene.from_dict(_scene({"name": "main", "shader": "/etc/hostname"}))
    assert "shader" in str(e.value) and "main" in str(e.value)


def test_shader_outside_base_dir_raises(tmp_path):
    (tmp_path / "in.glsl").write_text(GOOD)
    secret = tmp_path.parent / "secret.glsl"
    secret.write_text("stolen")
    for path in ("../secret.glsl", str(secret), "sub/../../secret.glsl"):
        with pytest.raises(ValueError, match="outside the scene directory"):
            Scene.from_dict(
                _scene({"name": "main", "shader": path}), base_dir=str(tmp_path)
            )


def test_shader_inside_base_dir_still_loads(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "in.glsl").write_text(GOOD)
    sc = Scene.from_dict(
        _scene({"name": "main", "shader": "sub/in.glsl"}), base_dir=str(tmp_path)
    )
    assert sc.passes[0].code == GOOD


def test_serialized_scene_reloads_without_a_directory():
    from ghee import load_scene
    from ghee.console import examples_dir

    scene, _ = load_scene(f"{examples_dir()}/bars.scene.json")
    d = scene.to_dict()
    assert "shader" not in json.dumps(d)  # what the console posts back: code, inline
    assert Scene.from_dict(d).passes[0].code == scene.passes[0].code


def test_missing_shader_file_names_scene_and_pass(tmp_path):
    p = tmp_path / "x.scene.json"
    p.write_text(json.dumps(_scene({"name": "main", "shader": "gone.glsl"})))
    with pytest.raises(ValueError) as e:
        Scene.load(str(p))
    msg = str(e.value)
    assert str(p) in msg and "main" in msg and "gone.glsl" in msg


# task 2: param values are typed


def test_float_and_int_params_need_numbers():
    for t in ("float", "int"):
        with pytest.raises(ValueError, match="needs a number"):
            Param("g", t, "<img src=x onerror=alert(1)>")
        with pytest.raises(ValueError, match="needs a number"):
            Param("g", t, True)
        with pytest.raises(ValueError, match="needs a number"):
            Param("g", t, None)
        with pytest.raises(ValueError, match="needs a number"):
            Param("g", t, [1, 2])
    assert Param("g", "float", 1).value == 1.0
    assert Param("g", "int", 3).value == 3


def test_param_range_metadata_needs_numbers():
    Param("g", "float", 0.5, min=0, max=1, step=0.1)  # constructs fine
    Param("g", "float", 0.5)  # all None
    for k in ("min", "max", "step"):
        with pytest.raises(ValueError, match=k):
            Param("g", "float", 0.5, **{k: "lo"})


def test_unknown_param_type_raises_at_parse_time():
    with pytest.raises(ValueError, match="unknown type"):
        Param("g", "flaot", 0.5)
    with pytest.raises(ValueError, match="unknown type"):
        Scene.from_dict(
            _scene(
                {"name": "m", "code": GOOD, "params": {"g": {"type": "x", "value": 1}}}
            )
        )


# task 3: array and pass sizes are bounded


def test_array_size_is_bounded():
    from ghee import ArrayInput

    ArrayInput("a", 16)  # constructs fine
    ArrayInput("a", (96, 16))
    for bad in (0, -4, 1e11, 100000, "16", 4.5, None):
        with pytest.raises(ValueError):
            ArrayInput("a", bad)
    for bad in ((0, 4), (4, 0), (100000, 3), (8, 4, 2), (8, "4")):
        with pytest.raises(ValueError):
            ArrayInput("a", bad)


def test_huge_array_rejected_from_dict():
    with pytest.raises(ValueError, match="size"):
        Scene.from_dict(
            _scene({"name": "m", "code": GOOD, "arrays": [{"name": "a", "size": 1e11}]})
        )


def test_2d_array_texel_cap():
    from ghee import ArrayInput

    ArrayInput("a", (4096, 4096))  # exactly the cap
    with pytest.raises(ValueError, match="cap"):
        ArrayInput("a", (8192, 4096))


def test_pass_size_is_bounded():
    Pass(name="p", code=GOOD, size=[32, 18])
    with pytest.raises(ValueError, match="16384"):
        Pass(name="p", code=GOOD, size=[100000, 100000])
    with pytest.raises(ValueError, match="cap"):
        Pass(name="p", code=GOOD, size=[16384, 16384])


# task 4: a pass name is a buffer name, so it is unique


def test_duplicate_pass_names_rejected():
    with pytest.raises(ValueError, match="twice"):
        Scene(passes=[Pass(name="a", code=GOOD), Pass(name="a", code=GOOD)])
    with pytest.raises(ValueError, match="twice"):
        Scene.from_dict(
            _scene({"name": "a", "code": GOOD}, {"name": "a", "code": GOOD})
        )


def test_filter_pass_never_collides_with_a_scene_pass():
    from ghee import apply_filters

    sc = Scene(passes=[Pass(name="dither", code=GOOD)])
    apply_filters(sc, ["dither"])
    assert [p.name for p in sc.passes] == ["dither", "dither1"]
    Scene(passes=sc.passes)  # the result is still a valid scene


# task 5: one ValueError, carrying scene, pass and field


def test_malformed_scene_json_reports_context(tmp_path):
    bad = {
        "param 'g' has no 'type'": _scene(
            {"name": "m", "code": GOOD, "params": {"g": {"value": 1}}}
        ),
        "'passes' needs a list": {"version": 1, "passes": "oops"},
        "neither 'code'": _scene({"name": "m"}),
        "needs a list": _scene({"name": "m", "code": GOOD, "arrays": "oops"}),
        "a channel needs": _scene({"name": "m", "code": GOOD, "channels": ["m"]}),
    }
    for fragment, d in bad.items():
        p = tmp_path / "x.scene.json"
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError) as e:
            Scene.load(str(p))
        msg = str(e.value)
        assert fragment in msg, msg
        assert str(p) in msg, msg


def test_invalid_json_reports_the_path(tmp_path):
    p = tmp_path / "x.scene.json"
    p.write_text("{oops")
    with pytest.raises(ValueError, match="invalid JSON"):
        Scene.load(str(p))


def test_valid_scenes_still_load():
    from ghee import load_scene
    from ghee.console import examples_dir

    for name in ("bars", "trail", "twopass", "blur", "echo", "mix"):
        scene, _ = load_scene(f"{examples_dir()}/{name}.scene.json")
        assert scene.passes and all(p.code for p in scene.passes)


# task 6: Ctrl-C out of the console is not a traceback


def test_serve_swallows_keyboard_interrupt(monkeypatch):
    from ghee import cli
    from ghee.console import server

    def boom(**kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "run", boom)
    cli.main(["serve", "--demo", "--size", "64x48"])


def test_serve_empty_scenes_dir_exits_with_the_message(tmp_path):
    from ghee.cli import main

    with pytest.raises(SystemExit) as e:
        main(["serve", str(tmp_path), "--port", "0"])
    assert "no .scene.json" in str(e.value.code)


def test_serve_renderer_failure_exits_with_the_message(tmp_path):
    from ghee.cli import main

    (tmp_path / "x.scene.json").write_text(
        json.dumps(_scene({"name": "main", "code": GOOD}))
    )
    with pytest.raises(SystemExit) as e:
        main(
            [
                "serve",
                str(tmp_path),
                "--port",
                "0",
                "--lib",
                str(tmp_path / "nope.so"),
            ]
        )
    assert "renderer:" in str(e.value.code)


def test_serve_failure_prints_no_traceback(tmp_path):
    p = subprocess.run(
        [sys.executable, "-m", "ghee.cli", "serve", str(tmp_path), "--port", "0"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 1
    assert "no .scene.json" in p.stderr
    assert "Traceback" not in p.stderr


# task 7: --timestep pins engine time on the H264 path too


class _FakeRenderer:
    def __init__(self, w, h, components=4, lib_path=None, dri_device=None, **kw):
        self.width, self.height, self.components = w, h, components
        self.times = []
        self.dts = []
        self.scene_params = []
        self.handle = 1
        self.lib = None

    def setup(self, scene):
        return ""

    def set_time(self, t):
        self.times.append(t)

    def set_scene_param(self, name, ptype, value):
        self.scene_params.append((name, ptype, value))

    def set_dt(self, value):
        self.dts.append(value)

    def render(self, provider=None, out=None, out_size=None):
        return np.zeros((4, 4, 4), np.uint8) if out is None else out

    def draw(self, provider=None):
        pass

    def _feed(self, provider):
        pass

    def last_error(self):
        return "v4l2 said no"

    def close(self):
        pass


class _FakeEncoder:
    def feed(self, rgb):
        pass

    def get_au(self, timeout=2.0):
        return b"", False

    def drain(self):
        return iter(())

    def close(self):
        pass


def _h264_run(monkeypatch, tmp_path, timestep, encoder=None, lib=None):
    """`render --h264` end to end against a fake renderer and encoder, so what
    is under test is the loop's own time and DT bookkeeping."""
    from ghee import cli, h264
    from ghee.engine import Renderer  # noqa: F401  the name cli imports

    made = {}

    def fake_renderer(*a, **kw):
        r = _FakeRenderer(*a, **kw)
        r.lib = lib
        if encoder is not None:
            encoder.r = r
        made["r"] = r
        return r

    monkeypatch.setattr("ghee.engine.Renderer", fake_renderer)
    monkeypatch.setattr(
        h264, "make_encoder", lambda *a, **kw: encoder or _FakeEncoder()
    )
    (tmp_path / "s.scene.json").write_text(
        json.dumps(_scene({"name": "main", "code": GOOD})), encoding="utf-8"
    )
    argv = [
        "render",
        str(tmp_path / "s.scene.json"),
        "--h264",
        str(tmp_path / "out.h264"),
        "--size",
        "4x4",
        "--fps",
        "1000",
        "--bitrate",
        "1000",
        "--frames",
        "3",
    ]
    if timestep is not None:
        argv += ["--timestep", str(timestep)]
    cli.main(argv)
    return made["r"]


def test_h264_honors_timestep(monkeypatch, tmp_path):
    r = _h264_run(monkeypatch, tmp_path, 0.5)
    assert r.times == [0.0, 0.5, 1.0]


def test_h264_without_timestep_leaves_engine_time_alone(monkeypatch, tmp_path):
    assert _h264_run(monkeypatch, tmp_path, None).times == []


def test_pinned_timestep_publishes_it_as_dt(monkeypatch, tmp_path):
    assert _h264_run(monkeypatch, tmp_path, 0.5).dts == [0.5, 0.5, 0.5]


def test_unpinned_dt_is_the_measured_frame_period(monkeypatch, tmp_path):
    dts = _h264_run(monkeypatch, tmp_path, None).dts
    assert len(dts) == 2  # frame 0 has no previous frame to measure against
    assert all(0.0 < v < 5.0 for v in dts)


def test_dt_is_never_a_scene_param(monkeypatch, tmp_path):
    """It rides the draw as an engine uniform, so it must not reach the
    per-pass set_scene_param fan-out."""
    r = _h264_run(monkeypatch, tmp_path, 0.5)
    assert [n for n, _, _ in r.scene_params if n == "DT"] == []


# the hardware encoder's tail: what ghee_encoder_flush still owes is written out


class _FakeEncoderLib:
    """Stands in for libghee: hands back a scripted [(rc, au), ...] per flush."""

    def __init__(self, script, render_rc=1):
        self.script = list(script)
        self.render_rc = render_rc
        self.flushes = 0
        self.destroyed = 0
        self._keep = []

    def _set(self, nal, ln, key, au):
        buf = ctypes.create_string_buffer(au)
        self._keep.append(buf)
        nal._obj.contents = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)).contents
        ln._obj.value = len(au)
        key._obj.value = 0

    def ghee_encoder_flush(self, scene, enc, nal, ln, key):
        self.flushes += 1
        rc, au = self.script.pop(0)
        if au is not None:
            self._set(nal, ln, key, au)
        return rc

    def ghee_encoder_encode_last(self, scene, enc, nal, ln, key):
        return self.render_rc

    def ghee_encoder_destroy(self, scene, enc):
        self.destroyed += 1


def _hw_encoder(lib):
    from ghee.h264 import HardwareH264

    enc = HardwareH264.__new__(HardwareH264)
    enc.r = types.SimpleNamespace(handle=1, lib=lib, last_error=lambda: "v4l2 said no")
    enc.handle = 2
    enc.w, enc.h = 4, 4
    return enc


def test_hardware_drain_yields_every_au_then_stops():
    lib = _FakeEncoderLib([(0, b"tail1"), (0, b"tail2"), (1, None)])
    assert list(_hw_encoder(lib).drain()) == [b"tail1", b"tail2"]
    assert lib.flushes == 3  # the 1 ends it: no spin past "drained"


def test_hardware_drain_surfaces_an_error_without_spinning():
    lib = _FakeEncoderLib([(0, b"tail1"), (-1, None)])
    enc = _hw_encoder(lib)
    with pytest.raises(RuntimeError, match="v4l2 said no"):
        list(enc.drain())
    assert lib.flushes == 2


def test_hardware_drain_without_an_encoder_is_a_no_op():
    lib = _FakeEncoderLib([])
    enc = _hw_encoder(lib)
    enc.handle = None
    assert list(enc.drain()) == []
    assert lib.flushes == 0


def test_render_h264_writes_the_hardware_tail(monkeypatch, tmp_path):
    lib = _FakeEncoderLib([(0, b"tail1"), (0, b"tail2"), (1, None)])
    enc = _hw_encoder(lib)
    _h264_run(monkeypatch, tmp_path, None, encoder=enc, lib=lib)
    assert (tmp_path / "out.h264").read_bytes() == b"tail1tail2"
    assert lib.destroyed == 1


# task 8: the raw paths write the frame's own buffer, byte for byte


def test_raw_stream_writes_exact_bytes():
    from ghee.output import RawStream

    frame = np.arange(6 * 5 * 4, dtype=np.uint8).reshape(5, 6, 4)
    buf = io.BytesIO()
    RawStream(buf).write(frame)
    assert buf.getvalue() == frame.tobytes()


def test_software_h264_feed_writes_exact_bytes():
    from ghee.h264 import SoftwareH264

    enc = SoftwareH264.__new__(SoftwareH264)
    enc.proc = types.SimpleNamespace(stdin=io.BytesIO())
    frame = np.arange(6 * 5 * 4, dtype=np.uint8).reshape(5, 6, 4)
    enc.feed(frame)
    enc.feed(frame.tobytes())
    assert enc.proc.stdin.getvalue() == frame.tobytes() * 2


# task 9: close() ends even when the encoder died


def test_pump_puts_the_sentinel_even_when_the_read_dies():
    from ghee.h264 import SoftwareH264

    class Dead:
        def read(self, n):
            raise OSError("gone")

    enc = SoftwareH264.__new__(SoftwareH264)
    enc._q = queue.Queue()
    enc.proc = types.SimpleNamespace(stdout=Dead())
    with pytest.raises(OSError):
        enc._pump()
    assert enc._q.get_nowait() is None


def test_close_kills_an_encoder_that_never_exits():
    from ghee.h264 import SoftwareH264

    class Proc:
        stdin = None
        killed = False

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)

        def kill(self):
            self.killed = True

    enc = SoftwareH264.__new__(SoftwareH264)
    enc._q = queue.Queue()  # no sentinel: the pump died before putting one
    enc.proc = Proc()
    enc.close(timeout=0.05)
    assert enc.proc.killed


# task 10: one bad filter file does not take the registry down


def test_malformed_filter_file_is_skipped(tmp_path, monkeypatch, capsys):
    from ghee import load_filters

    (tmp_path / "broken.filter.json").write_text("{not json")
    (tmp_path / "nameless.filter.json").write_text(json.dumps({"code": "x"}))
    (tmp_path / "invert.filter.json").write_text(
        json.dumps({"name": "invert", "code": GOOD})
    )
    monkeypatch.setenv("GHEE_FILTERS", str(tmp_path))
    reg = load_filters()
    assert "dither" in reg and "invert" in reg
    err = capsys.readouterr().err
    assert "broken.filter.json" in err and "nameless.filter.json" in err


# task 11: a filter name that does not exist is an error, wherever it came from


def test_scene_declared_unknown_filter_exits(tmp_path, monkeypatch):
    from ghee.cli import main

    (tmp_path / "typo.scene.json").write_text(
        json.dumps(_scene({"name": "main", "code": GOOD}, filters=["dithr"]))
    )
    monkeypatch.setenv("GHEE_SCENES", str(tmp_path))
    with pytest.raises(SystemExit, match="unknown filter: dithr"):
        main(["render", "typo", "-o", str(tmp_path / "x.png")])


# task 12: a missing extra says which extra


def test_missing_drivers_extra_names_the_extra(monkeypatch):
    import ghee.drivers as drivers

    def boom(name):
        raise ImportError("No module named 'spidev'")

    monkeypatch.setattr(drivers, "importlib", types.SimpleNamespace(import_module=boom))
    with pytest.raises(ImportError, match=r"ghee-render\[drivers\]"):
        drivers.get_driver("st7789")


# task 14: one ST7789 driver, any ST7789 panel


def _panel_stubs():
    """spidev and gpiod (2.x shape), enough to construct the driver. The bytes
    and the pin levels are test_drivers.py's job."""
    spidev = types.ModuleType("spidev")
    spidev.SpiDev = lambda: types.SimpleNamespace(
        open=lambda bus, dev: None,
        mode=0,
        max_speed_hz=0,
        writebytes2=lambda data: None,
        close=lambda: None,
    )
    line = types.ModuleType("gpiod.line")
    line.Direction = types.SimpleNamespace(OUTPUT="output")
    line.Value = types.SimpleNamespace(ACTIVE="active", INACTIVE="inactive")

    class Chip:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_info(self):
            return types.SimpleNamespace(label="pinctrl-bcm2835")

    gpiod = types.ModuleType("gpiod")
    gpiod.line = line
    gpiod.Chip = Chip
    gpiod.LineSettings = lambda **kw: kw
    gpiod.request_lines = lambda path, consumer=None, config=None: (
        types.SimpleNamespace(
            offsets=list(config), set_value=lambda off, v: None, release=lambda: None
        )
    )
    return {"spidev": spidev, "gpiod": gpiod, "gpiod.line": line}


def test_st7789_panel_geometry_is_a_constructor_argument(monkeypatch):
    from ghee.drivers.base import _REGISTRY

    for name, mod in _panel_stubs().items():
        monkeypatch.setitem(sys.modules, name, mod)
    for name in ("ghee.drivers.gpio", "ghee.drivers.st7789"):
        sys.modules.pop(name, None)
    try:
        mod = importlib.import_module("ghee.drivers.st7789")
        monkeypatch.setattr(
            sys.modules["ghee.drivers.gpio"], "_chip_paths", lambda: ["/dev/gpiochip0"]
        )
        d = mod.ST7789()  # --display st7789 takes no arguments
        assert (d.width, d.height) == (135, 240)
        assert (d.x_offset, d.y_offset) == (53, 40)
        assert (d.dc, d.rst, d.bl) == (25, None, 22)
        assert d.bl_pins is not None
        assert d._buf.size == 135 * 240 * 2

        other = mod.ST7789(
            width=240,
            height=240,
            x_offset=0,
            y_offset=80,
            dc=24,
            rst=27,
            backlight=None,
            baudrate=32000000,
        )
        assert (other.width, other.height) == (240, 240)
        assert (other.x_offset, other.y_offset) == (0, 80)
        assert (other.dc, other.rst, other.bl) == (24, 27, None)
        assert other._buf.size == 240 * 240 * 2
        assert other.bl_pins is None
        other.turn_on()  # a panel with no backlight pin: still a no-op, not a crash
        other.turn_off()
    finally:
        for name in ("ghee.drivers.gpio", "ghee.drivers.st7789"):
            sys.modules.pop(name, None)
        _REGISTRY.pop("st7789", None)


# task 15: a controller answers for its own arrays only


def test_text_example_provides_only_its_array(monkeypatch):
    from ghee.console import examples_dir
    from ghee.procedural import load_scene

    _, ctrl = load_scene(f"{examples_dir()}/text.py")
    assert ctrl.provide("text", 0, 0) is not None
    assert ctrl.provide("bars", 16, 0) is None


# task 16: a scene's common block is one block of GLSL, from one source


def test_common_needs_an_object_with_code_or_shader():
    for bad in (COMMON_TEXT, ["x"], 3, True):
        with pytest.raises(ValueError, match="'common' needs an object"):
            Scene.from_dict(_scene({"name": "m", "code": GOOD}, common=bad))


def test_common_takes_exactly_one_of_code_and_shader():
    for bad in ({}, {"code": COMMON_TEXT, "shader": "c.glsl"}):
        with pytest.raises(ValueError, match="exactly one"):
            Scene.from_dict(_scene({"name": "m", "code": GOOD}, common=bad))


def test_common_code_needs_text():
    with pytest.raises(ValueError, match="'common' needs GLSL text in 'code'"):
        Scene.from_dict(_scene({"name": "m", "code": GOOD}, common={"code": ["x"]}))
    with pytest.raises(ValueError, match="'common' needs GLSL text, got"):
        Scene(passes=[Pass(name="m", code=GOOD)], common=["x"])


def test_common_shader_without_base_dir_raises():
    with pytest.raises(ValueError) as e:
        Scene.from_dict(
            _scene({"name": "m", "code": GOOD}, common={"shader": "/etc/hostname"})
        )
    assert "common" in str(e.value) and "shader" in str(e.value)


def test_common_shader_never_escapes_the_scene_directory(tmp_path):
    secret = tmp_path.parent / "secret.glsl"
    secret.write_text("stolen")
    for path in ("../secret.glsl", str(secret)):
        with pytest.raises(ValueError, match="outside the scene directory"):
            Scene.from_dict(
                _scene({"name": "m", "code": GOOD}, common={"shader": path}),
                base_dir=str(tmp_path),
            )


def test_missing_common_file_names_the_scene_and_the_common(tmp_path):
    p = tmp_path / "x.scene.json"
    p.write_text(
        json.dumps(_scene({"name": "m", "code": GOOD}, common={"shader": "gone.glsl"}))
    )
    with pytest.raises(ValueError) as e:
        Scene.load(str(p))
    msg = str(e.value)
    assert str(p) in msg and "common" in msg and "gone.glsl" in msg
