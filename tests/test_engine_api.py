"""Engine API contract: staged setup, the readback modes, one texture per array
name, and the guards a bad scene has to hit instead of the GL. The ones that need
a GPU take the shared `renderer` fixture and skip cleanly without a device."""

import ctypes
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

from ghee import Channel, Pass, Scene
from ghee.cparam import NAME_MAX, create_c_parameter, set_c_name
from ghee.engine import (
    Renderer,
    bundled_lib,
    probe_display,
    probe_display_device,
    probe_dri_cards,
    probe_dri_devices,
)

from .helpers import scene

ENGINE_C = Path(__file__).resolve().parents[1] / "ghee" / "_engine" / "engine.c"

FLAT = "void main(){ gl_FragColor = vec4(0.5, 0.25, 0.75, 1.0); }"
BROKEN = "void main(){ gl_FragColor = no_such_symbol; }"
RAW = "#version 100\nprecision mediump float;\nvoid main(){ gl_FragColor = vec4(1.0); }"


class _CountingLib:
    """Forwards to the real engine, recording which arrays get uploaded."""

    def __init__(self, lib, calls):
        self._lib = lib
        self._calls = calls

    def __getattr__(self, name):
        fn = getattr(self._lib, name)
        if name != "ghee_mp_set_array":
            return fn
        calls = self._calls

        def counted(handle, arr_name, *rest):
            calls.append(arr_name)
            return fn(handle, arr_name, *rest)

        return counted


def test_probe_dri_devices_puts_the_override_first(monkeypatch):
    monkeypatch.setenv("GHEE_DRI", "/dev/dri/card9")
    devs = probe_dri_devices()
    assert devs[0] == "/dev/dri/card9"
    assert len(devs) == len(set(devs))
    assert "/dev/dri/card0" in devs


def test_probe_dri_devices_prefers_render_nodes(monkeypatch):
    # a primary node carries DRM master: a second concurrent process on a Pi 5's
    # card0 renders black, a render node has no master to contend for
    monkeypatch.delenv("GHEE_DRI", raising=False)
    devs = probe_dri_devices()
    assert devs[0] == "/dev/dri/renderD128"
    cards = [i for i, d in enumerate(devs) if "/card" in d]
    renders = [i for i, d in enumerate(devs) if "/renderD" in d]
    assert cards and renders
    assert max(renders) < min(cards)


def test_probe_dri_cards_are_primary_nodes(monkeypatch):
    monkeypatch.setenv("GHEE_DRI", "/dev/dri/card9")
    cards = probe_dri_cards()
    assert cards[0] == "/dev/dri/card9"
    monkeypatch.delenv("GHEE_DRI", raising=False)
    assert all("/renderD" not in d for d in probe_dri_cards())


def test_probe_display_device_names_the_card_it_found(monkeypatch):
    monkeypatch.delenv("GHEE_DRI", raising=False)
    found = probe_display_device()
    if found is None:
        pytest.skip("no connected display on this machine")
    dev, w, h = found
    # the connectors are not always on card0: on a Pi 5 they are on card1
    assert dev in probe_dri_cards()
    assert w > 0 and h > 0
    assert probe_display() == (w, h)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="not worth contending for the session's card as root",
)
def test_display_create_fails_once_without_drm_master():
    found = probe_display_device()
    if found is None:
        pytest.skip("no KMS display to contend for")
    dev, w, h = found
    before_the_master_attempt = ("cannot open DRI device", "eglInitialize failed")

    # An idle box has no master to lose the race against, so take it here rather
    # than relying on whatever happens to be running.
    try:
        holder = Renderer(w, h, display=True, dri_device=dev)
    except RuntimeError as e:
        msg = str(e)
        if any(s in msg for s in before_the_master_attempt):
            pytest.skip(f"{dev}: {msg}")
        assert "DRM master" in msg
        return

    try:
        with pytest.raises(RuntimeError) as e:
            Renderer(w, h, display=True, dri_device=dev)
    finally:
        holder.close()
    msg = str(e.value)
    if any(s in msg for s in before_the_master_attempt):
        pytest.skip(f"{dev}: {msg}")
    assert "DRM master" in msg


def test_a_flip_the_board_is_too_slow_to_finish_is_still_waited_for():
    """A page flip is fenced on the GPU finishing the frame, so its wait scales
    with the render rather than with the panel's refresh. Bound that wait near a
    frame time and a slow board leaves the flip pending, and every submit made
    while it stays pending is refused."""
    src = ENGINE_C.read_text()
    bound_ms = int(re.search(r"#define FLIP_STUCK_MS (\d+)", src).group(1))
    assert bound_ms >= 1000  # slower than any frame, i.e. only a stuck flip hits it

    present = src.split("void ghee_mp_present")[1]
    unreaped = present.split("!wait_flip(g)")[1].split("} else")[0]
    assert "gbm_surface_release_buffer(g->gbm_surface, bo)" in unreaped
    assert "drmModePageFlip" not in unreaped

    flipped = present.split("if (!drmModePageFlip")[1].split("} else")[0]
    assert "g->flip_failed = false;" in flipped


def test_jpeg_loader_is_gone():
    path = os.environ.get("GHEE_LIB") or bundled_lib()
    if not path:
        pytest.skip("no engine .so")
    lib = ctypes.CDLL(path)
    assert hasattr(lib, "ghee_mp_set_array")
    assert not hasattr(lib, "ghee_load_jpeg_texture")


def test_param_name_must_leave_room_for_the_terminator():
    cp = create_c_parameter({"type": "float", "value": 1.0})
    set_c_name(cp, "k" * NAME_MAX)
    assert cp.name == b"k" * NAME_MAX
    with pytest.raises(ValueError):
        set_c_name(cp, "k" * (NAME_MAX + 1))


def test_vector_and_texture_param_structs_have_no_producer():
    import ghee.cparam as cparam

    # colors and vecN reach the shader as decomposed float uniforms
    with pytest.raises(ValueError):
        create_c_parameter({"type": "vec3", "value": [0.0, 0.0, 0.0]})
    assert not hasattr(cparam, "CTextureParam")
    assert not hasattr(cparam.CParameterType, "TEXTURE")


class _SetupHarness:
    """Renderer.setup on a stub engine that refuses to compile, so setup returns
    either one of its own scene rejections or the stub's error. No GPU."""

    STUB = "stub engine: nothing compiles"

    def __init__(self):
        self.width, self.height = 64, 48
        self.error = ""
        self.error_line = None
        self.error_pass = None
        self.handle = None
        self.aborted = False
        harness = self

        class Lib:
            def ghee_mp_add_pass(self, handle, glsl):
                return -1

            def ghee_mp_abort(self, handle):
                harness.aborted = True

        self.lib = Lib()

    def max_texture_units(self):
        return 8

    def data_texture_linear(self):
        return True

    def last_error(self):
        return self.STUB

    def check(self, sc):
        return Renderer.setup(self, sc)


def test_channel_naming_no_pass_is_rejected():
    sc = Scene(
        passes=[
            Pass(name="a", code=FLAT),
            Pass(name="b", channels=[Channel("a"), Channel("typo")], code=FLAT),
        ]
    )
    err = _SetupHarness().check(sc)
    assert "'b'" in err and "channel 1" in err and "'typo'" in err


def test_channel_bound_to_nothing_is_allowed():
    # buffer=None is the deliberate unbound channel, not a broken reference
    h = _SetupHarness()
    sc = Scene(passes=[Pass(name="a", channels=[Channel(None)], code=FLAT)])
    assert h.check(sc) == h.STUB
    assert h.aborted


def test_setup_rejects_a_scene_with_no_passes(renderer):
    assert "no passes" in renderer.setup(Scene(passes=[]))


def test_unbound_channel_reads_black(renderer):
    # 'leak' leaves a white texture bound on unit 0; 'main' has an unbound
    # channel 0, whose sampler used to default to that unit and sample the leak
    sc = Scene(
        passes=[
            Pass(name="bright", code="void main(){ gl_FragColor = vec4(1.0); }"),
            Pass(
                name="leak",
                channels=[Channel("bright")],
                code="void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
            ),
            Pass(
                name="main",
                channels=[Channel(None)],
                code="void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
            ),
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    f = renderer.render(None).copy()
    assert int(f[..., :3].max()) <= 2


def test_compile_error_leaves_the_running_scene(renderer):
    good = scene([{"name": "main", "code": FLAT}])
    assert renderer.setup(good) == ""
    renderer.set_time(0.0)
    before = renderer.render(None).copy()

    bad = scene(
        [
            {"name": "main", "code": FLAT},
            {"name": "broken", "channels": [{"buffer": "main"}], "code": BROKEN},
        ]
    )
    err = renderer.setup(bad)
    assert err
    assert renderer.error_pass == 1
    assert np.array_equal(renderer.render(None), before)


def test_unallocatable_pass_buffer_fails_loudly(renderer):
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""
    renderer.set_time(0.0)
    before = renderer.render(None).copy()

    idx = renderer.lib.ghee_mp_add_pass(renderer.handle, RAW.encode("utf-8"))
    assert idx == 0
    assert (
        renderer.lib.ghee_mp_set_pass_size(renderer.handle, idx, 1 << 17, 1 << 17)
        is False
    )
    assert "pass buffer failed" in renderer.last_error()
    renderer.lib.ghee_mp_abort(renderer.handle)
    assert np.array_equal(renderer.render(None), before)


def test_render_on_an_empty_scene_reports_instead_of_indexing(renderer):
    renderer.reset()
    renderer.render(None)
    assert "no passes" in renderer.last_error()
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""


def test_downscaled_readback_matches_the_full_frame(renderer):
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""
    renderer.set_time(0.0)
    full = renderer.render(None).copy()
    small = renderer.render(None, out_size=(8, 6))
    assert small.shape == (6, 8, renderer.components)
    for c in range(3):
        assert abs(int(small[..., c].mean()) - int(full[..., c].mean())) <= 2

    buf = np.zeros((6, 8, renderer.components), dtype=np.uint8)
    assert renderer.render(None, out=buf, out_size=(8, 6)) is buf
    assert np.array_equal(buf, small)


def test_render_into_a_caller_buffer(renderer):
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""
    renderer.set_time(0.0)
    full = renderer.render(None).copy()
    buf = np.zeros(
        (renderer.height, renderer.width, renderer.components), dtype=np.uint8
    )
    assert renderer.render(None, out=buf) is buf
    assert np.array_equal(buf, full)


def test_render_rejects_a_buffer_it_cannot_write(renderer):
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""
    with pytest.raises(ValueError):
        renderer.render(None, out=np.zeros((renderer.height, renderer.width, 4)))


def test_draw_renders_with_no_readback(renderer):
    # feedback: each frame adds to the last, so a draw() that really rendered
    # shows up in the next readback
    sc = scene(
        [
            {
                "name": "main",
                "channels": [{"buffer": "main"}],
                "code": (
                    "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) "
                    "+ vec4(0.1, 0.0, 0.0, 1.0); }"
                ),
            }
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    one = renderer.render(None).copy()
    assert renderer.setup(sc) == ""  # fresh buffers, cleared to black
    assert renderer.draw() is None
    two = renderer.render(None).copy()
    assert int(two[..., 0].mean()) > int(one[..., 0].mean())


def test_one_upload_per_array_name(renderer):
    sc = scene(
        [
            {
                "name": "a",
                "arrays": [{"name": "shared", "size": 4}],
                "code": "void main(){ gl_FragColor = vec4(shared_at(0), 0.0, 0.0, 1.0); }",
            },
            {
                "name": "b",
                "channels": [{"buffer": "a"}],
                "arrays": [{"name": "shared", "size": 4}],
                "code": (
                    "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) "
                    "+ vec4(0.0, shared_at(1), 0.0, 0.0); }"
                ),
            },
            {
                "name": "c",
                "channels": [{"buffer": "b"}],
                "arrays": [{"name": "shared", "size": 4}],
                "code": (
                    "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) "
                    "+ vec4(0.0, 0.0, shared_at(2), 0.0); }"
                ),
            },
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    calls = []
    real = renderer.lib
    renderer.lib = _CountingLib(real, calls)
    try:
        f = renderer.render(lambda name, size: np.full(size, 0.5, dtype=np.float32))
    finally:
        renderer.lib = real
    assert calls == [b"shared"]  # one upload, three passes reading it
    for c in range(3):
        assert abs(int(f[..., c].mean()) - 128) <= 2


def test_texture_unit_budget_rejects_a_greedy_pass(renderer, monkeypatch):
    # desktop GL has 16+ units, VC4 has 8: pin the budget to exercise the check
    monkeypatch.setattr(renderer, "max_texture_units", lambda: 2)
    good = scene([{"name": "main", "code": FLAT}])
    assert renderer.setup(good) == ""
    renderer.set_time(0.0)
    before = renderer.render(None).copy()

    greedy = scene(
        [
            {"name": "a", "code": FLAT},
            {"name": "b", "code": FLAT},
            {
                "name": "c",
                "channels": [{"buffer": "a"}, {"buffer": "b"}],
                "arrays": [{"name": "v", "size": 4}],
                "code": "void main(){ gl_FragColor = vec4(v_at(0)); }",
            },
        ]
    )
    err = renderer.setup(greedy)
    assert "texture units" in err
    assert np.array_equal(renderer.render(None), before)


def test_channel_source_pass_is_bound_checked(renderer):
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""
    assert renderer.lib.ghee_mp_set_channel(renderer.handle, 0, 0, 7) is False
    assert renderer.lib.ghee_mp_set_channel(renderer.handle, 0, 0, -1) is True
    assert renderer.lib.ghee_mp_set_channel(renderer.handle, 0, 0, 0) is True


def test_set_time_none_returns_to_the_wall_clock(renderer):
    sc = scene(
        [
            {
                "name": "main",
                "code": "void main(){ gl_FragColor = vec4(mod(time * 37.0, 1.0), 0.0, 0.0, 1.0); }",
            }
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(1.0)
    pinned = renderer.render(None).copy()
    assert np.array_equal(renderer.render(None), pinned)
    renderer.set_time(None)
    a = renderer.render(None).copy()
    time.sleep(0.02)
    assert not np.array_equal(renderer.render(None), a)


def test_set_dt_reaches_every_pass(renderer):
    """DT rides the draw like time, so one store has to land in the uniform of every
    pass. Red carries the first pass's DT through a channel, green the second's."""
    sc = scene(
        [
            {
                "name": "src",
                "code": "void main(){ gl_FragColor = vec4(DT, 0.0, 0.0, 1.0); }",
            },
            {
                "name": "main",
                "code": "void main(){ gl_FragColor = vec4("
                "texture2D(iChannel0, v_texCoord).r, DT, 0.0, 1.0); }",
                "channels": [{"buffer": "src"}],
            },
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)

    def dt_channels():
        return renderer.render(None).copy()[..., :2].astype(int)

    assert (abs(dt_channels() - round(255 / 60)) <= 1).all()  # engine default
    renderer.set_dt(0.5)
    assert (abs(dt_channels() - 128) <= 1).all()
    renderer.set_dt(0.25)
    assert (abs(dt_channels() - 64) <= 1).all()

    renderer.set_dt(0.5)
    same = renderer.render(None).copy()
    assert np.array_equal(renderer.render(None), same)


def test_closing_a_second_renderer_leaves_the_first_alive(renderer):
    sc = scene([{"name": "main", "code": FLAT}])
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    before = renderer.render(None).copy()
    try:
        other = Renderer(16, 16, components=renderer.components)
    except Exception as e:  # one GPU node, one renderer at a time
        pytest.skip(f"no second renderer: {e}")
    assert other.setup(sc) == ""
    other.render(None)
    other.close()
    assert np.array_equal(renderer.render(None), before)


@pytest.mark.skipif(
    not os.path.exists("/dev/video11"), reason="no hardware H264 encoder"
)
def test_encoder_flush_drains_the_tail(renderer):
    assert renderer.setup(scene([{"name": "main", "code": FLAT}])) == ""
    err = ctypes.c_char_p()
    enc = renderer.lib.ghee_encoder_create(
        renderer.handle,
        renderer.width,
        renderer.height,
        30,
        2_000_000,
        ctypes.byref(err),
    )
    if not enc:
        pytest.skip(f"encoder create failed: {err.value}")
    nal = ctypes.POINTER(ctypes.c_ubyte)()
    ln, key = ctypes.c_int(), ctypes.c_int()
    args = (
        renderer.handle,
        enc,
        ctypes.byref(nal),
        ctypes.byref(ln),
        ctypes.byref(key),
    )
    try:
        for _ in range(8):
            renderer.lib.ghee_mp_render_encoded(*args)
        tail = 0
        while renderer.lib.ghee_encoder_flush(*args) == 0:
            tail += 1
        assert renderer.lib.ghee_encoder_flush(*args) == 1  # drained stays drained
    finally:
        renderer.lib.ghee_encoder_destroy(renderer.handle, enc)
