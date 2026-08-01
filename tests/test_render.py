"""Render self-consistency tests. These compare the engine's output against itself
(same inputs -> same pixels, different inputs -> different pixels), so they hold on any
GPU without a committed golden. They skip when no GL device is available."""

import os
import subprocess
import sys

import numpy as np
import pytest

from .helpers import const, scene


def _render(renderer, sc, t=0.0, provider=None, frames=1):
    renderer.setup(sc)
    renderer.set_time(t)
    out = None
    for _ in range(frames):
        out = renderer.render(provider).copy()
    return out


def test_solid_color(renderer):
    f = _render(
        renderer,
        scene(
            [
                {
                    "name": "main",
                    "code": "void main(){ gl_FragColor=vec4(0.5,0.25,0.75,1.0);}",
                }
            ]
        ),
    )
    assert abs(int(f[..., 0].mean()) - 128) <= 1
    assert abs(int(f[..., 1].mean()) - 64) <= 1
    assert abs(int(f[..., 2].mean()) - 191) <= 1


def test_determinism(renderer):
    sc = scene(
        [
            {
                "name": "main",
                "code": "void main(){ vec2 uv=gl_FragCoord.xy/resolution; gl_FragColor=vec4(uv, abs(sin(time)), 1.0);}",
            }
        ]
    )
    assert np.array_equal(_render(renderer, sc, t=1.0), _render(renderer, sc, t=1.0))


def test_time_uniform_changes_output(renderer):
    sc = scene(
        [
            {
                "name": "main",
                "code": "void main(){ gl_FragColor=vec4(abs(sin(time)),0.2,0.4,1.0);}",
            }
        ]
    )
    assert not np.array_equal(
        _render(renderer, sc, t=0.0), _render(renderer, sc, t=1.5)
    )


def test_named_array(renderer):
    sc = scene(
        [
            {
                "name": "main",
                "arrays": [{"name": "vals", "size": 4}],
                "code": "void main(){ gl_FragColor=vec4(vals_at(0), vals_at(1), vals_at(2), 1.0);}",
            }
        ]
    )
    red = const([1.0, 0.0, 0.0, 0.0])
    green = const([0.0, 1.0, 0.0, 0.0])
    a = _render(renderer, sc, provider=red)
    b = _render(renderer, sc, provider=green)
    a2 = _render(renderer, sc, provider=red)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, a2)


def test_param_live_update(renderer):
    sc = scene(
        [
            {
                "name": "main",
                "params": {"k": {"type": "float", "value": 0.2}},
                "code": "void main(){ gl_FragColor=vec4(k,k,k,1.0);}",
            }
        ]
    )
    renderer.setup(sc)
    renderer.set_time(0.0)
    a = renderer.render(None).copy()
    renderer.set_param("main", "k", "float", 0.8)
    b = renderer.render(None).copy()
    renderer.set_param("main", "k", "float", 0.2)
    c = renderer.render(None).copy()
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)
    assert abs(int(b[..., 0].mean()) - 204) <= 2


def test_color_param_renders_and_updates(renderer):
    from ghee import Param, Pass, Scene

    sc = Scene(
        passes=[
            Pass(
                name="main",
                code="void main(){ gl_FragColor = vec4(tint, 1.0); }",
                params={"tint": Param("tint", "color", [1.0, 0.0, 0.0])},
            )
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    red = renderer.render(None).copy()
    # RGBA readback: red is high in channel 0, low in channel 2 (asymmetric)
    assert int(red[..., 0].mean()) > 200
    assert int(red[..., 2].mean()) < 40
    renderer.set_param("main", "tint", "color", [0.0, 0.0, 1.0])
    blue = renderer.render(None).copy()
    assert int(blue[..., 2].mean()) > 200
    assert int(blue[..., 0].mean()) < 40


def test_vec2_param_renders_and_updates(renderer):
    from ghee import Param, Pass, Scene

    # a horizontal red gradient shifted by offset.x; offset.y is unused
    sc = Scene(
        passes=[
            Pass(
                name="main",
                code=(
                    "void main(){ vec2 uv = gl_FragCoord.xy/resolution; "
                    "gl_FragColor = vec4(clamp(uv.x + offset.x, 0.0, 1.0), "
                    "0.0, 0.0, 1.0); }"
                ),
                params={"offset": Param("offset", "vec2", [0.0, 0.0])},
            )
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    base = renderer.render(None).copy()
    h, w = base.shape[:2]
    # unshifted: dark at the left edge, bright at the right (asymmetric across x)
    left = int(base[:, w // 8, 0].mean())
    right = int(base[:, w - w // 8, 0].mean())
    assert right - left > 100
    # +0.5 x offset lifts the whole gradient: brighter everywhere
    renderer.set_param("main", "offset", "vec2", [0.5, 0.0])
    shifted = renderer.render(None).copy()
    assert int(shifted[..., 0].mean()) - int(base[..., 0].mean()) > 80
    # a live update back to zero restores the original frame
    renderer.set_param("main", "offset", "vec2", [0.0, 0.0])
    assert np.array_equal(renderer.render(None).copy(), base)


def test_scene_param_reaches_every_pass(renderer):
    from ghee import Channel, Param, Pass, Scene

    # pass a fills k into red; pass b adds k into green on top of a's buffer. A
    # scene-scoped k must reach BOTH passes, so both channels reflect it.
    sc = Scene(
        params={"k": Param("k", "float", 0.2)},
        passes=[
            Pass(
                name="a", code="void main(){ gl_FragColor = vec4(k, 0.0, 0.0, 1.0); }"
            ),
            Pass(
                name="b",
                channels=[Channel("a")],
                code=(
                    "void main(){ vec2 uv = gl_FragCoord.xy/resolution; "
                    "gl_FragColor = texture2D(iChannel0, uv) + vec4(0.0, k, 0.0, 0.0); }"
                ),
            ),
        ],
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    f = renderer.render(None).copy()
    # k=0.2 -> ~51 in R (pass a) AND ~51 in G (pass b); B stays ~0 (asymmetric)
    assert abs(int(f[..., 0].mean()) - 51) <= 3
    assert abs(int(f[..., 1].mean()) - 51) <= 3
    assert int(f[..., 2].mean()) <= 3
    # bump k on every pass at once; both R and G must move together to ~204
    renderer.set_scene_param("k", "float", 0.8)
    g = renderer.render(None).copy()
    assert abs(int(g[..., 0].mean()) - 204) <= 3  # pass a picked up 0.8
    assert abs(int(g[..., 1].mean()) - 204) <= 3  # pass b picked up 0.8 too
    # a half-updated composite would read 204/51 or 51/204; both moving rules it out


def test_multipass_forward_channel(renderer):
    sc = scene(
        [
            {"name": "a", "code": "void main(){ gl_FragColor=vec4(0.5,0.0,0.0,1.0);}"},
            {
                "name": "b",
                "channels": [{"buffer": "a"}],
                "code": "void main(){ vec2 uv=gl_FragCoord.xy/resolution; gl_FragColor=texture2D(iChannel0,uv)+vec4(0.0,0.5,0.0,0.0);}",
            },
        ]
    )
    f = _render(renderer, sc)
    assert abs(int(f[..., 0].mean()) - 128) <= 2
    assert abs(int(f[..., 1].mean()) - 128) <= 2
    assert int(f[..., 2].mean()) <= 2


def test_array_provider_dedup_by_name(renderer):
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
                "code": "void main(){ vec2 uv=gl_FragCoord.xy/resolution; gl_FragColor = texture2D(iChannel0,uv) + vec4(0.0, shared_at(1), 0.0, 0.0); }",
            },
        ]
    )
    renderer.setup(sc)
    calls = {}

    def prov(name, size):
        calls[name] = calls.get(name, 0) + 1
        return np.full(size, 0.5, dtype=np.float32)

    renderer.render(prov)
    assert calls == {"shared": 1}  # one call despite two passes declaring it


def test_conflicting_array_sizes_rejected(renderer):
    good = scene(
        [
            {
                "name": "main",
                "arrays": [{"name": "v", "size": 4}],
                "code": "void main(){ gl_FragColor = vec4(v_at(0), 0.0, 0.0, 1.0); }",
            }
        ]
    )
    assert renderer.setup(good) == ""
    bad = scene(
        [
            {
                "name": "a",
                "arrays": [{"name": "amps", "size": 16}],
                "code": "void main(){ gl_FragColor = vec4(amps_at(0), 0.0, 0.0, 1.0); }",
            },
            {
                "name": "b",
                "channels": [{"buffer": "a"}],
                "arrays": [{"name": "amps", "size": 8}],
                "code": "void main(){ vec2 uv=gl_FragCoord.xy/resolution; gl_FragColor = texture2D(iChannel0,uv); }",
            },
        ]
    )
    err = renderer.setup(bad)
    assert "amps" in err and "conflicting sizes" in err
    # engine untouched: the previous good scene still renders
    f = renderer.render(lambda n, s: np.full(s, 0.5, dtype=np.float32))
    assert abs(int(f[..., 0].mean()) - 128) <= 2


def test_multipass_feedback_accumulates(renderer):
    sc = scene(
        [
            {
                "name": "main",
                "channels": [{"buffer": "main"}],
                "code": "void main(){ vec2 uv=gl_FragCoord.xy/resolution; vec4 prev=texture2D(iChannel0,uv); gl_FragColor=prev+vec4(0.1,0.0,0.0,1.0);}",
            }
        ]
    )
    renderer.setup(sc)
    renderer.set_time(0.0)
    f1 = renderer.render(None).copy()
    f2 = renderer.render(None).copy()
    f3 = renderer.render(None).copy()
    assert int(f2[..., 0].mean()) > int(f1[..., 0].mean())
    assert int(f3[..., 0].mean()) > int(f2[..., 0].mean())


def test_2d_array_input(renderer):
    import numpy as np

    from ghee import ArrayInput, Pass, Scene

    scene = Scene(
        passes=[
            Pass(
                name="main",
                code="void main(){ gl_FragColor = vec4(vec3(mask_sample2(v_texCoord)), 1.0); }",
                arrays=[ArrayInput("mask", (8, 4))],
            )
        ]
    )
    assert renderer.setup(scene) == ""
    m = np.linspace(0.0, 1.0, 32, dtype=np.float32)  # 8x4 ramp
    frame = renderer.render(lambda name, size: m)
    lum = frame[:, :, 0].astype(int)
    assert lum.max() - lum.min() > 180  # the full ramp came through the 2D sampler
    h, w = frame.shape[:2]
    row = lum[h // 2]
    assert abs(int(row[w - 3]) - int(row[2])) > 30  # varies along x within a row


def _mask_roundtrip(renderer, w, h):
    """Feed a w x h 2D input holding a ramp in memory order, then read every texel
    back at the centre of the block it covers. Returns (read, fed) as 0-255 rows, so
    a row picked up at the wrong offset lands somewhere else along the ramp."""
    from ghee import ArrayInput, Pass, Scene

    sc = Scene(
        passes=[
            Pass(
                name="main",
                code=(
                    f"const vec2 dim = vec2({w}.0, {h}.0);\n"
                    "void main(){ vec2 uv = gl_FragCoord.xy/resolution;\n"
                    "  gl_FragColor = vec4(vec3(mask_sample2("
                    "(floor(uv*dim)+0.5)/dim)), 1.0); }"
                ),
                arrays=[ArrayInput("mask", (w, h))],
            )
        ]
    )
    assert renderer.setup(sc) == ""
    m = np.linspace(0.0, 1.0, w * h, dtype=np.float32)
    frame = renderer.render(lambda name, size: m)
    fh, fw = frame.shape[:2]
    read = np.array(
        [
            [
                frame[int((r + 0.5) * fh / h), int((c + 0.5) * fw / w), 0]
                for c in range(w)
            ]
            for r in range(h)
        ],
        dtype=int,
    )
    return read, np.round(m.reshape(h, w) * 255.0).astype(int)


def test_2d_array_input_odd_row_length(renderer):
    # 9 is the point: on the 8-bit data-texture path a row is as many bytes as the
    # input is wide, so a width that is not a multiple of 4 is what catches the
    # engine reading rows at a padded stride.
    read, fed = _mask_roundtrip(renderer, 9, 5)
    assert abs(read - fed).max() <= 3


def test_2d_array_input_odd_row_length_8bit():
    """The same round-trip on the 8-bit data-texture path, which is the one a GL
    without OES_texture_float takes. Mesa can be told to hide the extension, so a
    desktop covers that path too; skipped on a GL where the knob does nothing."""
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "--no-header",
            f"{__file__}::test_2d_array_input_odd_row_length",
        ],
        env=dict(
            os.environ,
            MESA_EXTENSION_OVERRIDE="-GL_OES_texture_float -GL_OES_texture_float_linear",
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = p.stdout + p.stderr
    if "8-bit LUMINANCE" not in out:
        pytest.skip("this GL keeps float data textures whatever the override says")
    assert p.returncode == 0, out


def test_output_pass_size_rejected_then_ok_with_filter(renderer):
    from ghee import Pass, Scene, apply_filters

    # a sized output (last) pass is rejected: it feeds readback/display.
    sc = Scene(
        passes=[
            Pass(
                name="lo",
                size=(8, 4),
                code="void main(){ gl_FragColor = vec4(1.0); }",
            ),
            Pass(
                name="main",
                size=(16, 8),
                code=(
                    "void main(){ vec2 uv = gl_FragCoord.xy/resolution; "
                    "gl_FragColor = vec4(uv, 0.0, 1.0); }"
                ),
            ),
        ]
    )
    err = renderer.setup(sc)
    assert "main" in err and "size" in err
    # a filter appends a full-size pass after 'main', so 'main' is no longer last
    apply_filters(sc, ["dither"])
    assert renderer.setup(sc) == ""


def test_sized_intermediate_pass_composes(renderer):
    from ghee import Channel, Pass, Scene

    # an 8x4 pass writes a 2D ramp (uv.x -> R, uv.y -> G); a full-size main
    # samples it through v_texCoord, so the small buffer upscales into the frame.
    sc = Scene(
        passes=[
            Pass(
                name="lo",
                size=(8, 4),
                code=(
                    "void main(){ vec2 uv = gl_FragCoord.xy/resolution; "
                    "gl_FragColor = vec4(uv.x, uv.y, 0.0, 1.0); }"
                ),
            ),
            Pass(
                name="main",
                channels=[Channel("lo")],
                code="void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
            ),
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    f = renderer.render(None).copy()
    h, w = f.shape[:2]
    # R rises left->right (uv.x survived the upscale), G differs bottom vs top (uv.y)
    assert int(f[:, w - 3, 0].mean()) - int(f[:, 2, 0].mean()) > 80
    assert abs(int(f[h - 3, :, 1].mean()) - int(f[2, :, 1].mean())) > 80


def test_sized_pass_resolution_is_its_own(renderer):
    from ghee import Channel, Pass, Scene

    # the sized pass writes its own resolution into R,G; main samples it. If the
    # pass saw the renderer size (64x48) instead of its own (20x10), the bytes
    # would read 64/48, not 20/10.
    sc = Scene(
        passes=[
            Pass(
                name="lo",
                size=(20, 10),
                code=(
                    "void main(){ gl_FragColor = "
                    "vec4(resolution.x/255.0, resolution.y/255.0, 0.0, 1.0); }"
                ),
            ),
            Pass(
                name="main",
                channels=[Channel("lo")],
                code="void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
            ),
        ]
    )
    assert renderer.setup(sc) == ""
    renderer.set_time(0.0)
    f = renderer.render(None).copy()
    assert abs(int(f[..., 0].mean()) - 20) <= 2
    assert abs(int(f[..., 1].mean()) - 10) <= 2


def test_sized_feedback_accumulates(renderer):
    from ghee import Channel, Pass, Scene

    # a 4x4 pass reads its own previous frame (feedback) and adds 0.1 each frame;
    # main samples the 4x4 buffer. The ping-pong runs at the pass's own size.
    sc = Scene(
        passes=[
            Pass(
                name="acc",
                size=(4, 4),
                channels=[Channel("acc")],
                code=(
                    "void main(){ vec4 prev = texture2D(iChannel0, v_texCoord); "
                    "gl_FragColor = prev + vec4(0.1, 0.0, 0.0, 1.0); }"
                ),
            ),
            Pass(
                name="main",
                channels=[Channel("acc")],
                code="void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
            ),
        ]
    )
    renderer.setup(sc)
    renderer.set_time(0.0)
    f1 = renderer.render(None).copy()
    f2 = renderer.render(None).copy()
    f3 = renderer.render(None).copy()
    assert int(f2[..., 0].mean()) > int(f1[..., 0].mean())
    assert int(f3[..., 0].mean()) > int(f2[..., 0].mean())
