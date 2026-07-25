"""Render self-consistency tests. These compare the engine's output against itself
(same inputs -> same pixels, different inputs -> different pixels), so they hold on any
GPU without a committed golden. They skip when no GL device is available."""

import numpy as np

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
