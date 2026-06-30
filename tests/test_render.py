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
