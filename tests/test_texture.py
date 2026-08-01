"""Data arrays are fed as 1xN textures (one sampler fetch, no VC4 size cap). Same
host feed and <name>_at(i) API, plus <name>_sample(t). Needs a GL device."""

import os
import subprocess
import sys

import numpy as np
import pytest

from ghee import ArrayInput, Pass, Scene

N = 64  # past VC4's ~48 uniform-array cap


def _hgrad():
    code = (
        "void main(){ vec2 uv=gl_FragCoord.xy/resolution; "
        "gl_FragColor=vec4(vec3(vals_at(int(uv.x*float(vals_size-1)))),1.0); }"
    )
    return Scene(passes=[Pass(name="main", code=code, arrays=[ArrayInput("vals", N)])])


def test_array_samples_correctly(renderer):
    assert renderer.setup(_hgrad()) == ""
    ramp = np.linspace(0.0, 1.0, N).astype(np.float32)  # vals[i] = i/(N-1)
    f = renderer.render(lambda n, s: ramp).copy().astype(float)
    assert f[:, 0, 0].mean() < 40  # left column ~ vals[0] = 0
    assert f[:, -1, 0].mean() > 215  # right column ~ vals[N-1] = 1
    row = f[f.shape[0] // 2, :, 0]
    assert (np.diff(row) >= -1).all()  # non-decreasing dark->bright across x


def test_sample_interpolates(renderer):
    code = (
        "void main(){ vec2 uv=gl_FragCoord.xy/resolution; "
        "gl_FragColor=vec4(vec3(vals_sample(uv.x)),1.0); }"
    )
    sc = Scene(passes=[Pass(name="main", code=code, arrays=[ArrayInput("vals", N)])])
    assert renderer.setup(sc) == ""
    ramp = np.linspace(0.0, 1.0, N).astype(np.float32)
    f = renderer.render(lambda n, s: ramp).copy().astype(float)
    mid = f[:, f.shape[1] // 2, 0].mean() / 255.0  # sample(0.5) ~ 0.5 for a ramp
    assert 0.4 < mid < 0.6


def _lerp1(vals, t):
    """GL's LINEAR + CLAMP_TO_EDGE on a 1xN row, in numpy."""
    x = np.asarray(t) * len(vals) - 0.5
    i = np.floor(x)
    lo = vals[np.clip(i, 0, len(vals) - 1).astype(int)]
    hi = vals[np.clip(i + 1, 0, len(vals) - 1).astype(int)]
    return lo + (hi - lo) * (x - i)


def test_sample_matches_texel_lerp(renderer):
    """Every column of a rendered _sample(t) ramp against the lerp computed here.
    Holds on a GL that linear-filters float textures and on one that cannot
    (V3D); which reader the renderer emitted is what makes the values agree."""
    vals = np.array([0.10, 0.90, 0.30, 0.70, 0.50], dtype=np.float32)
    code = "void main(){ gl_FragColor=vec4(vec3(vals_sample(v_texCoord.x)),1.0); }"
    sc = Scene(
        passes=[Pass(name="main", code=code, arrays=[ArrayInput("vals", len(vals))])]
    )
    assert renderer.setup(sc) == ""
    f = renderer.render(lambda n, s: vals).copy()
    w = f.shape[1]
    got = f[f.shape[0] // 2, :, 0].astype(float) / 255.0
    want = _lerp1(vals, (np.arange(w) + 0.5) / w)
    assert np.abs(got - want).max() < 4 / 255


def test_sample2_matches_bilinear(renderer):
    """_sample2 at points that land mid-texel in one axis, both, or neither."""
    m = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)  # 2x2, [y][x]
    for uv, want in (
        ((0.5, 0.5), m.mean()),  # centre: all four texels, quarter each
        ((0.25, 0.5), (m[0][0] + m[1][0]) / 2),  # on column 0, between rows
        ((0.5, 0.25), (m[0][0] + m[0][1]) / 2),  # on row 0, between columns
    ):
        code = (
            f"void main(){{ gl_FragColor=vec4(vec3(mask_sample2("
            f"vec2({uv[0]}, {uv[1]}))),1.0); }}"
        )
        sc = Scene(
            passes=[Pass(name="main", code=code, arrays=[ArrayInput("mask", (2, 2))])]
        )
        assert renderer.setup(sc) == ""
        f = renderer.render(lambda n, s: m.ravel()).copy()
        got = f[:, :, 0].astype(float) / 255.0
        assert abs(got.mean() - want) < 4 / 255, uv
        assert np.ptp(got) == 0  # one value, the whole frame


def test_interpolation_without_gpu_float_filter():
    """The same two tests on a GL with float textures it cannot linear-filter,
    which is what V3D (Pi 5) reports. Mesa can be told to hide the extension;
    skipped on a GL where the knob does nothing."""
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "--no-header",
            f"{__file__}::test_sample_matches_texel_lerp",
            f"{__file__}::test_sample2_matches_bilinear",
        ],
        env=dict(os.environ, MESA_EXTENSION_OVERRIDE="-GL_OES_texture_float_linear"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = p.stdout + p.stderr
    if "float (nearest)" not in out:
        pytest.skip("this GL keeps the float filter whatever the override says")
    assert p.returncode == 0, out


ARRAYS = [ArrayInput("vals", N), ArrayInput("mask", (4, 4))]
AT = "float vals_at(int i0){ return texture2D(vals_tex, "


def test_a_filtering_gpu_gets_the_one_fetch_readers():
    g = Pass(name="main", code="void main(){}", arrays=ARRAYS).glsl(hw_linear=True)
    assert (
        "float vals_sample(float t){ return texture2D(vals_tex, vec2(t, 0.5)).r; }" in g
    )
    assert "float mask_sample2(vec2 uv){ return texture2D(mask_tex, uv).r; }" in g
    assert AT in g
    assert "mix(" not in g


def test_a_gpu_without_the_filter_gets_the_shader_interpolation():
    g = Pass(name="main", code="void main(){}", arrays=ARRAYS).glsl(hw_linear=False)
    assert "float x = t * float(vals_size) - 0.5;" in g
    assert "return mix(a, b, x - i);" in g
    assert "vec2 dim = vec2(4.0, 4.0);" in g
    assert "return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);" in g
    assert AT in g  # _at is one fetch either way


def test_the_reader_without_a_gpu_to_ask_is_the_portable_one():
    p = Pass(name="main", code="void main(){}", arrays=ARRAYS)
    assert p.glsl() == p.glsl(hw_linear=False)


def test_setup_emits_the_readers_this_gpu_wants(renderer, monkeypatch):
    """The renderer answers for itself, no caller knob."""
    seen = []
    real = Pass.glsl

    def spy(self, scene_params=None, common=None, hw_linear=False):
        seen.append(hw_linear)
        return real(self, scene_params, common, hw_linear)

    monkeypatch.setattr(Pass, "glsl", spy)
    assert renderer.setup(_hgrad()) == ""
    assert seen == [renderer.data_texture_linear()]
