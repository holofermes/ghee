"""Data arrays are fed as 1xN textures (one sampler fetch, no VC4 size cap). Same
host feed and <name>_at(i) API, plus a free <name>_sample(t). Needs a GL device."""

import numpy as np

from ghee import ArrayInput, Pass, Scene

N = 64  # past the ~48 VC4 uniform-array cap the old path hit


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
