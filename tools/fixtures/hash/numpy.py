"""Writes values.json, which src/engine.rs holds ghee_hash and ghee_noise to:
the formulas of src/noise.rs in numpy float32, one GLSL operation at a time,
at a set of points.

usage: uv run --no-project --with numpy==2.3.4 python -I numpy.py

Every operation of the hash is exact but q - floor(q), which only a
comparison with 0.5 reads, so a hash is an integer k of 2^-23ths.
A noise is a float, from smoothstep's weights and GLSL's mix,
x * (1 - a) + y * a.
"""

import json
from pathlib import Path

import numpy as np

f = np.float32
E = f(2.0**-23)
K1, K2, K3 = f(40503 / 65536), f(27147 / 65536), f(9279 / 65536)


def fract(x):
    return f(x - np.floor(x))


def wrap(v):
    return f(v - step(f(1.0), v))


def round_(s):
    a = np.floor(f(s * f(2048.0)))
    b = f(f(s - f(a * f(2.0**-11))) * f(2048.0))
    g = f(f(a * f(f(a * f(2.0)) + f(3.0))) + f(1031.0))
    return wrap(f(b + f(g * E)))


def mix4(s):
    for _ in range(4):
        s = round_(s)
    return s


def step(edge, x):
    return f(x >= edge)


def cell(x):
    q = f(x * f(2048.0))
    X = np.floor(q)
    u = f(X * E)
    return f(f(u - np.floor(u)) + f(step(f(0.5), q - X) * E))


def hash1(x):
    q = f(x * f(2048.0))
    X = np.floor(q)
    u = f(X * E)
    c = f(f(u - np.floor(u)) + f(step(f(0.5), q - X) * E))
    v = f(np.floor(u) * E)
    hi = f(f(v - np.floor(v)) + f(step(f(1.0), c) * E))
    s = round_(wrap(f(c + K1)))
    return mix4(wrap(f(s + hi)))


def hash2(x, y):
    return mix4(wrap(f(round_(wrap(f(cell(x) + K2))) + cell(y))))


def hash3(x, y, z):
    s = wrap(f(round_(wrap(f(cell(x) + K3))) + cell(y)))
    return mix4(wrap(f(round_(s) + cell(z))))


def smooth(t):
    return f(f(t * t) * f(f(3.0) - f(f(2.0) * t)))


def mix(x, y, a):
    return f(f(x * f(f(1.0) - a)) + f(y * a))


def noise2(x, y):
    ix, iy = f(np.floor(x)), f(np.floor(y))
    fx, fy = smooth(f(x - ix)), smooth(f(y - iy))
    h = [hash2(f(ix + dx), f(iy + dy)) for dy in (0, 1) for dx in (0, 1)]
    return mix(mix(h[0], h[1], fx), mix(h[2], h[3], fx), fy)


def noise3(x, y, z):
    i = [f(np.floor(v)) for v in (x, y, z)]
    fx, fy, fz = (smooth(f(v - w)) for v, w in zip((x, y, z), i))
    h = [
        mix(
            hash3(f(i[0] + dx), f(i[1] + dy), i[2]),
            hash3(f(i[0] + dx), f(i[1] + dy), f(i[2] + 1)),
            fz,
        )
        for dy in (0, 1)
        for dx in (0, 1)
    ]
    return mix(mix(h[0], h[1], fx), mix(h[2], h[3], fx), fy)


FLOATS = [0.0, 1.0, -1.0, 0.5, -0.25, 37.375, -4097.5, 4095.99951171875,
          1234.0625, 1000000.0, -30000000.0, 0.0009765625, 65539.0, -7.7,
          -0.0001, 0.000244140625, -0.000244140625, 1e20, -3e15]
PAIRS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, -1.0), (0.5, 0.5),
         (255.5, 13.5), (-20.3125, 7.75), (4096.0, 0.0), (100000.0, -100000.0),
         (0.123, 0.456), (-0.001, 2047.999), (-0.0003, 3e9), (1e20, -5e16)]
TRIPLES = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, 0.5, 9.0), (12.25, -3.0, 0.75),
           (0.3, 0.6, 0.9), (-500.5, 250.25, 4000.0), (-0.0002, 7e12, 3e30)]
NOISE2 = [(0.0, 0.0), (0.5, 0.5), (3.25, -2.75), (100.1, 7.9), (-0.3, -0.7),
          (17.0, 4.5), (1023.875, -64.125)]
NOISE3 = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (3.25, -2.75, 1.1), (-0.3, 9.6, -12.2),
          (40.0, 41.5, 42.25)]


def k(h):
    v = float(h) * 2.0**23
    assert v == int(v) and 0 <= v < 2**23
    return int(v)


def num(p):
    return [float(f(v)) for v in p]


out = {
    "hash1": [[num([x]), k(hash1(f(x)))] for x in FLOATS],
    "hash2": [[num(p), k(hash2(*map(f, p)))] for p in PAIRS],
    "hash3": [[num(p), k(hash3(*map(f, p)))] for p in TRIPLES],
    "noise2": [[num(p), float(noise2(*map(f, p)))] for p in NOISE2],
    "noise3": [[num(p), float(noise3(*map(f, p)))] for p in NOISE3],
}
assert out["noise2"][0][1] == out["hash2"][0][1] / 2.0**23
assert out["hash1"][1][1] != out["hash2"][1][1]
rows = ",\n".join(
    f' "{name}": [\n' + ",\n".join(f"  {json.dumps(r)}" for r in out[name]) + "\n ]"
    for name in out
)
(Path(__file__).resolve().parent / "values.json").write_text("{\n" + rows + "\n}\n")
