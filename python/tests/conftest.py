"""What every test here shares: the repo paths and the scenes."""

import json
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

RED = {
    "version": 1,
    "params": {"tint": {"type": "color", "value": [1.0, 0.0, 0.0]}},
    "passes": [
        {
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(tint * gain * level_at(0), 1.0); }",
            "arrays": [{"name": "level", "size": 8}],
            "params": {"gain": {"type": "float", "value": 1.0}},
        }
    ],
}

# Each quarter of the frame, left to right, shows one vector param: a, b,
# the rgb of c times its w, and tint.
VECS = {
    "version": 1,
    "params": {
        "a": {"type": "vec2", "value": [0.25, 0.5]},
        "tint": {"type": "color", "value": [0.5, 0.25, 1.0]},
    },
    "passes": [
        {
            "name": "main",
            "code": "void main(){ float x = v_texCoord.x;"
            " gl_FragColor = x < 0.25 ? vec4(a, 0.0, 1.0) : x < 0.5 ? vec4(b, 1.0)"
            " : x < 0.75 ? vec4(c.xyz * c.w, 1.0) : vec4(tint, 1.0); }",
            "params": {
                "b": {"type": "vec3", "value": [0.75, 0.5, 0.25]},
                "c": {"type": "vec4", "value": [1.0, 0.5, 0.25, 0.5]},
            },
        }
    ],
}


def quarters(h):
    """The rgb at the centre of each quarter of an 8x1 frame, 0..1."""
    px = h.pixels()
    n = h.components
    return [[v / 255 for v in px[x * n : x * n + 3]] for x in (1, 3, 5, 7)]


def near(got, want):
    return all(
        abs(g - w) <= 2 / 255 for gs, ws in zip(got, want) for g, w in zip(gs, ws)
    )


BROKEN = {
    "version": 1,
    "passes": [
        {"name": "main", "code": "void main(){ gl_FragColor = no_such_symbol; }"}
    ],
}


@pytest.fixture(autouse=True, scope="session")
def cache_home(tmp_path_factory):
    """A draw's glyph atlas lands in a cache of the session's own, never in the
    user's."""
    saved = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(tmp_path_factory.mktemp("cache"))
    yield
    if saved is None:
        del os.environ["XDG_CACHE_HOME"]
    else:
        os.environ["XDG_CACHE_HOME"] = saved


def scene_file(tmp_path, doc, name="s.scene.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


@pytest.fixture
def red():
    return dict(RED)
