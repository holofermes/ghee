"""What every test here shares: the repo paths and the scenes."""

import json
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

BROKEN = {
    "version": 1,
    "passes": [
        {"name": "main", "code": "void main(){ gl_FragColor = no_such_symbol; }"}
    ],
}


def scene_file(tmp_path, doc, name="s.scene.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


@pytest.fixture
def red():
    return dict(RED)
