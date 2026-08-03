"""The pinned frames, rendered through the Python API.

The provider and the manifest are ghee's own (src/golden.rs,
tools/fixtures/golden/manifest.json). A hash that differs is a render that
changed.
"""

import array
import hashlib
import json

import pytest
from conftest import REPO

import ghee

MANIFEST = REPO / "tools" / "fixtures" / "golden" / "manifest.json"


def load_manifest():
    return json.loads(MANIFEST.read_text())


MAN = load_manifest()


def value_at(i, frame, namehash):
    """The pinned provider of src/golden.rs."""
    return ((i * 2654435761 + frame * 10007 + namehash * 31) % 65536) / 65536.0


def name_hash(name):
    return sum(name.encode("utf-8"))


def arrays_of(path):
    """(name, count) for every array the scene declares, in order."""
    doc = json.loads(path.read_text())
    out = []
    for p in doc["passes"]:
        for a in p.get("arrays", []):
            size = a["size"]
            count = size if isinstance(size, int) else size[0] * size[1]
            out.append((a["name"], count))
    return out


@pytest.mark.parametrize("name", sorted(MAN["scenes"]))
def test_a_bundled_scene_renders_its_pinned_frames(name):
    entry = MAN["scenes"][name]
    path = REPO / entry["path"]
    declared = arrays_of(path)
    with ghee.open(
        str(path),
        width=MAN["width"],
        height=MAN["height"],
        timestep=1.0 / MAN["fps"],
    ) as h:
        assert h.size == (MAN["width"], MAN["height"])
        assert h.components == MAN["components"]
        for frame in range(MAN["frames"]):
            for array_name, count in declared:
                nh = name_hash(array_name)
                h.feed(
                    array_name,
                    array.array("f", [value_at(i, frame, nh) for i in range(count)]),
                )
            h.frame()
            got = hashlib.sha256(h.pixels()).hexdigest()
            assert got == entry["shas"][frame], f"{name}: frame {frame}"
