"""Golden reference frames from the Python pipeline. The Rust `ghee golden`
subcommand renders the same manifest and must hash-match every frame.

The array provider is pinned integer math so both languages can reproduce it
exactly: v[i] = ((i*2654435761 + frame*10007 + namehash*31) % 65536) / 65536
with namehash = sum of the name's UTF-8 bytes. Division by 2^16 is exact in f32.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from ghee.engine import Renderer
from ghee.scene import Scene

FPS = 30.0


def name_hash(name):
    return sum(name.encode("utf-8"))


def feed(frame):
    def provider(name, size):
        n = size if isinstance(size, int) else size[0] * size[1]
        i = np.arange(n, dtype=np.uint64)
        v = (
            i * np.uint64(2654435761)
            + np.uint64(frame) * np.uint64(10007)
            + np.uint64(name_hash(name)) * np.uint64(31)
        ) % np.uint64(65536)
        return v.astype(np.float32) / np.float32(65536.0)

    return provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--components", type=int, default=4)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", action="store_true", help="also write raw frames")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    man = {
        "width": w,
        "height": h,
        "components": a.components,
        "frames": a.frames,
        "fps": FPS,
        "scenes": {},
    }
    os.makedirs(a.out, exist_ok=True)
    for path in a.scenes:
        sc = Scene.load(path)
        name = os.path.basename(path).replace(".scene.json", "")
        with Renderer(w, h, components=a.components) as r:
            err = r.setup(sc)
            if err:
                print(f"{name}: setup failed: {err}", file=sys.stderr)
                return 1
            r.set_dt(1.0 / FPS)
            shas = []
            for f in range(a.frames):
                r.set_time(f / FPS)
                rgb = r.render(feed(f))
                shas.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                if a.dump:
                    rgb.tofile(os.path.join(a.out, f"{name}-{f:03d}.raw"))
            man["scenes"][name] = {"path": path, "shas": shas}
        print(f"{name}: {a.frames} frames")
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"wrote {a.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
