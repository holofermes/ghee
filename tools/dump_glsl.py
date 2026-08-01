"""Assembled GLSL per pass, both data-texture-filter variants: byte-exact
fixtures for the scene port. Usage: dump_glsl.py OUTDIR SCENE.json..."""

import os
import sys

from ghee.scene import Scene

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for path in sys.argv[2:]:
    sc = Scene.load(path)
    name = os.path.basename(path).replace(".scene.json", "")
    for linear in (False, True):
        tag = "lin" if linear else "near"
        for i, p in enumerate(sc.passes):
            src = p.glsl(sc.params, sc.common, linear)
            with open(f"{out}/{name}-p{i}-{tag}.glsl", "w") as fh:
                fh.write(src)
    print(name, len(sc.passes))
