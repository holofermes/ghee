#!/usr/bin/env python3
import json
import math
import os
import struct
import sys

FRAGMENT = """
void main() {
  float x = uv.x * 16.0;
  float bar = step(1.0 - uv.y, levels_at(int(x))) * step(0.2, fract(x));
  gl_FragColor = vec4(vec3(bar), 1.0);
}
"""

scene = {
    "passes": [
        {
            "name": "main",
            "fragment": FRAGMENT,
            "arrays": [{"name": "levels", "size": 16}],
        }
    ]
}
lockstep = os.environ.get("GHEE_LOCKSTEP") == "1"
if lockstep:
    scene["lockstep"] = True

out = sys.stdout.buffer
out.write(json.dumps(scene).encode() + b"\n")
out.flush()

for line in sys.stdin:
    tick = json.loads(line)
    levels = [0.5 + 0.4 * math.sin(2 * tick["time"] + i / 2) for i in range(16)]
    name = b"levels"
    # kind 0, the name's length, the name, the count, 16 little-endian floats
    out.write(struct.pack("<BB", 0, len(name)) + name)
    out.write(struct.pack("<I16f", 16, *levels))
    if lockstep:
        out.write(b"\x05")  # kind 5: the answer to this tick is complete
    out.flush()
