# ghee for Python

`pip install ghee-render` installs the [ghee](https://github.com/holofermes/ghee)
renderer as a Python library, `ghee.open`, and the Python sugar for writing
a [procedural scene](https://github.com/holofermes/ghee/blob/main/docs/procedural.md).
The package installs no `ghee` command. That command comes from the deb on
the [releases page](https://github.com/holofermes/ghee/releases), or from a
[source build](https://github.com/holofermes/ghee/blob/main/docs/building.md).

Build the wheel yourself with maturin:

```
maturin build --release --compatibility linux
```

## Drawing on a panel

This loop draws the time, the load of the machine as a share of its cores
with a bar beside it, and a blinking HOT past 90% on a 128x32 panel, set up as
[Hardware displays](https://github.com/holofermes/ghee/blob/main/docs/displays.md#hardware-displays)
shows:

```python
import os
import time

import ghee

screen = ghee.open(display="kms")
while True:
    load = os.getloadavg()[0] / os.cpu_count()
    with screen.draw() as d:
        d.text(time.strftime("%H:%M"), x=0, y=0, font_size=16)
        d.text(f"load {load:.0%}", x=0, y=20, font_size=8, font_family="pixel")
        d.rect(
            x=64,
            y=20,
            width=64,
            height=8,
            color="transparent",
            border={"width": 1, "color": "white"},
        )
        d.rect(x=66, y=22, width=60 * min(load, 1), height=4)
        if load > 0.9:
            d.text("HOT", x=100, y=0, font_size=12, blink={"rate": 2})
```

## A Python scene

This procedural scene draws the time on each tick, and `python clock.py`
shows it on the connected display:

```python
import time

import ghee

ghee.args()  # the probe, this scene takes no arguments


@ghee.tick
def clock(tick):
    with ghee.draw() as d:
        d.text(time.strftime("%H:%M:%S"))


ghee.run()
```

## Docs

- [Python](https://github.com/holofermes/ghee/blob/main/docs/python.md):
  `ghee.open`, drawing, the shim and direct mode
- [Scenes](https://github.com/holofermes/ghee/blob/main/docs/scenes.md)
- [Layers](https://github.com/holofermes/ghee/blob/main/docs/layers.md)
- [Filters](https://github.com/holofermes/ghee/blob/main/docs/filters.md)
- [Outputs and displays](https://github.com/holofermes/ghee/blob/main/docs/displays.md)
- [The console](https://github.com/holofermes/ghee/blob/main/docs/console.md),
  and its [HTTP API](https://github.com/holofermes/ghee/blob/main/docs/http.md)
- [Procedural scenes](https://github.com/holofermes/ghee/blob/main/docs/procedural.md),
  the wire between ghee and a scene
- [From Rust](https://github.com/holofermes/ghee/blob/main/docs/rust.md)
- [Build](https://github.com/holofermes/ghee/blob/main/docs/building.md)
- [Troubleshooting](https://github.com/holofermes/ghee/blob/main/docs/troubleshooting.md)

## License

MIT. The text shader ports Eric Lengyel's
[Slug](https://github.com/EricLengyel/Slug), also MIT. ghee builds FreeType
2.14.3 in under the FreeType License.
[NOTICE](https://github.com/holofermes/ghee/blob/main/NOTICE) carries both
notices, and
[THIRD-PARTY-NOTICES](https://github.com/holofermes/ghee/blob/main/THIRD-PARTY-NOTICES)
the licenses of the Rust crates that ghee compiles in.
