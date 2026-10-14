# ghee

ghee is a headless GPU renderer for small screens: Shadertoy-style GLSL, text
and images, from the command line or Python, out to panels, HDMI, files or a
stream.

![a four-pass particle scene, procedural text, and the bundled dither filter, looping](https://raw.githubusercontent.com/holofermes/ghee/main/media/hero.webp)

## What it does

- [Passes](docs/scenes.md) in Shadertoy-style GLSL, wired by channels.
- [Layers](docs/layers.md) of text, images and shapes, with effects.
- [Filters](docs/filters.md), such as dither and FRC for 1-bit and 16-level panels.
- [Outputs](docs/displays.md): image files, raw frames, H264, HDMI and SPI/I2C panels.
- [The web console](docs/console.md), a live preview and editor of the scene.
- [Python](docs/python.md) scenes, and drawing on a screen from a loop.
- [Rust](docs/rust.md), the renderer as a library.

## Install

On a Pi, take the deb from the [releases
page](https://github.com/holofermes/ghee/releases). It installs the `ghee`
command and the Python package for the system Python. The deb is the only
install for the 32-bit Pis:

```
sudo apt install ./ghee_*.deb
```

In a uv project or a venv, on any 64-bit OS, add the Python package:

```
uv add ghee-render                # or, in a venv: pip install ghee-render
```

The package installs no `ghee` command. That command comes from the deb, or
from a [source build](docs/building.md).

## Quick start

With `--demo`, ghee takes the scene from its bundled
[examples](ghee/examples). Render one of the JSON scenes to an image sequence
of sixty frames:

```
ghee render --demo mix -o frame.####.png --frames 60
```

Serve the web [console](docs/console.md), to preview and author every demo
scene:

```
ghee serve --demo
```

This loop draws the time, the load of the machine as a share of its cores
with a bar beside it, and a blinking HOT past 90% on a 128x32 panel, set up as
[Hardware displays](docs/displays.md#hardware-displays) shows:

```python
import os
import time

import ghee

screen = ghee.open(display="kms")
while True:
    load = os.getloadavg()[0] / os.cpu_count()
    with screen.draw() as d:
        d.text(time.strftime("%H:%M"), x=0, y=0, font_size=16)
        d.text(
            f"load {load:.0%}",
            x=0,
            y=20,
            font_size=8,
            font_family="Pixel Operator 8",
        )
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

This procedural scene draws the time on each tick, and `uv run clock.py`
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

- [Scenes](docs/scenes.md)
- [Layers](docs/layers.md)
- [Filters](docs/filters.md)
- [Outputs and displays](docs/displays.md)
- [The console](docs/console.md), and its [HTTP API](docs/http.md)
- [Python](docs/python.md)
- [Procedural scenes](docs/procedural.md), the wire between ghee and a scene
- [From Rust](docs/rust.md)
- [Build](docs/building.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT. The text shader ports Eric Lengyel's Slug, also MIT. ghee builds
FreeType 2.14.3 from `vendor/freetype` under the FreeType License.
[NOTICE](NOTICE) carries both notices, and
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) the licenses of the Rust crates
that ghee compiles in.

![](https://fabiopiparo.com/pixel.gif?site=ghee)
