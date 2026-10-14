# ghee

ghee runs your GLSL headless on the GPU and puts the frames on an SPI/I2C
panel, HDMI, image files, or an H264 stream. Multi-pass, any Raspberry Pi.

![a four-pass particle scene, procedural text, and the bundled dither filter, looping](https://raw.githubusercontent.com/holofermes/ghee/main/media/hero.webp)

## What it does

- [Scenes](docs/scenes.md): GLSL passes that read each other's pictures.
- [Layers](docs/layers.md) of text, images and shapes, with effects.
- [Filters](docs/filters.md), such as dither and FRC for 1-bit and 16-level panels.
- [Outputs](docs/displays.md): image files, raw frames, H264, HDMI and SPI/I2C panels.
- [The console](docs/console.md), a web page to watch and edit a scene at runtime.
- [Python](docs/python.md): draw on a screen from your own program, or write a scene in Python.
- [Procedural scenes](docs/procedural.md): a program in any language that writes and feeds a scene.
- [Rust](docs/rust.md), the renderer as a library.

## Install

On a Pi, take the deb from the [releases
page](https://github.com/holofermes/ghee/releases). It installs the `ghee`
command and the Python package for the system Python. The deb is the only
install for the 32-bit Pis:

```
sudo apt install ./ghee_*.deb
```

On 64-bit Linux, x86_64 or Arm, the Python package `ghee-render` installs the
`ghee` command too. It needs Python 3.10 or newer. Add it to a uv project or a
venv, or install the command alone:

```
uv add ghee-render                # or, in a venv: pip install ghee-render
uv tool install ghee-render       # the ghee command alone, on your PATH
```

## Quick start

With `--demo`, ghee takes the scene from its bundled
[examples](ghee/examples). Render one of the JSON scenes to an image sequence
of sixty frames, here a neon sign at night:

```
ghee render --demo neon -o frame.####.png --frames 60
```

To show it on the connected display, a panel or HDMI, use `--display kms`
([Hardware displays](docs/displays.md#hardware-displays)):

```
ghee render --demo neon --display kms
```

To watch and edit every demo in the [console](docs/console.md), serve it and
open the address that ghee prints:

```
ghee serve --demo
```

From Python, open a screen and draw on it. This draws `hello` on a 128x64
screen in memory and saves it as a PNG ([Python](docs/python.md#drawing)):

```python
import ghee

screen = ghee.open(width=128, height=64)
with screen.draw() as d:
    d.text("hello", color="orange", glow={})
screen.save("hello.png")
```

To write a scene that ghee runs, see [Python scenes](docs/python.md#python-scenes).

## Speed

Each program runs at 30 frames a second, with the CPU clock held at its top
and each run starting cool, and ms counts from the start of a frame's draw to
the end of the bus transfer that puts it on the panel, timed by the kernel. A
Raspberry Pi 5 drives a 256x64 SSD1322 on SPI at 16 MHz, and a Pi Zero 2 W a
128x32 SSD1306 on I2C at 1 MHz. CPU is a share of one core, and the first
frame counts from the program's start. Each figure is the median of three
runs. On the Pi 5, ghee's GPU stays under 0.2% busy.

A status screen: the time and the load, with a bar and a blinking HOT, new on
every frame:

| Program | Pi 5: ms | Pi 5: CPU | Pi 5: first frame | Pi Zero 2 W: ms | Pi Zero 2 W: CPU | Pi Zero 2 W: first frame |
| --- | --- | --- | --- | --- | --- | --- |
| ghee | 5.11 | 1.4% | 0.13 s | 6.46 | 6.5% | 1.33 s |
| Pillow | 5.61 | 3.0% | 0.13 s | 9.80 | 16.3% | 0.75 s |
| pygame | 4.93 | 0.8% | 0.13 s | 5.24 | 1.3% | 0.71 s |

Per-pixel effects: luma.examples' chroma.py, colour patterns computed for
every pixel, and pygame's liquid.py, a picture rippling in blocks, each as
written and ported to a ghee shader:

| Program | Pi 5: ms | Pi 5: CPU | Pi 5: first frame | Pi Zero 2 W: ms | Pi Zero 2 W: CPU | Pi Zero 2 W: first frame |
| --- | --- | --- | --- | --- | --- | --- |
| chroma.py in luma | 24.74 | 60.1% | 0.16 s | 56.95, at 19 fps at most | 99.6% | 1.08 s |
| chroma.py in ghee | 4.98 | 0.9% | 0.15 s | 5.37 | 2.9% | 1.83 s |
| liquid.py in pygame | 5.10 | 1.3% | 0.14 s | 7.04 | 7.6% | 0.90 s |
| liquid.py in ghee | 4.95 | 0.9% | 0.13 s | 5.32 | 2.6% | 0.79 s |

[tools/compare](tools/compare) runs the comparison.

## Docs

- [Scenes](docs/scenes.md)
- [Layers](docs/layers.md)
- [Filters](docs/filters.md)
- [Outputs and displays](docs/displays.md)
- [The console](docs/console.md), and its [HTTP API](docs/http.md)
- [Python](docs/python.md)
- [Procedural scenes](docs/procedural.md), for a program in any language
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
