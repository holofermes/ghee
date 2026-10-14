# ghee for Python

`ghee-render` installs [ghee](https://github.com/holofermes/ghee), a GPU
renderer for small screens: the Python package `ghee`, and the `ghee`
command. It runs on 64-bit Linux, x86_64 or Arm, with Python 3.10 or newer:

```
pip install ghee-render
```

On a Pi with a 32-bit OS, take the deb from the [releases
page](https://github.com/holofermes/ghee/releases).

This program opens a 128x64 screen in memory, draws `hello` on it, and saves
the frame as a PNG:

```python
import ghee

screen = ghee.open(width=128, height=64)
with screen.draw() as d:
    d.text("hello", color="orange", glow={})
    d.line((8, 56), (120, 56), color="orange")
screen.save("hello.png")
```

![hello in orange with an orange glow, and an orange line under it](https://raw.githubusercontent.com/holofermes/ghee/main/media/python/hello.png)

To draw on the connected display instead, use `ghee.open(display="kms")`:
HDMI, or a panel on SPI or I2C, set up with its device tree overlay.

Either way ghee draws on the GPU.

Use the `ghee` command to render a scene to an image:

```
ghee render --demo bars -o bars.png
```

Or serve the web console, to author and edit the demo scenes:

```
ghee serve --demo
```

## Docs

- [Python](https://github.com/holofermes/ghee/blob/main/docs/python.md):
  drawing, opening a scene, and Python scenes
- [Scenes](https://github.com/holofermes/ghee/blob/main/docs/scenes.md)
- [Layers](https://github.com/holofermes/ghee/blob/main/docs/layers.md)
- [Outputs and displays](https://github.com/holofermes/ghee/blob/main/docs/displays.md)
- [The console](https://github.com/holofermes/ghee/blob/main/docs/console.md)
- [Every page](https://github.com/holofermes/ghee#docs)

## License

MIT. The text shader ports Eric Lengyel's
[Slug](https://github.com/EricLengyel/Slug), also MIT. ghee builds FreeType
2.14.3 in under the FreeType License.
[NOTICE](https://github.com/holofermes/ghee/blob/main/NOTICE) carries both
notices, and
[THIRD-PARTY-NOTICES](https://github.com/holofermes/ghee/blob/main/THIRD-PARTY-NOTICES)
the licenses of the Rust crates that ghee compiles in.
