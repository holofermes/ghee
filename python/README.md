# ghee for Python

`pip install ghee-render` installs the [ghee](https://github.com/holofermes/ghee)
renderer as a Python library, `ghee.open`, and the Python sugar for writing
a [procedural scene](https://github.com/holofermes/ghee/blob/main/docs/procedural.md).
The package installs no `ghee` command. That command comes from the deb on
the [releases page](https://github.com/holofermes/ghee/releases), or from a
source build.

Build the wheel yourself with maturin:

```
maturin build --release --compatibility linux
```

## Text on your OLED

This script shows a clock on a 128x32 SSD1306. The text `clock` names no
font, so it draws in ghee's built-in DejaVu Sans. Its layer names no size, so
the words fit the panel. Each turn of the loop sets the words and draws one
frame:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["ghee-render"]
# ///
import time

import ghee

scene = ghee.Scene(
    texts=[ghee.Text("clock")],
    passes=[ghee.Pass("main", layers=[ghee.Layer("clock")])],
)

with ghee.open(scene, output="ssd1306", fps=4) as screen:
    while True:
        screen.set_param("clock", time.strftime("%H:%M:%S"))
        screen.frame()
```

`output="kms"` takes a panel with a kernel driver in place of ghee's own
`ssd1306` driver.

The same clock with a glow under the digits takes a second layer. The glow
layer reaches 0.2 em past the letters, and its `fragment` turns the distance
from the letters into light. A 1-bit panel leaves every dim pixel dark, so
the `dither` filter draws the glow as dots:

```python
GLOW = """
void main() {
    float far = max(-glyph_round_dist(), 0.0) / glyph_font_size;  // em outside the letters
    float a = 0.4 * exp(-14.0 * far);
    gl_FragColor = vec4(vec3(a), a);
}
"""

scene = ghee.Scene(
    texts=[ghee.Text("clock")],
    passes=[
        ghee.Pass(
            "main",
            layers=[
                ghee.Layer("clock", GLOW, pad=0.2, name="glow"),
                ghee.Layer("clock", name="digits"),
            ],
        )
    ],
    filters=["dither"],
)
```

[Text](https://github.com/holofermes/ghee#text) holds more: `font_size`,
`at` for a pixel position, the recipes for an outline, a shadow, a wave and a
marquee, and what each filter does on a 1-bit panel.

## ghee.open

```python
import ghee

h = ghee.open(
    "scene.json",      # a path, the JSON text of a scene, a dict, or a ghee.Scene
    output="st7789",   # a driver name, "kms", or None for headless
    opts={"width": 240, "height": 240, "backlight": 26},
    width=None, height=None,   # a headless size, a panel states its own
    fps=30,            # pace cap, None runs free
    stats=8090,        # console port, ("0.0.0.0", 8090) binds elsewhere
    stream_max=None,   # h264 stream cap: pixels, or a fraction of the render size
    verbose=False,
    timestep=None,     # pin time to frame*S for a reproducible render
    alpha=False,       # keep the output pass's alpha in pixels()
    base_dir=None,     # where a scene given as no path finds its files
)

h.set_param("gain", 1.5)                 # a scene param
h.set_param("snap", 1.0, pass_="hold")   # the param of one pass
h.set_param("hold.snap", 1.0)            # the same param, dotted
h.set_param("words", "hello")            # the words of a text

h.feed("level", bars)      # bytes, bytearray, memoryview, array, numpy, numbers
tick = h.frame()           # {"frame": 12, "time": 0.4, "dt": 0.0166}
tick = h.skip()            # the same tick, with no draw
h.changed_params()         # {"words": "hi"}: string params that the console set

h.size          # (width, height)
h.components    # 3 for RGB, 4 for RGBA
h.pixels()      # the last frame, row-major (height, width, components)
h.save("frame.####.png")   # the last frame as a .png or a .jpg file
h.close()       # also a context manager
```

## What each call does

`set_param` sets a live uniform, with no recompile. A bare name sets the
scene param, else the param of the first pass that declares it. The name of a
text sets its words, and ghee lays them out again, also with no recompile.

A scene file finds its files beside itself, and a Python scene beside its
script, under ghee and in [direct mode](#direct-mode) alike. A scene given as
a dict, as JSON text or as a `ghee.Scene` finds its files in `base_dir`, the
working directory by default. A file missing there resolves through
`$GHEE_ASSETS`, a colon-separated list of directories.

`feed` holds the data for one declared array, and the next frame consumes it
once. A uint8 buffer reaches the texture as byte / 255, and any other buffer
or iterable of numbers as floats. An array keeps its texture data until a
feed changes it.

`frame` draws and hands the frame to the output. `time` is the wall clock
since the first frame, `dt` is the measured frame period, and `timestep`
pins both to `frame * S`.

`pixels` reads alpha 255 in every pixel, unless `alpha=True` keeps the alpha
that the output pass wrote, with the colour premultiplied by it.

`skip` takes the tick and draws nothing, and the panel keeps its last frame.

`changed_params` returns the string params that changed since its last call,
with their words, and the first call returns every string param. It applies
the console's sets at once. A set from the host itself stays out.

`save` writes the last frame to a `.png` file, or to a `.jpg`, which holds no
alpha. A run of `#` in the name takes the number of the frame's tick, as
`ghee render -o` numbers its files.

`stats` serves the console on `127.0.0.1`, and a `(host, port)` pair binds it
elsewhere. Anyone who reaches the port can change params, so bind beyond
loopback only on a network you trust.
[docs/http.md](https://github.com/holofermes/ghee/blob/main/docs/http.md)
holds the API. This console shows the host's own scene. Param changes apply,
and the console refuses scene switches and recompiles. The video stream
advances with `frame()`: viewers see a new frame only when the host draws one.

## Rules

ghee is thread-affine: open a handle on one thread and use it on that same
thread. A call from another thread raises RuntimeError.

A scene, GLSL or param error raises ValueError, and a driver or system error
raises RuntimeError.

`frame()` releases the GIL, so the host's other threads run while it renders.

## Python scenes

The shim writes a procedural scene. A PEP 723 scene declares `ghee-render`
inline:

    #!/usr/bin/env -S uv run --script
    # /// script
    # dependencies = ["ghee-render"]
    # ///

The shim gives:

- `ghee.Scene / Pass / Param / ArrayInput / Channel / Asset / Text / Layer`
  to build the scene.
- `ghee.Arg` and `ghee.args([...])` to declare the scene's arguments, and to
  answer `-- --help`.
- `@ghee.feed("name")` to supply the values of a named array, each tick. The
  function returns a buffer or an iterable of numbers. A uint8 buffer, such
  as `bytes`, a `bytearray` or a numpy uint8 array, goes out as byte / 255,
  and any other numbers go out as float32. None leaves the array as it
  stands.
- `@ghee.tick` to run a hook on each tick, before the feeds. The hook
  receives a `ghee.Tick` and calls `ghee.set_param(name, value)` and
  `ghee.skip()`.
- `ghee.run(scene)` for the handshake, then the tick loop.

A `ghee.Tick` holds `frame`, the tick's number, `time` and `dt` in seconds,
and `params`, the string params that changed, by name. The first tick's
`params` holds every string param. This scene is the clock above, written as
a procedural scene:

```python
import time

import ghee

ghee.args()  # the probe, this scene takes no arguments


@ghee.tick
def clock(tick):
    ghee.set_param("clock", time.strftime("%H:%M:%S"))


ghee.run(
    ghee.Scene(
        texts=[ghee.Text("clock")],
        passes=[ghee.Pass("main", layers=[ghee.Layer("clock")])],
    )
)
```

`params` on `ghee.Scene` and `ghee.Pass` takes a list of `ghee.Param`, such as
`params=[ghee.Param("gain", "float", 0.5, min=0.0, max=1.0)]`. A dict keyed
by name works too, and a key that differs from its param's name raises
ValueError.

The shim builds the scene and leaves every check to ghee, so a mistake in the
scene reads the same in `ghee.open`, in a direct run and under `ghee render`.

`ghee.set_param` raises ValueError for a param that the scene does not
declare, or for a value that does not fit the param's type. A param set in a
hook lands in the frame of its tick in [direct mode](#direct-mode), and under
ghee with `--timestep`. Under ghee without it, the param lands a frame later
or more.

## Text

`ghee.Text(name, font=None, value="", align="left", fill="mtsdf", atlas=None)`
holds the words in `value` and the font to draw them from. A pass draws a
text in one of two ways. A `ghee.Layer` on the pass paints the words over its
buffer, one quad over each glyph, for words that sit on the screen. The
pass's own shader reads `<name>_cover`, the coverage of the words at any
point, when the words feed its math, such as a warp or a mask.
[Text](https://github.com/holofermes/ghee#text) holds the GLSL of both.

ghee declares the string param of the text's name, and new words from
`set_param` or the console lay out again with no recompile. A scene that
declares its own param under a text's name raises ValueError.

`font` takes a `.ttf` or `.otf` file, which resolves as an asset file does,
the name of a font `ghee.Asset`, or a list that mixes both. None draws from
the built-in DejaVu Sans. With `ghee.Text("words", ["sans", "NotoColorEmoji.ttf"])`,
each character draws from the first font in the list whose cmap holds it:
letters from `sans`, and emoji from a colour emoji font such as Noto Color
Emoji. A character followed by U+FE0F draws from the first font with a colour
glyph for it. Line height and baseline come from the first font.

A text problem leaves the scene running, and `set_param` raises nothing for
it. ghee prints it once on stderr, and the console shows it. A font that fails
to load drops out of the list, and a text with no font left draws every
character as an error box, diagonal stripes of coverage. A character whose
glyph has neither an outline nor a colour bitmap draws an error box in its
place, and the rest of the words draw. Words that do not fit leave the last
words that did. An emoji of a colour font has no coverage, and only a layer
paints it.

## Layers

A pass's code fills its buffer, then each `ghee.Layer` in its `layers` paints
a text over it, in order: one quad over each glyph, coloured by the layer's
own fragment shader. This scene draws `words` in amber at 48 pixels per em,
centred on black:

    AMBER = """
    void main() {
      float c = glyph_cover();
      gl_FragColor = vec4(vec3(1.0, 0.7, 0.2) * c, c);
    }
    """
    BLACK = "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"

    scene = ghee.Scene(
        texts=[ghee.Text("words", value="hello")],
        passes=[ghee.Pass("main", BLACK, layers=[ghee.Layer("words", AMBER, font_size=48)])],
    )

`ghee.Layer(text, fragment=None, vertex=None, font_size=None, at=None,
fade=None, blend="over", pad=None, name=None, emoji=None)`:

- `text` names a `ghee.Text` of the scene.
- `fragment` holds a `void main()` that writes the colour premultiplied by
  alpha. None draws the letters white.
- `vertex` defines `vec2 glyph_place(vec2 em)`. The function takes a point of
  the block, in em from its top left with y down, and returns the pass pixel
  where that point lands. A layer with `vertex` takes neither `font_size` nor
  `at`.
- `font_size` draws the block at that many pixels per em. `at` puts its top
  left at a pixel of the pass, `(x, y)` with y down. Without `at` the block
  sits centred in the pass, and without `font_size` it takes the largest
  size that fits.
- `fade` sets the band in em, at the outer edge of the pad, over which the
  layer fades to 0: a quarter of `pad` for None, and 0 turns the fade off.
- `blend` is `"over"`, `"add"` or `"replace"`.
- `pad` is how far past the letters the layer reaches, in em, for a glow or
  an outline: 0 for None.
- `name` names the layer in errors and in the console: ASCII letters, digits,
  `_` and `-`, unique among the layers of its pass. None goes by the text's
  name.
- `emoji` sets whether the layer paints the text's colour glyphs. None leaves
  them to the last layer of the text in its pass, so a shadow drawn under the
  fill paints no copy.

A pass of layers alone, `ghee.Pass("words", layers=[...])`, starts each frame
transparent, for a later pass to read through a channel.

`fill` on `ghee.Text` sets where a layer of the text takes its coverage from.
`"mtsdf"`, the default, reads an atlas of the glyphs' distance fields: it is
fast, and it gives `glyph_dist`, `glyph_round_dist` and `glyph_fade`.
`"slug"` computes the coverage from the curves, with exact edges at any size.
`atlas={"font_size": 48}` sets the texels per em of the atlas, whose field
reaches as far as the largest `pad` of the text's layers. New words of an
MTSDF text wait for the glyphs that its atlas lacks, and the last words stay
on screen meanwhile. With `timestep` set, `frame()` waits for them.

[Layers](https://github.com/holofermes/ghee#layers) holds the GLSL that
`fragment` and `vertex` read and how they run, and
[Fills](https://github.com/holofermes/ghee#fills) what each fill costs.
`ghee render --demo karaoke` sings a line in layers.

## Coverage in a shader

Every pass except a filter pass reads the words of a text with
`float <name>_cover(vec2 p, float font_size)`: `p` in em from the top left of
the block, y down like `v_texCoord`, and `font_size` the pixels per em there,
which sets the antialiasing. `<name>_size` is the block in em. With `CODE`
holding a shader such as the one in
[Coverage in a shader](https://github.com/holofermes/ghee#coverage-in-a-shader):

    scene = ghee.Scene(
        texts=[ghee.Text("words", value="hello", align="center")],
        passes=[ghee.Pass("main", CODE)],
    )

`ghee render --demo lettering -o lettering.png` shows the words at every
size, and `ghee render --demo lens` bends them under a lens.

## Direct mode

The same scene file runs two ways. When ghee starts it, as
`ghee render spectrum.py` or from a scenes directory, `run()` feeds ghee over
the
[wire protocol](https://github.com/holofermes/ghee/blob/main/docs/procedural.md).
When you run the file itself, `run()` opens the embedded renderer and renders
directly:

    ./spectrum.py --display --frames 300
    ./spectrum.py -o frame.####.png --size 320x240 --frames 60

ghee sets `GHEE_WIRE=1` in every scene process it starts, and `run()` tells
the two apart by it.

The scene takes its own arguments first, then these flags of `ghee render`:

- `-o PATTERN`: image files, as `frame.####.png`, or `-` for raw RGBA frames
  on stdout. Files and the pipe take every frame, a skipped one too.
- `--display [NAME]`: a driver name, or bare or `kms` for the connected
  display. It is the default with no `-o` and no `--serve`.
- `--display-opts K=V,...`: the options of the display driver.
- `--serve`, `--port PORT`, `--host HOST`: the console.
- `--size WxH`: the render size. A display or a panel states its own.
- `--fps N`, `--stats` and `--timestep S`.
- `--frames N`: the frame count. It defaults to 1 for image files or the
  pipe, and to 0 for a display or the console, which runs until SIGINT or
  SIGTERM.

The feeds, the hooks and `ghee.skip()` work the same. The tick keeps its wire
shape: `params` holds the string params that the console changed, and the
first tick holds every string param. If one of your own arguments collides
with these names, argparse says so.

## What the wire carries

The wire suits small per-frame data. Megabyte-scale data costs a
serialization and two copies for each frame, and on a small Pi that shows up
as a low frame rate. A data-heavy host takes `ghee.open` instead: the
renderer in the same process, with no pipe.

## License

MIT. The text shader ports Eric Lengyel's
[Slug](https://github.com/EricLengyel/Slug), also MIT, and
[NOTICE](https://github.com/holofermes/ghee/blob/main/NOTICE) carries its
copyright notice.
