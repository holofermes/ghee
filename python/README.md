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

This script shows a clock on a 128x32 SSD1306. The text layer `clock` names
no font, so it draws in ghee's built-in DejaVu Sans, and no size, so the words
fit the panel. Each turn of the loop sets the words and draws one frame:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["ghee-render"]
# ///
import time

import ghee

scene = ghee.Scene(
    passes=[ghee.Pass("main", layers=[ghee.Layer("clock", text="00:00:00")])],
)

with ghee.open(scene, output="ssd1306", fps=4) as screen:
    while True:
        screen.layers.clock.text = time.strftime("%H:%M:%S")
        screen.frame()
```

`output="kms"` takes a panel with a kernel driver in place of ghee's own
`ssd1306` driver.

The same clock with a glow around the digits takes the `glow` effect. A 1-bit
panel leaves every dim pixel dark, so the `dither` filter draws the glow as
dots:

```python
scene = ghee.Scene(
    passes=[
        ghee.Pass(
            "main",
            layers=[
                ghee.Layer(
                    "clock",
                    text="00:00:00",
                    glow={"color": "#ffffff", "strength": 0.4},
                )
            ],
        )
    ],
    filters=["dither"],
)
```

[Layers](https://github.com/holofermes/ghee#layers) holds more: `font_size`,
`x` and `y`, every effect, image layers, and what each filter does on a 1-bit
panel.

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
h.set_param("title", "hello")            # the words of the text layer title
h.layers.title.x = 10                    # a live field of a layer
h.layers.title.karaoke.sung = 12         # a live field of one of its effects

h.feed("level", bars)      # bytes, bytearray, memoryview, array, numpy, numbers
tick = h.frame()           # {"frame": 12, "time": 0.4, "dt": 0.0166}
tick = h.skip()            # the same tick, with no draw
h.changed_params()         # {"title": "hi"}: string params that the console set

h.size          # (width, height)
h.components    # 3 for RGB, 4 for RGBA
h.pixels()      # the last frame, row-major (height, width, components)
h.save("frame.####.png")   # the last frame as a .png or a .jpg file
h.close()       # also a context manager
```

## What each call does

`set_param` sets a live param, with no recompile. A bare name sets the scene
param, else the param of the first pass that declares it. A text layer's
words are the string param of the layer's name, and ghee lays new words out
again, also with no recompile.

`layers` sets a live field of a layer, by attribute or by key:
`h.layers.title.x = 10`, `h.layers["title"]["karaoke"]["sung"] = 12`.
`set_layer(name, field, value, effect=None)` makes the same set as a call. The
live fields are a layer's `x`, `y`, `color`, `opacity` and `visible`, a text
layer's `text`, the words, an image layer's `image`, the name of an image
asset of the scene or an image file, an effect's `visible`, `color`,
`strength`, `sung`, `speed` and `rate`, and every field of a shape layer but
`shape` and `blend`, a rect's border as `h.layers.frame.border.color`. A value
takes the form that the scene gives the field, and None takes `x`, `y`,
`color`, or a rect's `width` or `height` back to its default. A set raises
ValueError at once for a layer, an effect or a field that the scene lacks, for
a field that changes with a rebuild, and for a value that the field does not
take. `layers` sets fields and reads none back.

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
holds the API. This console shows the host's own scene. Params and the live
fields of layers change from it, and it refuses scene switches, saves and
every edit that rebuilds the scene. The video stream advances with
`frame()`: viewers see a new frame only when the host draws one.

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

- `ghee.Scene / Pass / Param / ArrayInput / Channel / Asset / Layer` to build
  the scene.
- `ghee.Arg` and `ghee.args([...])` to declare the scene's arguments, and to
  answer `-- --help`.
- `@ghee.feed("name")` to supply the values of a named array, each tick. The
  function returns a buffer or an iterable of numbers. A uint8 buffer, such
  as `bytes`, a `bytearray` or a numpy uint8 array, goes out as byte / 255,
  and any other numbers go out as float32. None leaves the array as it
  stands.
- `@ghee.tick` to run a hook on each tick, before the feeds. The hook
  receives a `ghee.Tick` and calls `ghee.set_param(name, value)`,
  `ghee.layers` and `ghee.skip()`.
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
    ghee.layers.clock.text = time.strftime("%H:%M:%S")


ghee.run(
    ghee.Scene(
        passes=[ghee.Pass("main", layers=[ghee.Layer("clock", text="00:00:00")])],
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

`ghee.layers` makes the sets of a handle's `layers` from a tick or a feed,
and they land as a param does. Under ghee, each set goes out with the tick's
answer, and the first one loads the native half of the package, which checks
every set against the scene and raises ValueError where ghee would refuse
it. In a direct run, each set goes to the handle. A set before `ghee.run`
raises RuntimeError.

## Layers

`ghee.Layer(name, text=None, visible=True, x=None, y=None, color=None,
opacity=None, blend="normal", font_size=None, font_family=None,
text_align="left", render=None, text_shadow=None, glow=None,
text_stroke=None, karaoke=None, wave=None, scroll=None, blink=None,
image=None, width=None, height=None, shape=None, border=None,
border_radius=None, points=None, stroke_width=None)` takes the keys of a layer
as keywords, and [Layers](https://github.com/holofermes/ghee#layers) holds
what each does. A text layer takes `text`, an image layer `image`, and a shape
layer `shape`. A keyword left at None leaves its key to ghee's default, and an
effect takes a dict of its fields, `{}` for every default. This scene draws
`hello` in a gradient with a soft shadow, and the image `stamp.png` in the top
left corner:

```python
scene = ghee.Scene(
    assets=[ghee.Asset("stamp", "stamp.png")],
    passes=[
        ghee.Pass(
            "main",
            "void main() { gl_FragColor = vec4(0.1, 0.12, 0.2, 1.0); }",
            layers=[
                ghee.Layer(
                    "title",
                    text="hello",
                    color={"from": "#ff0000", "to": "#ffff00", "angle": 90},
                    text_shadow={},
                ),
                ghee.Layer("icon", image="stamp", x=4, y=4, width=16),
            ],
        )
    ],
)
```

A pass of layers alone, `ghee.Pass("words", layers=[...])`, starts each frame
transparent, for a later pass to read through a channel.

A text problem leaves the scene running, and `set_param` raises nothing for
it. ghee prints it once on stderr, and the console shows it.

## Coverage in a shader

Every pass except a filter pass reads the words of a text layer with
`float <name>_cover(vec2 p, float font_size)`: `p` in em from the top left of
the block, y down like `v_texCoord`, and `font_size` the pixels per em there,
which sets the antialiasing. `<name>_size` is the block in em. A hidden layer
gives a shader its words and draws none itself. With `CODE` holding a shader
such as the one in
[Coverage in a shader](https://github.com/holofermes/ghee#coverage-in-a-shader):

    scene = ghee.Scene(
        passes=[ghee.Pass("main", CODE, layers=[ghee.Layer("title", text="hello", visible=False)])],
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

The feeds, the hooks, `ghee.layers` and `ghee.skip()` work the same. The tick keeps its wire
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
