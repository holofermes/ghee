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

## Drawing on a panel

This script shows the time on a 128x32 SSD1306 that the kernel drives, set up
as [Hardware displays](https://github.com/holofermes/ghee#hardware-displays)
shows. Each `with` block draws one frame, four a second, and the text names
no size or position, so it fits the panel, centred:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["ghee-render"]
# ///
import time

import ghee

screen = ghee.open(display="kms", fps=4)
while True:
    with screen.draw() as d:
        d.text(time.strftime("%H:%M:%S"))
```

[Drawing](#drawing) holds every call, and [Text on 1-bit and 16-level
panels](https://github.com/holofermes/ghee#text-on-1-bit-and-16-level-panels)
holds the fonts and the filters that suit a 1-bit panel.

## ghee.open

```python
import ghee

h = ghee.open(
    "scene.json",      # a path, the JSON text of a scene, a dict, a ghee.Scene, or None
    display="kms",     # the connected display, or None for headless
    opts=None,         # {"rgb565": True} scans "kms" out in 16-bit color
    dri=None,          # the card, as --dri takes it
    rotate=0,          # 0, 90, 180 or 270 degrees clockwise
    width=None, height=None,   # a headless size, a display states its own
    fps=30,            # pace cap, None runs free
    serve=8090,        # console port, ("0.0.0.0", 8090) binds elsewhere
    stream_max=None,   # h264 stream cap: pixels, or a fraction of the render size
    verbose=False,
    timestep=None,     # pin time to frame*S for a reproducible render
    alpha=False,       # keep the output pass's alpha in pixels()
    base_dir=None,     # where a scene given as no path finds its files
)

h.set_param("gain", 1.5)                 # a scene param
h.set_param("snap", 1.0, pass_="hold")   # the param of one pass
h.set_param("hold.snap", 1.0)            # the same param, dotted
h.layers.title.text = "hello"            # the words of the text layer title
h.layers.title.x = 10                    # a live field of a layer
h.layers.title.karaoke.sung = 12         # a live field of one of its effects

h.feed("level", bars)      # bytes, bytearray, memoryview, array, numpy, numbers
tick = h.frame()           # {"frame": 12, "time": 0.4, "dt": 0.0166, "params": {}}
tick = h.skip()            # the same tick, with no draw

with h.draw() as d:        # one frame of words, pictures and shapes
    d.text("hello", x=4, y=4)
w, ht = h.measure("hello", font_size=12)   # the block's size in pixels
h.brightness = 0.5         # a KMS display's backlight, 0 to 1

h.size          # (width, height)
h.components    # 3 for RGB, 4 for RGBA
h.pixels()      # the last frame, row-major (height, width, components)
h.save("frame.####.png")   # the last frame as a .png or a .jpg file
h.close()       # also a context manager
```

## What each call does

`set_param` sets a live param, with no recompile. A bare name sets the scene
param, else the param of the first pass that declares it.

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
since the first frame, and `dt` is the `iTimeDelta` that the frame draws with:
the time since the frame before, a quarter second at most, and 1/60 for the
first frame. `timestep` pins `time` to `frame * S` and `dt` to S. `params`
holds the params that the console changed since the last tick, with their
values as a scene file writes them, and a text layer's words as its `text`,
such as `title.text`. The first tick holds them all, and a set from the host
itself stays out.

`pixels` reads alpha 255 in every pixel, unless `alpha=True` keeps the alpha
that the output pass wrote, with the colour premultiplied by it.

`skip` takes the tick and draws nothing, and the panel keeps its last frame.

`save` writes the last frame to a `.png` file, or to a `.jpg`, which holds no
alpha. A run of `#` in the name takes the number of the frame's tick, as
`ghee render -o` numbers its files.

`serve` serves the console on `127.0.0.1`, and a `(host, port)` pair binds it
elsewhere. Anyone who reaches the port can change params, so bind beyond
loopback only on a network you trust.
[docs/http.md](https://github.com/holofermes/ghee/blob/main/docs/http.md)
holds the API. This console shows the host's own scene, the layers of a
drawing among them, as the host changes it. Params, the live fields of
layers and the look of each layer change from it, and it refuses scene
switches, saves and every other edit. The video stream advances with
`frame()`: viewers see a new frame only when the host draws one.

With no scene, `ghee.open` makes one pass, `main`, that clears to black, for
a drawing to paint on. `rotate` turns the frame clockwise on its way to every
output, the display, the files and the console's video, and at 90 and 270 the
scene renders at the frame's size turned, so a 128x32 panel gives the scene
32x128. `h.size` is the frame's size, the size of `pixels()`.

`brightness` reads and sets the backlight of a KMS display, 0 to 1, through
the backlight device that the kernel registered for it, such as a panel's
contrast. It reads None for a display with no backlight device, and a set
raises ValueError there and on a screen with no KMS display. Closing the
screen puts the earlier level back.

## Drawing

`screen.draw()` opens one frame for a `with` block, and each call in the
block writes a layer. Leaving the block draws the frame, and the panel holds
it until the next, so the host's loop sets the pace: a loop that sleeps
freezes the scroll, the wave and the blink between its frames.

This loop draws the time, the load of the machine as a share of its cores
with a bar beside it, and a blinking HOT past 90%:

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
        d.rect(x=64, y=20, width=64, height=8, color="transparent",
               border={"width": 1, "color": "white"})
        d.rect(x=66, y=22, width=60 * min(load, 1), height=4)
        if load > 0.9:
            d.text("HOT", x=100, y=0, font_size=12, blink={"rate": 2})
```

The calls, each with the layer that it writes:

```python
d.text("hello", x=4, y=4, font_size=12, color="orange", glow={"blur": 0.25})
d.text("hello")                                       # fitted and centred
d.text("hi", color=ghee.gradient("red", "yellow", angle=90))
d.image("wifi.png", x=110, y=0, width=32, color="orange")   # an image layer
d.rect(x=0, y=20, width=91, height=9, color="white")  # a shape layer
d.rect(color="navy")                                  # the whole screen
d.circle((110, 14), 10, color="white")                # a box with round ends
d.line((0, 31), (127, 31), color="white", stroke_width=1)
d.lines(points, color="lime", stroke_width=1)         # a waveform
```

A call takes the keys of its layer as keywords, the effects included, as
`ghee.Layer` takes them, and `name`. Positions and box sizes are pixels from
the top left, y down, on the scene's last pass before its filters, where the
drawing paints over the scene's own layers in the order of the calls.
`d.circle(centre, radius)` writes a box of `2 * radius` with `border_radius`
at `radius`, and `d.line(a, b)` is `d.lines([a, b])`, whose points stand
where they say: `x` and `y` shift a line, and no line is centred.

Each frame states the whole look of each call, so a keyword left out takes
its default again. A call keeps its layer from frame to frame by the line
of code that makes it, and by its count from that line in the frame, so a
call that comes and goes above another leaves the other's layer alone.
`name` names the layer, for a shader that reads `<name>_cover`, and takes the
place of the line, and when the scene holds a layer of that name, the call
draws it. A call with no name names its layer after its kind, `text`,
`text2`, `image`, `rect`, `line`, past every name of the scene. Two calls
that draw one layer in a frame raise ValueError.

Leaving the block sends ghee what changed. A live field changes live. Any
other change builds that layer alone again, so an animated `font_size`
builds its layer every frame. A call that runs for the first time adds its
layer after the layer of the call before it, a layer whose call did not run
hides, and an effect that a call stops naming is bypassed. A layer keeps its
place once added, and stays, hidden, after its call stops. A frame whose
calls changed nothing sends nothing.

`screen.measure(words, font_size=None, font_family=None)` gives the width
and height in pixels of the block that a drawing's text lays `words` out in,
at the size that fits the screen when `font_size` is None.

## Rules

ghee is thread-affine: open a handle on one thread and use it on that same
thread. A call from another thread raises RuntimeError.

A scene, GLSL or param error raises ValueError, and a system error raises
RuntimeError.

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
- `ghee.render_size()`, the size that the scene renders at, as `(w, h)`. A
  pass with a size or a scale reads its own size in `iResolution`, so a
  scene that needs the frame's shape there writes it from this.
- `@ghee.feed("name")` to supply the values of a named array, each tick. The
  function returns a buffer or an iterable of numbers. A uint8 buffer, such
  as `bytes`, a `bytearray` or a numpy uint8 array, goes out as byte / 255,
  and any other numbers go out as float32. None leaves the array as it
  stands.
- `@ghee.tick` to run a hook on each tick, before the feeds. The hook
  receives a `ghee.Tick` and calls `ghee.set_param(name, value)`,
  `ghee.layers`, `ghee.draw()`, `ghee.measure` and `ghee.skip()`.
- `ghee.run(scene)` for the handshake, then the tick loop.

A `ghee.Tick` holds `frame`, the tick's number, `time` and `dt` in seconds,
`dt` the `iTimeDelta` that its frame draws with, and `params`, the params
that changed, by name, and a text layer's words as its `text`, such as
`title.text`. The first tick's `params` holds them all.

This scene is the clock above, written as a procedural scene. With no scene,
`ghee.run()` starts from one pass, `main`, that clears to black, and the
first tick's drawing grows it:

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

`ghee render --demo stats` draws the time and the load of the machine this
way.

`params` on `ghee.Scene` and `ghee.Pass` takes a list of `ghee.Param`, such as
`params=[ghee.Param("gain", "float", 0.5, min=0.0, max=1.0)]`. A dict keyed
by name works too, and a key that differs from its param's name raises
ValueError.

`ghee.Pass` takes `size=(w, h)` for a buffer of its own in pixels, or
`scale=`, a share of the frame, one number or `(x, y)`, so
`ghee.Pass("blur", BLUR, scale=0.5)` renders at half the frame on every
screen.

`ghee.Asset(name, file)` takes the keys of an asset of the scene format:
`sampling`, "nearest" or "linear", and for a `.raw` file of bare pixels its
`width`, `height`, `format`, "u8" or "f32", and `components`.

The shim builds the scene and leaves every check to ghee, so a mistake in the
scene reads the same in `ghee.open`, in a direct run and under `ghee render`.

`ghee.set_param` raises ValueError for a param that the scene does not
declare, or for a value that does not fit the param's type. A param set in a
hook lands in the frame of its tick in [direct mode](#direct-mode), and under
ghee with `--timestep`. Under ghee without it, the param lands a frame later
or more.

`ghee.measure` measures words as a handle's `measure` does, on the last pass
of the scene that `run()` runs, at the render size that ghee gives the scene,
and raises RuntimeError outside a tick or a feed.

`ghee.layers` makes the sets of a handle's `layers` from a tick or a feed,
and they land as a param does. Under ghee, each set goes out with the tick's
answer, and the first one loads the native half of the package, which checks
every set against the scene and raises ValueError where ghee would refuse
it. In a direct run, each set goes to the handle. A set before `ghee.run`
raises RuntimeError.

`ghee.draw()` opens a [drawing](#drawing) of the tick's frame in a tick or
a feed, and the frame of that tick draws its calls, so one `with` block in
a tick states the whole drawing of that frame. The calls land as the sets of
`ghee.layers` do, checked against the scene first, and under ghee a layer
that a call adds or changes goes out with the tick's answer.

The same clock from a scene of its own declares its text layer, and the
tick sets the layer's words through `ghee.layers` in place of the drawing:

```python
@ghee.tick
def clock(tick):
    ghee.layers.clock.text = time.strftime("%H:%M:%S")


ghee.run(
    ghee.Scene(
        passes=[ghee.Pass("main", layers=[ghee.Layer("clock", text="00:00:00")])],
    )
)
```

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

A text problem leaves the scene running, and a set of the layer's `text`
raises nothing for it. ghee prints it once on stderr, and the console shows
it.

## Coverage in a shader

Every pass except a filter pass reads the words of a text layer with
`float <name>_cover(vec2 p, float font_size)`: `p` in em from the top left of
the block, y down like `uv`, and `font_size` the pixels per em there,
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
  on stdout. Files and the pipe take every frame, a skipped one too, and
  more than one frame takes a pattern that numbers them.
- `--display [kms]`: the connected display, bare or `kms`. It is the default
  with no `-o` and no `--serve`.
- `--serve`, `--port PORT`, `--host HOST`: the console.
- `--size WxH`: the render size. With no `--size`, a display states its
  own, which `ghee.render_size()` and a size arg given as a share, such as
  `half`, take too, turned by `--rotate`. A run with no display renders at
  320x240.
- `--rotate DEGREES`: 0, 90, 180 or 270, the frame turned clockwise on its
  way to every output.
- `--brightness LEVEL`: the backlight of the `kms` display, 0 to 1. A
  display that refuses it prints the error once, and the run goes on.
- `--rgb565`: the `kms` display scanned out in 16-bit color.
- `--dri DEVICE`: the DRI device, and with `--display` the card whose
  display it drives.
- `--fps N`, `--stats` and `--timestep S`.
- `--frames N`: the frame count. It defaults to 1 for image files or the
  pipe, and to 0 for a display or the console, which runs until SIGINT or
  SIGTERM.

The feeds, the hooks, `ghee.layers`, `ghee.draw()` and `ghee.skip()` work the
same. The tick keeps its wire shape: `dt` is the `iTimeDelta` that its frame
draws with, `params` holds the params that the console changed, and the first
tick holds every param. A scene or a display that ghee refuses
stops the run with ghee's one line and exit status 1. If one of your own
arguments collides with these names, argparse says so.

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
