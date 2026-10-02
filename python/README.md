# ghee for Python

`pip install ghee-render`: one package, two surfaces. It embeds the
[ghee](https://github.com/holofermes/ghee) renderer, and it provides the
Python sugar for writing a
[procedural scene](https://github.com/holofermes/ghee/blob/main/docs/procedural.md).

Build the wheel yourself with maturin:

```
maturin build --release --compatibility linux
```

## Use

To use the `ghee` renderer:

```python
import ghee

h = ghee.open(
    "scene.json",      # a path, the JSON text of a scene, or a dict
    output="st7789",   # a driver name, "kms", or None for headless
    opts={"width": 240, "height": 240, "backlight": 26},
    width=None, height=None,   # a headless size, a panel states its own
    fps=30,            # pace cap, None runs free
    stats=8090,        # console port, ("0.0.0.0", 8090) binds elsewhere
    stream_max=None,   # h264 stream cap: pixels, or a fraction of the render size
    verbose=False,
    timestep=None,     # pin time to frame*S for a reproducible render
)

h.set_param("gain", 1.5)                 # a scene param
h.set_param("snap", 1.0, pass_="hold")   # the param of one pass
h.set_param("hold.snap", 1.0)            # the same param, dotted
h.set_param("words", "hello")            # a string param, drawn by a text

h.feed("level", bars)      # bytes, bytearray, memoryview, array, numpy, numbers
tick = h.frame()           # {"frame": 12, "time": 0.4, "dt": 0.0166}
tick = h.skip()            # the same tick, with no draw

h.size          # (width, height)
h.components    # 3 for RGB, 4 for RGBA
h.pixels()      # the last frame, row-major (height, width, components)
h.close()       # also a context manager
```

## What each call does

`set_param` sets a live uniform, no recompile. A bare name is the scene
param, else the param of the first pass that declares it. A string param
that a text draws lays out the new words, also with no recompile.

A scene given as a dict or as JSON text resolves its files in the working
directory, a scene file beside itself.

`feed` holds the data for one declared array, and the next frame consumes
it once. A uint8 buffer reaches the texture as byte / 255, a float32
buffer passes through. An array keeps its texture data until a feed
changes it.

`frame` draws and hands the frame to the output. `time` is the wall clock
since the first frame, `dt` is the measured frame period, and `timestep`
pins both to `frame * S`.

`skip` takes the tick and draws nothing, the panel keeps its last frame.

`stats` serves the console on `127.0.0.1`, and a `(host, port)` pair binds
it elsewhere. Anyone who reaches the port can change params, so bind beyond
loopback only on a network you trust.
[docs/http.md](https://github.com/holofermes/ghee/blob/main/docs/http.md)
holds the API. This console shows the host's own scene, param changes
apply, and scene switching and recompiles are refused. The video stream
advances with `frame()`: viewers see a new frame only when the host draws
one.

## Rules

ghee is thread-affine: open a handle on one thread and use it on that same
thread, a call from another thread raises RuntimeError.

A scene, GLSL or param error raises ValueError, a driver or system error
raises RuntimeError.

`frame()` releases the GIL, so the host's other threads run while it
renders.

## Python scenes

The shim is used to write a procedural scene. A PEP 723 scene should
declare `ghee-render` inline:

    #!/usr/bin/env -S uv run --script
    # /// script
    # dependencies = ["ghee-render"]
    # ///

The shim gives:

- `ghee.Scene / Pass / Param / ArrayInput / Channel / Asset / Text`: build
  the scene
- `ghee.Arg`, `ghee.args([...])`: argument declarations, and the `-- --help` answer
- `@ghee.feed("name")`: the values of a named array, each tick. A uint8
  buffer, `bytes` or a numpy uint8 array
- `@ghee.tick`: a hook for each tick, with `ghee.set_param(name, value)` and
  `ghee.skip()`
- `ghee.run(scene)`: the handshake, then the tick loop

## Text

`ghee.Text` draws the scene string param of its name on the GPU, from the
outlines of a font, sharp at any size. Every pass reads the words with
`float <name>_cover(vec2 p, float ppem)`: `p` in em from the top left of
the block, y down like `v_texCoord`, and `ppem` the pixels per em.
`<name>_size` is the block in em. A new value of the param lays out the
words again with no recompile, from `set_param` or the console.

    scene = ghee.Scene(
        params={"words": ghee.Param("words", "string", "hello")},
        assets=[ghee.Asset("sans", "DejaVuSans.ttf")],
        texts=[ghee.Text("words", "sans", align="center")],
        passes=[ghee.Pass("main", CODE)],
    )

The font is an asset file, beside the scene when ghee runs it and in the
working directory in a direct run. `ghee render --demo lettering -o
lettering.png --frames 1` shows the words at every size.

## Direct mode

The same scene file runs two ways. When ghee starts it, as `ghee render
spectrum.py` or from a scenes directory, `run()` feeds ghee over the
[wire protocol](https://github.com/holofermes/ghee/blob/main/docs/procedural.md).
When you run the file itself, `run()` opens the embedded renderer and
renders directly:

    ./spectrum.py --output kms --frames 300
    ./spectrum.py --bins 32 --output none --size 320x240 --frames 1

ghee sets `GHEE_WIRE=1` in every scene process it starts, and `run()`
tells the two apart by it.

The scene takes its own arguments first, then these:

- `--output NAME`: a driver name, `kms`, or `none` for a headless render
  (default `kms`)
- `--size WxH`: the render size. A display or a panel states its own
- `--fps N`, `--stats PORT`, `--timestep S`: as `ghee render` takes them
- `--frames N`: 0 runs until SIGINT or SIGTERM (default 0)

The feeds, the hooks and `ghee.skip()` work the same. The tick keeps its
wire shape, with `params` empty. If one of your own arguments collides with
these names, argparse says so.

## What the wire carries

The wire suits small per-frame data. Megabyte-scale data costs a
serialization and two copies for each frame, and on a small Pi that shows
up as a low frame rate. A data-heavy host takes `ghee.open` instead: the
renderer in the same process, no pipe.

## License

MIT. The text shader ports Eric Lengyel's
[Slug](https://github.com/EricLengyel/Slug), also MIT, and
[NOTICE](https://github.com/holofermes/ghee/blob/main/NOTICE) carries its
copyright notice.
