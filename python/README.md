# ghee for Python

`pip install ghee-render`, then `import ghee`. One package holds two surfaces.

**The renderer.** `ghee.open` opens a GLSL scene, feeds it and asks it for
frames. The renderer, the panel drivers and the console are the ones the `ghee`
command uses. It lives in the native module `ghee._native`, and the package
imports that on the first `open()`.

**The scene shim.** `ghee.Scene`, `@ghee.feed`, `@ghee.tick` and `ghee.run`
write a procedural scene: an executable ghee starts, feeds and renders.
`docs/procedural.md` holds it. A scene no ghee started hosts itself on the
renderer above. The shim is Python alone, and it needs no native module.

```
maturin build --release --compatibility linux
```

The wheel is abi3-py39: one wheel per architecture serves CPython 3.9 and up.
It links libEGL, libgbm and libdrm from the system, as the ghee package does.

## Use

```python
import ghee

h = ghee.open(
    "scene.json",      # a path, the JSON text of a scene, or a dict
    output="st7789",   # a driver name, "kms", or None for headless
    opts={"width": 240, "height": 240, "backlight": 26},
    width=None, height=None,   # a headless size; a panel states its own
    fps=30,            # pace cap; None runs free
    stats=8090,        # console port; ("0.0.0.0", 8090) binds elsewhere
    verbose=False,
    timestep=None,     # pin time to frame*S for a reproducible render
)

h.set_param("gain", 1.5)                 # a scene param
h.set_param("snap", 1.0, pass_="hold")   # the param of one pass
h.set_param("hold.snap", 1.0)            # the same param, dotted

h.feed("level", bars)      # bytes, bytearray, memoryview, array, numpy, numbers
tick = h.frame()           # {"frame": 12, "time": 0.4, "dt": 0.0166}
tick = h.skip()            # the same tick, with no draw

h.size          # (width, height)
h.components    # 3 for RGB, 4 for RGBA
h.pixels()      # the last frame, row-major (height, width, components)
h.close()       # also a context manager
```

## What each call does

`set_param` sets a live uniform, with no recompile. A bare name is the scene
param, else the param of the first pass that declares it.

`feed` holds data for one declared array. The next frame consumes it once. A
uint8 buffer reaches the texture as byte / 255, and a float32 buffer passes
through. An array no feed set keeps what its texture holds.

`frame` draws and hands the frame to the output. `time` is the wall clock since
the first frame, and `dt` is the measured frame period. `timestep` pins both to
`frame * S`.

`skip` takes the tick and draws nothing. The panel keeps its last frame.

`stats` serves the console on 127.0.0.1: `/stats`, `/scene`, `/scenes`,
`/stream`, `/h264`, `/frame.png` and POST `/set`. A `(host, port)` pair binds it
on another address. The console has no auth, so give it a network you trust.
This console renders the one scene its host opened, so it switches no scene and
compiles none.

`/h264` is the console's video stream, framed for a browser's WebCodecs
decoder. One encoder feeds every viewer, and it runs only while a viewer
watches. A board with a V4L2 encoder takes the frame off the GPU with no
readback. Every other board encodes through ffmpeg/libx264, which takes ffmpeg
on PATH. The stream advances with `frame()`: a host that draws nothing sends
nothing.

## Rules

ghee is thread-affine. Open a handle on one thread, and use it on that same
thread. A call from another thread raises RuntimeError.

A scene, a GLSL or a param error raises ValueError. A driver or system error
raises RuntimeError.

A frame releases the GIL, so the host's other threads run while it renders.
