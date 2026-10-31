# Procedural scenes

A procedural scene is a program that ghee runs in place of a scene file. The
program writes its [scene](scenes.md) when it starts: its passes, each a
shader that draws a picture, and its layers, text, images and shapes drawn
over them. At runtime it changes the scene: it feeds arrays of numbers to
the shaders, sets params, the named values that the shaders read, and adds,
changes or removes layers. Write one in any language that reads stdin and
writes stdout. To write one in Python, use the `ghee` package, which does
everything on this page for you ([Python scenes](python.md#python-scenes)).

For example, this program, `bars.py`, writes a scene of 16 bars, then sends
their heights for each frame. It uses only Python's standard library, so
every byte that it sends is in the code. The scene declares the array
`levels` ([Arrays](scenes.md#arrays)), so ghee gives its shader
`levels_at(i)` to read it, beside `uv`, the pixel's position ([Built-in
names](scenes.md#built-in-names)). The program runs in
[lockstep](#lockstep), so each saved frame in the render below shows its own
heights:

```python
#!/usr/bin/env python3
import json
import math
import struct
import sys

FRAGMENT = """
void main() {
  float x = uv.x * 16.0;
  float bar = step(uv.y, levels_at(int(x))) * step(0.2, fract(x));
  gl_FragColor = vec4(vec3(bar), 1.0);
}
"""

scene = {
    "lockstep": True,
    "passes": [
        {
            "name": "main",
            "fragment": FRAGMENT,
            "arrays": [{"name": "levels", "size": 16}],
        }
    ],
}

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
    out.write(b"\x05")  # kind 5: done with this frame
    out.flush()
```

Make it executable, then render it. These commands render 63 frames, 0.05
seconds of scene time apart, one rise and fall of the wave, into
`f.0000.png` to `f.0062.png`:

```shell
chmod +x bars.py
ghee render bars.py -o f.####.png --size 128x64 --frames 63 --timestep 0.05
```

![16 white bars on black, rising and falling in a wave](../media/procedural/bars.webp)

To show it on HDMI or a small SPI or I2C panel instead, use `--display kms`.
It draws at the display's own size until Ctrl-C ([Hardware
displays](displays.md#hardware-displays)):

```shell
ghee render bars.py --display kms
```

## Running one

Binaries and scripts need the execute bit (`chmod +x`). Without it, ghee
stops with an error. A file whose name ends in `.scene.json` is always a
scene file, regardless of its permissions.

ghee finds a program the same way it finds a scene file ([Where ghee finds a
scene](scenes.md#where-ghee-finds-a-scene)), so `ghee render bars` finds
`bars.py` too. Next to `bars.scene.json`, the program runs in its place. Two
programs of one name stop ghee with `ambiguous scene 'bars': bars.py,
bars.sh: choose one`. `ghee serve` shows the program in the
[console](console.md)'s scene list, beside your scene files.

ghee starts the program in the program's own folder, so a relative path is
relative to the program.

## Arguments

To give the program arguments, put them after `--` on ghee's command line.
ghee hands the program everything after `--`. For a version of `bars.py`
that reads a `--count` flag, this renders one frame with 32 bars:

```shell
ghee render bars.py -o bars.png -- --count 32
```

Ignore the flags that your program doesn't take: `ghee serve` hands
everything after its `--` to every scene that it starts.

`ghee render bars.py -- --help` can list the program's flags:

```
usage: ghee render bars -- [ARGS]

bars:
  --count N               how many bars (default: 16)
```

To know them, ghee runs the program once with `--ghee-args` as its only
argument. The program prints its flags as one line of JSON and exits,
before it writes its scene:

```python
if "--ghee-args" in sys.argv:
    entry = {"name": "count", "type": "int", "default": 16, "help": "how many bars"}
    print(json.dumps({"args": [entry]}))
    sys.exit()
```

| Key | Values | Default | What it changes |
| --- | --- | --- | --- |
| `name` | the flag without its dashes | required | the flag that the listing shows |
| `type` | `string`, `int`, `float`, `size`, `bool` | `string` | the value that the listing shows: `STR`, `N`, `X`, `WxH`, or `--name, --no-name` for a bool |
| `default` | a value of the type: `true` for a bool, `"128x64"` or `[128, 64]` for a size | none | the default that the listing shows |
| `help` | text | none | the line that the listing shows |

Printing the flags is optional. A program that doesn't still gets its
flags, and the listing shows it as `bars: pass-through`: ghee doesn't know
the program's flags.

## What ghee gives the program

ghee sets these in the program's environment:

| Variable | Value | What it tells the program |
| --- | --- | --- |
| `GHEE_WIRE` | `1` | ghee started the program. It is unset when you run the program yourself |
| `GHEE_SIZE` | `WxH` | the size in pixels that the scene draws at. `--rotate 90` or `270`, which turn the picture on its way to the display, swap the two |

Use `GHEE_SIZE` when the scene's own sizes depend on the frame's, such as a
pass's `size` ([Size and sampling](scenes.md#size-and-sampling)).

## The wire

Every frame, ghee sends the program a tick, such as "frame 12, at 0.4
seconds". The program sends its data for that frame, such as the heights of
the bars. ghee doesn't wait for that data: it draws the frame with whatever
has arrived, so a slow program can't freeze the screen, and the data shows
up a frame late.

In [lockstep](#lockstep), the program asks ghee to wait, so frame 12 shows
what the program sent for frame 12. ghee waits on every frame when it
renders with `--timestep`, as into files. On a display it waits for the
first frame only, so a slow program still can't freeze the screen.

The program and ghee talk over three pipes, together called the wire. stdin
carries the ticks. stdout carries the program's scene, then its messages:
the messages that it writes after it reads a tick are its data for that
frame. stderr goes to ghee's terminal, and to the console when the scene
fails.

ghee reads only the bytes that reach the pipe. If your language buffers
stdout, as Python and Go's `bufio` do, flush it after the scene and after
each frame's data.

Careful: ghee reads every byte on stdout as part of the wire. Write anything
else, such as a debug line, to stderr.

### The scene

Before any frame, write the scene on stdout as one line of JSON, ending in a
newline: the same JSON that a [scene file](scenes.md#the-scene-file) holds.
To draw only layers, write one pass that clears to black:

```json
{"passes": [{"name": "main", "fragment": "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"}]}
```

Then add layers with kind 6, such as the text layer `{"name": "title",
"text": "hello"}` ([Changing layers](#changing-layers)).

ghee waits 30 seconds for the scene, then stops with an error.

Messages may follow the scene at once, before any tick. ghee applies each
to the first frame that starts after it arrives.

### Ticks

ghee writes one line of JSON to the program's stdin for each frame:

```json
{"frame": 2, "time": 0.1, "dt": 0.05, "params": {}}
```

| Entry | What it holds |
| --- | --- |
| `frame` | the frame's number, 0 at the program's first tick |
| `time` | seconds of real time since the program's first tick, 0 at that tick |
| `dt` | seconds since the tick before that the program got |
| `params` | the params that changed since the tick before, and every param in the first tick ([Params in a tick](#params-in-a-tick)) |

In real time, `dt` is at most 0.25, and 1/60 for the first tick. Under
`--timestep S`, `time` is `frame` times S, and `dt` is S for each frame
since the tick before, with no cap.

ghee sends tick N as it starts frame N, and draws frame N with the messages
that arrived before it.

ghee sends a tick for every frame, drawn or skipped. When the program reads
stdin slower than ghee sends ticks, ghee drops the ticks that it hasn't
written yet. The next tick carries their params, `frame` jumps, and `dt`
covers the whole gap. A program that never reads stdin gets no ticks,
writes its messages on its own clock, and ghee never waits on it.

Under `ghee serve` with no display, ghee draws a frame only while a page or
a stream watches, or for a request that changes the scene or asks for
`/frame.png`, so the program gets a tick only then.

#### Params in a tick

When someone changes a param in the [console](console.md), or through
[`POST /set`](http.md), the next tick tells the program:
`"params": {"speed": 2.0}`. A pass's own param comes as `blur.gain`, and a
text layer's words as `title.text`. The program's own changes never come
back to it.

### Lockstep

To ask ghee to wait, add `"lockstep": true` to the scene, and end each
frame's data with kind 5, a single byte that means "done with this frame".
Send kind 5 for every tick that you read, even when you have nothing else
to send, and never before the first tick.

### Messages

Each message is a header and its payload:

| Field | Size | Value |
| --- | --- | --- |
| kind | 1 byte | 0 to 6 |
| name length | 1 byte | 0 to 255 |
| name | the name length | UTF-8 |
| count | 4 bytes | an unsigned little-endian integer |
| payload | `count` × 4 bytes for kinds 0 and 1, `count` bytes for the rest | little-endian 32-bit floats, or bytes |

Kinds 2 and 5 are the kind byte alone, with no name length, name or count.
Every other kind writes its whole header. An empty name has a name length
of 0, and a message with no payload has a count of 0.

In the names below, `<layer>` and `<pass>` stand for a layer's and a pass's
names.

| Kind | Name | Payload | What it does |
| --- | --- | --- | --- |
| 0 | an array | floats | feeds the [array](scenes.md#arrays), a grid's values with the top row first |
| 0 | `<layer>.points` | floats | sets a line's points, two values each, in pixels from the pass's top left, y down |
| 1 | a param, or `<pass>.param` | floats | sets the param: 1 value for a float, an int or a bool (0 is false), 3 for a colour, N for a vecN |
| 1 | `<layer>.key` | floats | sets a number key of the layer, 1 value |
| 1 | `<layer>.visible` | floats | 1 value: 0 hides the layer, any other shows it |
| 1 | `<layer>.color` | floats | 3 values, red, green and blue from 0 to 1. A text's gradient takes 7: 3 for from, 3 for to, and the angle in degrees, 0 up and 90 right |
| 1 | `<layer>.key`, count 0 | none | puts `x`, `y`, `color`, `width`, `height` or `radius` back to its default |
| 2 | no name field | none | skips the next frame that it reaches, and a display keeps its last one. A render into files draws every frame |
| 3 | a param, or `<pass>.param` | UTF-8 text | sets a `string` param, or a colour param as a CSS hex colour or a CSS colour name |
| 3 | `<layer>.text` | UTF-8 text | sets a text layer's words |
| 3 | `<layer>.image` | UTF-8 text | sets an image layer's picture: an [image asset](layers.md#assets)'s name, or a `.png` or `.jpg` file's path |
| 3 | `<layer>.color` | UTF-8 text | a colour as a CSS hex colour or a CSS colour name, or a text's gradient as JSON, `{"from": "red", "to": "yellow", "angle": 90}` |
| 3 | `<layer>.key` | UTF-8 text | sets any other key of the layer: text, such as `mono` for `render`, or JSON for a list or an effect, such as `{"strength": 0.5}` for `glow` |
| 4 | an array | bytes | feeds the array, each byte read as byte / 255 |
| 5 | no name field | none | done with this frame, under [lockstep](#lockstep) |
| 6 | empty, length 0 | JSON | changes the scene's layers with a JSON Patch ([Changing layers](#changing-layers)) |

Kinds 0, 1 and 3 set a key of a layer. A key that doesn't change at runtime,
such as `font_size`, makes ghee build the layer again ([Changing a layer at
runtime](layers.md#changing-a-layer-at-runtime)). Passes and layers can't
share a name: a scene that reuses one fails to load.

Before each frame, ghee applies what arrived: the newest data of each array,
and the other messages in their order. Arrays and params keep their values
until a message changes them. Send as many values as an array holds: ghee
ignores extra values, and fills a short message with zeros.

These change nothing, and ghee prints the reason on its stderr, once for
each mistake:

- a param that the scene doesn't declare
- a value of the wrong type, or too few or too many values
- a layer or a pass that the scene doesn't have
- JSON that holds no JSON Patch, or a patch that fails ([Changing
  layers](#changing-layers))

```
  careful  bars.py: param 'nope' is not a param this scene declares, so it changes nothing
```

### Changing layers

To add, change, move or take out layers, send kind 6 with an [RFC
6902](https://www.rfc-editor.org/rfc/rfc6902) JSON Patch. Its document is
the scene's layers: an object of each pass's name and the list of its
layers, in the order that they draw, each layer as a scene file writes it,
with the keys at their defaults left out. `""` is the whole document,
`/main` the layers of pass `main`, `/main/2` the third of them, `/main/-`
the end of its list, `/main/2/x` a key of that layer, and
`/main/2/glow/strength` a field of its effect. This patch adds a text layer
as the first of `main`, moves the layer at 2 to the end, and puts the `x`
of the layer at 1 back to its default:

```json
[{"op": "add", "path": "/main/0", "value": {"name": "title", "text": "hi"}},
 {"op": "move", "from": "/main/2", "path": "/main/-"},
 {"op": "remove", "path": "/main/1/x"}]
```

The operations are `add`, `remove`, `replace`, `move`, `copy` and `test`.
To set a key, use `add`. `replace`, `remove`, and the `from` of `move` and
`copy` need the key to be there, so they fail on a key at its default. A
`null` value puts a key back to its default, as `remove` does.

A `replace` of a whole layer gives it the keys of the value. A key that the
value leaves out keeps the value from the
[console](console.md#what-a-program-locks), else the scene's own value for a
layer that the scene declares, else takes its default. A layer of the scene
keeps its name, and moves within its own pass. A pass stays in the scene.

ghee applies the patch whole or not at all, as RFC 6902 does. The
operations run in order, and ghee checks the layers' names on the result,
so a `copy` of a layer can take a new name in a later operation. When an
operation fails, or a `test` finds another value, the patch changes
nothing, and ghee prints the operation's place in the patch and why:

```
  careful  wire: kind 6 operation 1 at '/main/0/name': the test found "floor", so the patch changes nothing
```

### Checking the wire

To see what the program writes, run it yourself: pipe a tick into it, and
dump stdout. At time 0, `bars.py` ends with these 77 bytes: its `levels`
message, which is kind 0, a name length of 6, `levels`, a count of 16 and 16
floats, then kind 5:

```shell
echo '{"frame": 0, "time": 0.0, "dt": 0.016, "params": {}}' | GHEE_WIRE=1 ./bars.py | tail -c 77 | xxd
```

```
00000000: 0006 6c65 7665 6c73 1000 0000 0000 003f  ..levels.......?
00000010: da17 313f a82a 563f bc24 663f b01c 5d3f  ..1?.*V?.$f?..]?
00000020: 9748 3d3f 6073 0e3f db28 b83e 8203 4a3e  .H=?`s.?.(.>..J>
00000030: 1335 df3d 0173 ee3d bc02 5f3e 95c6 c63e  .5.=.s.=.._>...>
00000040: 3e07 163f 8246 433f 1b0d 603f 05         >..?.FC?..`?.
```

### Stopping

When the scene stops, ghee closes the program's stdin. Exit then: 2 seconds
later ghee sends SIGTERM, and SIGKILL 1 second after that, whether or not
the program reads stdin.

If the program exits while its scene runs, the scene stops with an error,
and ghee shows what the program wrote on stderr.
