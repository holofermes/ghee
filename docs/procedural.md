# Procedural scenes

A procedural scene is an executable that supplies its own scene and data.
ghee starts the executable when the scene becomes active, and talks to it
over stdin and stdout. A scene written in Python takes the
[shim](../python/README.md), which handles all of this.

## Scene shapes

A scene is one of:

- a `.scene.json` file
- a directory of `.scene.json` files (a set, for `ghee serve`)
- an executable file (a procedural scene)

## Probe

`ghee render <scene> -- --help` asks the scene to describe its arguments.
ghee runs `<scene> --ghee-args`, and the scene writes one JSON line and
exits:

    {"args": [{"name": ..., "type": ..., "default": ..., "help": ...}]}

The types are `string`, `int`, `float`, `size` and `bool`.

ghee gives the scene about two seconds to answer. If it answers, its
arguments are listed, if it is still starting the listing says that,
otherwise the scene is shown as pass-through, either way, everything after
`--` reaches the scene untouched. `--no-probe` skips the asking.

## Start

ghee runs the scene with everything after `--` as its arguments, unchecked.
If the scene rejects an argument and exits, ghee shows its stderr. The scene
runs in its own directory, with these variables in its environment:

- `GHEE_WIRE=1` marks a process that ghee started.
- `GHEE_LOCKSTEP=1` says that ghee reads kind 5 and the handshake's
  `"lockstep"`. A process that finds no `GHEE_LOCKSTEP` sends neither.
- `GHEE_SIZE` holds ghee's render size as WxH, so the scene can size its
  buffers before it writes its scene.

## Handshake

Before any frame, the process writes one line to stdout: a scene JSON object.
ghee waits 30 seconds for the scene. If the process exits before it writes
the scene, that is an error, and ghee shows its stderr. A process that ends
its answer to every tick with kind 5 sets `"lockstep": true` in that object.

The console saves its edits of a procedural scene in a sidecar beside the
executable, `modified/<name>.values.json`, and lays the sidecar over the
scene that the process writes at each start. The process keeps which passes
and layers exist and their order. `/save` in [http.md](http.md) lists what
the sidecar holds.

## Messages, scene to ghee

After the handshake, stdout carries binary messages:

    kind: u8, name_len: u8, name: UTF-8, count: u32 LE, values: count x f32 LE

- Kind 0 sets the named array. A 2D array takes w*h values, in its declared
  shape. A name whose first part names a layer, such as `scope.points`, sets
  that line's points, x and y in turn.
- Kind 1 sets the named live param. A float takes one value, a color three,
  and a vecN N. A name of the form `pass.param` sets the param of that one
  pass. A bare name sets the scene param, or the param of the first pass that
  declares it. A name whose first part names a layer, `layer.field` or
  `layer.effect.field` such as `title.x` or `title.karaoke.sung`, sets that
  live field of the layer, one of the fields that `/set?layer=` takes in
  docs/http.md. A number takes one value, `visible` takes 0 or 1, a color
  three, and a text layer's gradient seven: from, to and its angle. A rect's
  border takes an effect's place, as `frame.border.width`. A message with no
  values takes a layer's `x`, `y` or `color`, or a rect's `width` or
  `height`, back to its default.
- Kind 2 marks the tick clean: it needs no render, and the display keeps the
  last frame. The message is the kind byte alone, with no name and no values.
- Kind 3 sets the named string param. The name of a text layer, or
  `layer.text` such as `title.text`, sets its words, and a name of the form
  `layer.image`, such as `icon.image`, sets that image layer's `image` to the
  image asset or the `.png` or `.jpg` file that the text names. A color param
  and a layer's colour, such as `title.color` or `title.glow.color`, take the
  colour as text, `#rrggbb` or a CSS colour name, which ghee keeps as written,
  and a text layer's `color` takes its gradient as the JSON object. The header
  is the same as kind 0's, and the payload is count bytes of UTF-8 text.
- Kind 4 sets the named array from u8 values. The header is the same as kind
  0's, and the payload is count u8 bytes, each value byte / 255. A big mask
  then costs a quarter of the bytes on the pipe.
- Kind 5 ends the messages that answer one tick. Like kind 2, it is the kind
  byte alone.

ghee reads without blocking and applies the newest data. Arrays and params
keep their values until a message changes them. A message that names no param
or live field of the scene, or that does not fit its type, changes nothing,
and ghee logs it once.

## Ticks, ghee to scene

ghee writes one JSON line to the process's stdin for each render tick:

    {"frame": N, "time": T, "dt": D, "params": {...}}

`D` is the DT that the tick's frame draws with: the time since the frame
before, a quarter second at most, 1/60 for the first frame, or the fixed
`--timestep`. `params` holds the string params that changed, and the first
tick holds them all.

Reading stdin paces your process to the render, sleeping when it sleeps.
Ignoring stdin lets it run free, but it then gets no ticks, no params, no
skip. ghee never blocks on a full stdin pipe.

A frame takes the messages that arrived before it starts, so the answer to a
tick lands one frame later at the earliest, and later still when the process
falls behind. With a fixed `--timestep`, ghee waits up to 30 seconds for a
lockstep process to end its answer to each tick, and the answer lands in that
tick's frame.

## Stop

When the scene stops, ghee closes stdin, and the process must exit. ghee
sends SIGTERM after 2 seconds and SIGKILL 1 second later. If the process
exits while its scene is still active, the scene goes into the error state,
and ghee shows its stderr.
