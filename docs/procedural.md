# Procedural scenes

A procedural scene is an executable that supplies its own scene and data.
ghee runs the executable when the scene becomes active and talks to it
over stdin and stdout. If you write yours in Python, the
[shim](../python/README.md) handles all of this for you.

## Scene shapes

A scene is one of:

- a `.scene.json` file
- a directory of `.scene.json` files (a set, for `ghee serve`)
- an executable file (a procedural scene)

## Probe

`ghee render <scene> -- --help` asks the scene to describe its arguments
by probing it with `<scene> --ghee-args`, which expects the program to
write one JSON line and exit afterward:

    {"args": [{"name": ..., "type": ..., "default": ..., "help": ...}]}

The types are `string`, `int`, `float`, `size` and `bool`.

ghee gives the scene about two seconds to answer. If it answers, its
arguments are listed, if it is still starting the listing says that,
otherwise the scene is shown as pass-through, either way, everything after
`--` reaches the scene untouched. `--no-probe` skips the asking.

## Start

ghee runs the scene with everything after `--` as its arguments, unchecked.
If the scene rejects an argument and exits, ghee shows its stderr. The
working directory is the scene's own directory, and ghee sets
`GHEE_WIRE=1` in the process. ghee also sets `GHEE_LOCKSTEP=1` to say that
it reads kind 5 and the handshake's `"lockstep"`, and a process that finds
no `GHEE_LOCKSTEP` sends neither.

## Handshake

Before any frame, the process writes one line to stdout: a scene JSON
object. ghee waits 30 seconds for the scene, however if the process exits
without the scene, that's an error and ghee shows its stderr. A process that
ends its answer to every tick with kind 5 sets `"lockstep": true` in that
object.

## Messages, scene to ghee

After the handshake, stdout carries binary messages:

    kind: u8, name_len: u8, name: UTF-8, count: u32 LE, values: count x f32 LE

- kind 0 sets the named array. A 2D array takes w*h values, in its declared
  shape.
- kind 1 sets the named live param. A float takes one value, a color three,
  a vecN takes N. A name of the form `pass.param` targets that one pass. A
  bare name targets the scene param, or the first pass that declares it.
- kind 2 means clean: this tick needs no render, and the display keeps the
  last frame. It's the single kind byte, no name, no values.
- kind 3 sets the named string param, and the name of a text sets its
  words. Same header as kind 0, the payload is count bytes of UTF-8 text.
- kind 4 sets the named array from u8 values. Same header as kind 0, the
  payload is count u8 bytes, value = byte / 255. A big mask costs a quarter
  of the pipe.
- kind 5 ends the messages that answer one tick. It's the single kind byte,
  like kind 2.

ghee reads without blocking and applies the newest data. Arrays and params
keep their values until a message changes them. A message that names a
param the scene does not hold, or that does not fit the param's type,
changes nothing, and ghee logs it once.

## Ticks, ghee to scene

ghee writes one JSON line to the process stdin for each render tick:

    {"frame": N, "time": T, "dt": D, "params": {...}}

`params` holds the string params that changed, and the first tick holds
them all.

Reading stdin paces your process to the render, sleeping when it sleeps.
Ignoring stdin lets it run free, but it then gets no ticks, no params, no
skip. ghee never blocks on a full stdin pipe.

A frame takes the messages that arrived before it starts, so the answer to a
tick lands one frame later at the earliest, and later still when the process
falls behind. With a fixed `--timestep`, ghee waits up to 30 seconds for a
lockstep process to end its answer to each tick, and the answer lands in
that tick's frame.

## Stop

When the scene stops, ghee closes stdin and the process must exit. ghee
sends SIGTERM after 2 seconds and SIGKILL 1 second later. If the process
exits while its scene is still active, the scene goes into the error state
and ghee shows its stderr.
