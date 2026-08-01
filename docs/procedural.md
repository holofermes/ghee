# Procedural scenes

A procedural scene is an executable that supplies its scene and its data.
ghee starts the scene process when the scene becomes active.
The process can be a program in any language.
ghee talks to the process on its stdin and stdout.

## Scene shapes

A scene is one of:

- a `.scene.json` file
- a directory of `.scene.json` files (a set, for serve)
- an executable file (a procedural scene)

## Probe

ghee runs `<scene> --ghee-args` before the first start.
The scene writes one JSON line and stops:

    {"args": [{"name": ..., "type": ..., "default": ..., "help": ...}]}

The types are `string`, `int`, `float`, `size` and `bool`.
ghee waits 2 seconds for the line.
A scene that does not answer is a pass-through.
ghee hands a pass-through scene the tokens as they are, and checks nothing.

## Start

ghee runs the scene with the tokens after `--` as arguments.
The working directory is the directory of the scene.

## Handshake

The process writes one line to stdout before any frame: a scene JSON object.
ghee waits 30 seconds for that line.
A first `uv run` resolves packages inside that wait.
A process that stops without the line is an error.
ghee shows the stderr of the process.

## Messages, scene to ghee

After the handshake, stdout carries binary messages:

    kind: u8, name_len: u8, name: UTF-8, count: u32 LE, values: count x f32 LE

- kind 0 sets the named array. A 2D array sends w*h values. The declaration of
  the scene gives the shape.
- kind 1 sets the named live param. A float takes one value, a color takes
  three, and a vecN takes N. The param is the param the console and the scene
  file hold. A name of the form `pass.param` sets the param of that one pass. A
  bare name is the scene param, else the param of the first pass that declares
  it.
- kind 2 is clean. This tick needs no render, and the display keeps the last
  frame. Kind 2 carries no name and no values: it is the single byte.

ghee reads without blocking, and applies the newest data.
A name that no message sets keeps its data.

## Ticks, ghee to scene

ghee writes one JSON line to the process stdin for each render tick:

    {"frame": N, "time": T, "dt": D, "params": {...}}

The params object holds the string params that changed.
The first tick holds every string param.
A process that reads stdin and answers each tick runs at the pace of ghee.
Such a process sleeps when the render sleeps.
A process that ignores stdin runs free.
A free process loses the ticks, the params and the skip.
ghee never blocks on a full stdin pipe.

## Stop

ghee closes stdin when the scene stops.
The process must exit.
ghee sends SIGTERM after 2 seconds, and SIGKILL 1 second later.
A process that exits while its scene is active puts the scene in the error
state.
ghee shows the stderr of the process.

## Python scenes

A Python scene imports the shim (`import ghee`).
A scene that needs packages declares them inline (PEP 723).
Such a scene starts with `#!/usr/bin/env -S uv run --script`.
The bundled examples ship the shim beside them, under `#!/usr/bin/env python3`.
The shim gives:

- `ghee.Scene / Pass / Param / ArrayInput / Channel`: build the scene
- `ghee.Arg`, `ghee.args([...])`: argument declarations, and the probe answer
- `@ghee.feed("name")`: the values of a named array, for each tick
- `@ghee.tick`: a hook for each tick, with `ghee.set_param(name, value)` and
  `ghee.skip()`
- `ghee.run(scene)`: the handshake, then the tick loop

## Future

A feeder is a process that feeds data alone.
A feeder attaches to a plain JSON scene with `--feed name=cmd:...`.
Audio input becomes a feeder like any other.
This version has no feeder, and the word names nothing else.
