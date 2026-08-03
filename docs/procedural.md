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

ghee runs `<scene> --ghee-args` only to build `-- --help`.
The scene writes one JSON line and stops:

    {"args": [{"name": ..., "type": ..., "default": ..., "help": ...}]}

The types are `string`, `int`, `float`, `size` and `bool`.
ghee waits 2 seconds for the line.
A scene that does not answer is listed as a pass-through.
ghee caches no answer.

`ghee render <scene> -- --help` asks that one scene.
`ghee serve -- --help` asks every executable scene of the set.
A render and a serve with any other tokens run no probe.

## Start

ghee runs the scene with the tokens after `--` as arguments.
ghee reads no token and checks none.
A scene that refuses a token exits, and ghee shows its stderr.
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
- kind 4 sets the named array from u8 values. The header is the header of kind
  0. The payload is count u8 bytes, and the value is byte / 255. A large mask
  costs a quarter of the pipe, and the 8-bit texture takes the bytes as they
  lie.

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

A Python scene imports the shim: `import ghee`.
The shim is the ghee-render package, and pip installs it.
Nothing vendors it, and no `ghee.py` ships beside a scene.

    pip install ghee-render

A scene that declares its packages inline (PEP 723) names it there:

    #!/usr/bin/env -S uv run --script
    # /// script
    # dependencies = ["ghee-render"]
    # ///

The shim gives:

- `ghee.Scene / Pass / Param / ArrayInput / Channel`: build the scene
- `ghee.Arg`, `ghee.args([...])`: argument declarations, and the probe answer
- `@ghee.feed("name")`: the values of a named array, for each tick. A uint8
  buffer, such as `bytes` or a numpy array of dtype uint8, sends kind 4
- `@ghee.tick`: a hook for each tick, with `ghee.set_param(name, value)` and
  `ghee.skip()`
- `ghee.run(scene)`: the handshake, then the tick loop

## Direct mode

ghee sets `GHEE_WIRE=1` in the process it starts.
A scene that sees the mark answers the wire above.
A scene a user started sees no mark, so `run()` opens the scene on the ghee
bindings and drives it there:

    ./spectrum.py --output kms --frames 300
    ./spectrum.py --bins 32 --output none --size 320x240 --frames 1

The scene takes its own arguments first, then these:

- `--output NAME`: a driver name, `kms`, or `none` for a headless render
  (default `kms`)
- `--size WxH`: the render size. A display or a panel states its own
- `--fps N`, `--stats PORT`, `--timestep S`: as `ghee render` takes them
- `--frames N`: 0 runs until SIGINT or SIGTERM (default 0)

The hooks and the feeds are the same, and `ghee.skip()` skips the same frame.
The tick keeps its wire shape. `params` stays empty, because no ghee holds the
string params here.
A scene that declares an argument of one of these names collides, and argparse
says so.

## What the wire carries

The wire suits the arrays one frame carries: a spectrum, a mask, a few
thousand values.
Megabyte-scale data costs a serialization and two copies for each frame.
A shared-memory lane may answer that later.
Today a data-heavy Python host takes the bindings instead: `ghee.open` feeds
the renderer in the same process.

## Future

A feeder is a process that feeds data alone.
A feeder attaches to a plain JSON scene with `--feed name=cmd:...`.
Audio input becomes a feeder like any other.
This version has no feeder, and the word names nothing else.
