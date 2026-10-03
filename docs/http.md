# The console HTTP API

`ghee serve` and `--serve` serve this API on `127.0.0.1:8090`, and
`ghee.open(stats=8090)` from Python serves the same API. The web console
runs on this API, and your scripts can call it the same way. Anyone who
reaches the port can change params and post GLSL, so keep the port on
loopback, or open it with `--host` only on a network you trust.

## Read

- `GET /stats`: `{fps, frame_ms, stream_fps, out, error, error_line,
  error_pass, error_layer, text_errors, active, display, stale}`. On a
  compile error, `error_pass` is the index of the pass that holds it, or
  `"common"` for the common block, and `error_layer` is `{index, part}` when
  one of that pass's layers holds it, with `part` set to `"vertex"` or
  `"fragment"`. `error_line` counts from the first line of the GLSL holding
  the error. `text_errors` lists the problems of the scene's texts, a line
  each, which leave the scene running.
- `GET /scenes`: `{scenes: [name...], active}`
- `GET /scene?name=`: the resolved scene JSON, shader files inlined and
  filters expanded. For the active scene ghee answers with the document it
  is rendering, applied edits and /set values included. For any other name
  it reads from disk.
- `GET /filters`: the filter registry, `{name: filter def}`
- `GET /hwinfo`: the host hardware report
- `GET /frame.png`: one lossless PNG of the next frame, at the render
  size. The request goes on the command queue, so the frame carries every
  `/set` that returned before it. 409 while the scene holds an error, 503
  when the loop draws nothing.

## Change

The active scene is the one that the last switch asked for. `/set` takes
the scene that it edits in `scene=`, `POST /scene` in `name=` and `/save` in
its body, and each answers 409 and changes nothing when that scene is not the
active one. The console names its scene in each request, so a page left open
on one scene never writes into the scene that the console switched to.

- `POST /set?scene=&pass=&name=&value=`: a live uniform, no recompile.
  `pass=scene` sets the scene param, and the name of a text sets its words,
  which ghee lays out again. `value` reads as the param's declared type: a number,
  `true` or `false`, comma-separated floats for a color or a vec, or the
  words of a string.
- `POST /switch?name=`: switch to a named scene. Switching to the active
  name reloads it from disk.
- `POST /next`, `POST /prev`: step through the scene list.
- `POST /scene?name=`: full scene JSON in the body, recompiles the active
  scene as a working copy.
- `POST /save`: `{name, scene}` in the body, writes the scene into
  `<scenes>/modified/`. A JSON scene saves a copy that shadows the
  original and finds its files beside it, a procedural scene saves its
  params and filters as a `.values.json` sidecar. A read-only host refuses
  the save.

## Streams

- `GET /stream?vid=`: MJPEG, one part per rendered frame.
- `GET /h264?vid=`: framed H264 for WebCodecs, `[u8 type][u32 len]` with
  type 0 = config JSON, 1 = key AU, 2 = delta AU, AVCC inside, and 3 = the
  UTF-8 reason the stream ends. One shared encoder feeds every viewer, and
  it runs only while a viewer watches. The status line waits for the first
  frame, so a stream with no encoder that can start answers 503 with the
  reason as the body. An encoder that fails after the first frame ends the
  stream with a type 3 frame.
- `POST /ping?vid=`: heartbeat for the streams opened under that vid.

For example, set the scene param `gain` to `0.8` and read back the
result:

```
curl -X POST 'http://127.0.0.1:8090/set?pass=scene&name=gain&value=0.8'
curl -o out.png http://127.0.0.1:8090/frame.png
```
