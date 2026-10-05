# The console HTTP API

`ghee serve` and `--serve` serve this API on `127.0.0.1:8090`, and
`ghee.open(stats=8090)` serves the same API from Python. The web console is
built on this API, and a script can call it the same way. Anyone who reaches
the port can change params and post GLSL, so keep the port on loopback, or
open it with `--host` only on a network you trust.

## Read

- `GET /stats`: `{fps, frame_ms, stream_fps, out, error, error_line,
  error_pass, error_layer, text_errors, active, display, stale}`. On a compile
  error, `error_pass` is the index of the pass that holds the error, or
  `"common"` for the common block. `error_layer` is `{index, part}` when one
  of that pass's layers holds the error, with `part` set to `"vertex"` or
  `"fragment"`. `error_line` counts from the first line of the GLSL that holds
  the error. `text_errors` lists the problems of the scene's texts, one line
  each. A text problem leaves the scene running.
- `GET /scenes`: `{scenes: [name...], active}`
- `GET /scene?name=`: the resolved scene JSON, with shader files inlined and
  filters expanded. For the active scene, ghee answers with the document that
  it renders, applied edits and `/set` values included. For any other name,
  ghee reads the scene from disk.
- `GET /filters`: the filter registry, `{name: filter def}`
- `GET /hwinfo`: the host hardware report
- `GET /frame.png`: one lossless PNG of the next frame, at the render size.
  The request goes on the command queue, so the frame carries every `/set`
  that returned before it. It answers 409 while the scene holds an error, and
  503 when the loop draws nothing.

## Change

The active scene is the one that the last switch asked for. An edit names the
scene that it edits: `/set` in `scene=`, `POST /scene` in `name=`, and
`/save` in its body. An edit for a scene other than the active one answers
409 and changes nothing. `/set` and `POST /scene` without a name edit the
active scene. The console names its scene in each request, so a page left
open on one scene never writes into a scene that the console switched to
later.

- `POST /set?scene=&pass=&name=&value=`: sets a live uniform, with no
  recompile. `pass=scene` sets the scene param, and `pass=0` and up set the
  param of that pass. The name of a text sets its words, and ghee lays them
  out again. `value` reads as the param's declared type: a number, `true` or
  `false`, `#rrggbb` or comma-separated floats for a color, comma-separated
  floats for a vec, or the words of a string. A color keeps the form that
  `/set` gives it in `GET /scene` and in a save.
- `POST /switch?name=`: switches to a named scene. Switching to the active
  name reloads it from disk.
- `POST /next`, `POST /prev`: step through the scene list.
- `POST /scene?name=`: takes the full scene JSON in the body, and recompiles
  the active scene as a working copy.
- `POST /save`: takes `{name, scene}` in the body, and writes the scene into
  `<scenes>/modified/`. A JSON scene saves a copy that shadows the original
  and finds its files beside the original. A procedural scene saves its
  params and filters as a `.values.json` sidecar. A read-only host refuses
  the save.

## Streams

- `GET /stream?vid=`: MJPEG, one part per rendered frame.
- `GET /h264?vid=`: framed H264 for WebCodecs, `[u8 type][u32 len]`, with
  type 0 = config JSON, 1 = key AU, 2 = delta AU, AVCC inside, and 3 = the
  UTF-8 reason that the stream ends. One shared encoder feeds every viewer,
  and it runs only while a viewer watches. The status line waits for the
  first frame, so when no encoder can start, the stream answers 503 with the
  reason as the body. An encoder that fails after the first frame ends the
  stream with a type 3 frame.
- `POST /ping?vid=`: a heartbeat for the streams opened under that vid.

For example, set the scene param `gain` to `0.8` and read back the
result:

```
curl -X POST 'http://127.0.0.1:8090/set?pass=scene&name=gain&value=0.8'
curl -o out.png http://127.0.0.1:8090/frame.png
```
