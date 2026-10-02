# The console HTTP API

`ghee serve` and `--serve` serve this API on `127.0.0.1:8090`, and
`ghee.open(stats=8090)` from Python serves the same API. The web console
runs on this API, and your scripts can call it the same way. Anyone who
reaches the port can change params and post GLSL, so keep the port on
loopback, or open it with `--host` only on a network you trust.

## Read

- `GET /stats`: `{fps, frame_ms, stream_fps, out, error, error_line,
  error_pass, active, display, stale}`
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

- `POST /set?pass=&name=&value=[&type=]`: a live uniform, no recompile.
  `pass=scene` sets the scene param, and the string param of a text lays its
  words out again.
- `POST /switch?name=`: switch to a named scene. Switching to the active
  name reloads it from disk.
- `POST /next`, `POST /prev`: step through the scene list.
- `POST /scene`: full scene JSON in the body, recompiles the active scene
  as a working copy.
- `POST /save`: `{name, scene}` in the body, writes the scene into
  `<scenes>/modified/`. A JSON scene saves a copy that shadows the
  original, a procedural scene saves its params and filters as a
  `.values.json` sidecar. A read-only host refuses the save.

## Streams

- `GET /stream?vid=`: MJPEG, one part per rendered frame.
- `GET /h264?vid=`: framed H264 for WebCodecs, `[u8 type][u32 len]` with
  type 0 = config JSON, 1 = key AU, 2 = delta AU, AVCC inside. One shared
  encoder feeds every viewer, and it runs only while a viewer watches.
- `POST /ping?vid=`: heartbeat for the streams opened under that vid.

For example, set the scene param `gain` to `0.8` and read back the
result:

```
curl -X POST 'http://127.0.0.1:8090/set?pass=scene&name=gain&value=0.8'
curl -o out.png http://127.0.0.1:8090/frame.png
```
