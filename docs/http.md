# The console HTTP API

`ghee serve` and `--serve` serve this API on `127.0.0.1:8090`, and
`ghee.open(serve=8090)` serves the same API from Python. The web console is
built on this API, and a script can call it the same way. Anyone who reaches
the port can change params and post GLSL, so keep the port on loopback, or
open it with `--host` only on a network you trust.

## Read

- `GET /stats`: `{fps, frame_ms, stream_fps, out, error, error_line,
  error_pass, text_errors, active, display, stale, layer_changes}`. On a
  compile error, `error_pass` is the index of the pass that holds the error,
  or `"common"` for the common block, and `error_line` counts from the first
  line of that GLSL. `text_errors` lists the problems of the scene's text
  layers, one line each, such as `layer 'title': 'X' has more than 64 curves
  in one band`. A text problem leaves the scene running. `layer_changes`
  counts the changes that the program behind the scene made to its layers: a
  drawing that added a layer or gave one a new look, or a field that a tick
  or the Python host set. A client that shows the layers reads `GET /scene`
  again when the count moves.
- `GET /scenes`: `{scenes: [name...], active, procedural: [name...]}`.
  `procedural` names the scenes that a program writes, and is an empty list
  when there are none. A program keeps which layers its scene has and their
  order. `ghee.open(serve=)` lists its scene there, since the Python host
  writes it.
- `GET /scene?name=`: the resolved scene JSON, with shader files inlined and
  filters expanded. For the active scene, ghee answers with the document that
  it renders: applied edits, `/set` values, and the layers that a drawing
  added and changed. For any other name, ghee reads the scene from disk.
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

- `POST /set?scene=&pass=&name=&value=`: sets a live param, with no recompile.
  `pass=scene` sets the scene param, and `pass=0` and up set the param of that
  pass. A text layer's words are the scene param of its name:
  `pass=scene&name=title` sets the words of the layer `title`, and ghee lays
  them out again. `value` reads as the param's declared type: a number, `true`
  or `false`, `#rrggbb`, a CSS colour name or comma-separated floats for a
  color, comma-separated floats for a vec, or the words of a string. A color
  keeps the form that `/set` gives it in `GET /scene` and in a save.
- `POST /set?scene=&layer=&effect=&field=&value=`: sets a live field of a
  layer, with no rebuild, and `effect=` names one of the layer's effects for a
  field of that effect. The live fields are `x`, `y`, `color`, `opacity` and
  `visible`, a text layer's `text`, an image layer's `image`, an effect's
  `visible`, `color`, `strength`, `sung`, `speed` and `rate`, and every field
  of a shape layer but `shape` and `blend`. A rect's border takes an effect's
  place, `effect=border` with `width`, `color` or `visible`, and a set gives a
  rect with no border its border. `text` sets the words, as the scene param of
  the layer's name does. Any other field answers 400 and names the live
  fields, and so does an unknown layer, effect or field. `value` reads as the
  field takes it: the words for `text`, a number, `true` or `false`,
  `#rrggbb`, a CSS colour name or comma-separated floats for a color, a JSON
  object `{"from", "to", "angle"}` for a text layer's gradient, the name of an
  image asset of the scene or a `.png` or `.jpg` file for `image`, a JSON list
  of `[x, y]` for a line's `points`, or `none`, which takes `x`, `y`, `color`
  or a rect's `width` or `height` back to its default.
- `POST /switch?name=`: switches to a named scene. Switching to the active
  name reloads it from disk.
- `POST /next`, `POST /prev`: step through the scene list.
- `POST /scene?name=`: takes the full scene JSON in the body, and recompiles
  the active scene as a working copy. A procedural scene keeps its layers:
  each layer of the running scene that the posted scene leaves out stays, in
  its place. `ghee.open(serve=)` compiles no scene: there `POST /scene` gives
  each layer of the running scene the look that the posted scene gives it,
  and answers 403 to a posted scene that differs in anything but its layers
  and its param values.
- `POST /save`: takes `{name, scene}` in the body, and writes the scene into
  `<scenes>/modified/`. A JSON scene saves a copy that shadows the original
  and finds its files beside the original. A procedural scene saves a
  `.values.json` sidecar: `scene` and `passes` hold the param values, the
  words of each text layer among them, `filters` the filter list, `visible`
  each pass's eye by the pass's name, and `layers` the settings and effects
  of each layer by the layer's name. At its next start the sidecar applies
  over the scene that the process writes: the process keeps which layers
  exist and their order, a saved layer that it no longer declares is skipped
  with a log line, and a value that the process sets after the start wins
  over the saved one. A read-only host refuses the save.

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

For example, set the scene param `gain` to `0.8`, sing the first 12
characters of the layer `title`, and read back the result:

```
curl -X POST 'http://127.0.0.1:8090/set?pass=scene&name=gain&value=0.8'
curl -X POST 'http://127.0.0.1:8090/set?layer=title&effect=karaoke&field=sung&value=12'
curl -o out.png http://127.0.0.1:8090/frame.png
```
