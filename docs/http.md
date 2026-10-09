# The console HTTP API

`ghee serve` and `--serve` serve this API on `127.0.0.1:8090`, and
`ghee.open(serve=8090)` serves the same API from Python. The web console is
built on this API, and a script can call it the same way. Anyone who reaches
the port can change params and post GLSL, so keep the port on loopback, or
open it with `--host` only on a network you trust.

ghee holds a working copy of each scene that the console opens: the scene's
defaults, with its saved edits and every edit made since laid over them. The
defaults are the scene's files, or the scene that a procedural scene's program
writes at its start. Every page reads the same working copy, and every edit
applies to it at once. ghee keeps the working copy until it stops, through a
switch away and back. `/save` writes what differs from the defaults to the
scene's state file, as [Saved edits](console.md#saved-edits) describes.

A query with a malformed percent escape answers 400, `bad query: '%zz' is
not a percent escape`. A POST body over 1 MiB answers 413, `body over the
limit of 1048576 bytes`, with its length in `Content-Length` or sent
chunked, and a body whose read fails answers 400, `bad body:` and the
error.

## Read

- `GET /stats`: `{fps, frame_ms, stream_fps, out, error, error_line,
  error_pass, text_errors, active, display, stale, layer_changes, rev,
  locked, gpu}`.
  `active` is the scene of the last switch, even one that failed to compile,
  and `rev` is its revision. While the active scene fails to compile, ghee
  keeps drawing the last program that compiled. On a compile error,
  `error_pass` is the index of the pass that holds the error, or `"common"`
  for the common block, and `error_line` counts from the first line of that
  GLSL. `display` holds the frame as the outputs take it: `w` and `h` in
  pixels, `format`, and `rotation`, the degrees clockwise of `--rotate`.
  `text_errors` lists the problems of the scene's text layers, one
  line each, such as `layer 'title': 'X' has more than 64 curves in one
  band`. A text problem leaves the scene running. `layer_changes` counts the
  changes that the program behind the scene made to its layers: a drawing
  that added a layer or gave one a new look, or a field that a tick or the
  Python host set. A client that shows the layers reads `GET /scene` again
  when the count or `rev` moves. `locked` holds what the program behind
  the active scene set since it started, each in a list: `params`, a scene
  param by its name and a param of a pass as `pass.name`, `fields`, each
  live field of a layer as `layer.field` or `layer.effect.field`, and
  `layers`, each layer that the program added or gave a look. `gpu` holds
  what the GPU can do, and is empty until the renderer starts:
  `mipmaps_npot` is true when the GPU mipmaps a pass of any size, and false
  when a pass with `mipmaps` takes a power-of-two size alone, as on the vc4
  of a Pi 1 to a Pi 3 and a Zero 2 W.
- `GET /stats?usage=1`: the same, with what the ghee process uses, read
  from files that it may read without root, at this request alone:
  - `rss_mib`: the process's resident set, in MiB.
  - `gpu_mib`: the GPU buffers that the process holds, in MiB, from the
    `drm-total` keys in the fdinfo of its DRM descriptors. On a Pi 4 and a
    Pi 5 those buffers are system RAM outside the resident set.
  - `gpu_busy`: `{queue: percent}`, the share of the time since the last
    request that asked for usage that the process kept each queue of the
    GPU busy, from the `drm-engine` keys of the same fdinfo, for each queue
    that ran. A queue holds null in the first answer.
  - `machine_free_mib`: the machine's free memory, MemAvailable, in MiB.
  - `soc_celsius`: on a Raspberry Pi, the SoC's temperature, from its one
    sensor.

  A figure that the machine does not report is null: `gpu_mib` and
  `gpu_busy` on a Pi 1 to a Pi 3 and a Zero 2 W, whose vc4 driver reports
  nothing per process, and under NVIDIA's driver.
- `GET /scenes`: `{scenes: [name...], active, procedural: [name...]}`.
  `procedural` names the scenes that a program writes, and is an empty list
  when there are none. A program keeps which layers its scene has and their
  order. `ghee.open(serve=)` lists its scene there, since the Python host
  writes it.
- `GET /scene?name=`: the working copy of the scene, resolved, with each
  GLSL file as its text and filters expanded, and the scene's revision in
  the `ETag` header. With no name, it answers the active scene. For a
  procedural scene that runs, the document also holds the layers that its
  program drew and the values that it set. `GET /scene?name=&defaults=1`
  answers the defaults, with no edit. While the active scene loads, `GET
  /scene` and `GET /edits` for it wait up to 35 seconds for the load.
- `GET /edits?name=`: `{rev, values, items, unsaved, aside}`. `values` and
  `items` are the working copy's edits, as a save writes them in the state
  file. `unsaved` says whether they differ from the state file. `aside`
  lists the items set aside, each as `{item, pass, param}` with the keys
  that its kind takes.
- `GET /format`: the layer format that the console edits. `effects` holds
  each effect's fields in order, each as `{key, default, live}`, where
  `live` says that a `/set` changes it with no rebuild. `border` holds a
  rect's border's fields in the same form. `live` holds the live fields of
  each kind of layer, `text`, `image`, `rect` and `line`. `empty` holds what
  a field left out takes, for each kind: `centre`, `fit`, `auto`, `pass` or
  a number. `shapes` maps each shape, `rect` and `line`, to its own keys,
  past `shape` and the keys that every layer takes. `fonts` maps each
  built-in font's name to the `font_family` that picks it, null for the
  default font. `render` gives the render of a text layer that names none:
  `fonts` maps a `font_family` to its render, and every other font takes
  `default`.
- `GET /filters`: the filter registry, `{name: filter def}`
- `GET /hwinfo`: the host's hardware, each fact as text: `model`, the board
  as the device tree names it, or the kernel's name and release on a host
  with no device tree, `gpu`, which reads `n/a` off a Raspberry Pi, `ram`
  and `ram_free`, such as `408 MB`, and on a Raspberry Pi `gpu_mem` and
  `temp`. A fact that the host cannot read is absent.
- `GET /frame.png`: one lossless PNG of the next frame, at the render size.
  The request goes on the command queue, so the frame carries every `/set`
  that returned before it. It answers 409 while the scene holds an error, and
  503 when the loop draws nothing.

## Change

An edit names the scene that it edits: `/set` in `scene=`, every other edit
in `name=`. An edit for a scene other than the active one answers 409 and
changes nothing. An edit without a name edits the active scene. While the
active scene loads, an edit answers 503, `scene 'clocks' is loading`, and
once a load of it fails, 500 with the load's error.

Every change applied through the API gives the scene a new revision: a set,
an apply, a save, a reset, an answer and a switch. An edit takes `rev=`,
the revision of the scene that its page last read, and `vid=`, the page's
id, the one that its streams use. An edit whose scene another client
changed after `rev` answers 412, `this scene changed since you loaded it`,
and changes nothing. A page's own changes never make its `rev` stale, so a
slider's sets, sent before the last one answers, all land. An edit with no
`rev` is never refused this way.

A param, a live field or a layer that `locked` in `GET /stats` lists stays
the program's while it runs: a `/set` of it answers 403, `the scene's
program sets param 'main.amt'`, and an apply and a save leave its value out
of the working copy's edits.

- `POST /set?scene=&pass=&name=&value=`: sets a live param, with no recompile.
  `pass=` names the pass that holds the param. With no `pass=`, `name` is the
  scene param, or the param of the first pass that declares it, as Python's
  `set_param` takes a name. `value` reads as the param's declared type: a
  number, `true` or `false`, `#rrggbb`, a CSS colour name or comma-separated
  floats for a color, comma-separated floats for a vec, or the words of a
  string. A color keeps the form that `/set` gives it in `GET /scene` and in
  a save. A name that no param answers to is a 400 that lists the names that
  do, as Python's `set_param` lists them.
- `POST /set?scene=&layer=&effect=&field=&value=`: sets a live field of a
  layer, with no rebuild, and `effect=` names one of the layer's effects for a
  field of that effect. The live fields are `x`, `y`, `color`, `opacity` and
  `visible`, a text layer's `text`, an image layer's `image`, an effect's
  `visible`, `color`, `strength`, `sung`, `speed` and `rate`, and every field
  of a shape layer but `shape` and `blend`. A rect's border takes an effect's
  place, `effect=border` with `width`, `color` or `visible`, and a set gives a
  rect with no border its border. `text` sets the words, and ghee lays them
  out again. Any other field answers 400 and names the live fields, and so
  does an unknown effect or field. An unknown layer answers 400 and names the
  scene's layers, as Python's `set_layer` names them. `value` reads as the
  field takes it: the words for `text`, a number, `true` or `false`,
  `#rrggbb`, a CSS colour name or comma-separated floats for a color, a JSON
  object `{"from", "to", "angle"}` for a text layer's gradient, the name of an
  image asset of the scene or a `.png` or `.jpg` file for `image`, a JSON list
  of `[x, y]` for a line's `points`, or `none`, which takes `x`, `y`, `color`
  or a rect's `width` or `height` back to its default.
- `POST /switch?name=`: switches to a named scene, and `iFrame` and `iTime`
  start at 0, with `iTimeDelta` at 1/60 on the first frame. Switching to the
  active name reloads its files, keeps the working copy's edits and checks
  its items as a start does. A switch that carries `rev=` is a page's
  reload, and answers 409 when the console is on another scene.
- `POST /next`, `POST /prev`: step through the scene list.
- `POST /scene?name=`: takes the full scene JSON in the body, applies it to
  the working copy, and recompiles. `iFrame` and `iTime` keep running. A
  procedural scene keeps its layers: each layer of the running scene that
  the posted scene leaves out stays, in its place. `ghee.open(serve=)`
  compiles no scene: there `POST /scene` gives each layer of the running
  scene the look that the posted scene gives it, and answers 403 to a posted
  scene that differs in anything but its layers and its param values.
- `POST /save?name=`: writes the working copy's edits to the scene's state
  file, with the items still set aside, or removes the file when there are
  none. `ghee.open(serve=)` answers 403, `this console saves no scene`.
- `POST /reset?name=`: takes the working copy back to the defaults. The
  state file keeps its edits until the next save.
- `POST /aside?name=&item=&pass=&param=&answer=`: answers an item set aside,
  named as `GET /edits` lists it, and writes the state file at once.
  `answer=keep` applies the edit over the new default and takes the default
  item's hash as its base, and `answer=default` removes the item. It answers
  404 when no such item is set aside, as when another page answered first.

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
curl -X POST 'http://127.0.0.1:8090/set?name=gain&value=0.8'
curl -X POST 'http://127.0.0.1:8090/set?layer=title&effect=karaoke&field=sung&value=12'
curl -o out.png http://127.0.0.1:8090/frame.png
```
