# HTTP API

Use the console's HTTP API to drive a running ghee from another program:
change params and layers at runtime, switch scenes, and read a frame. The
[console](console.md) page runs on the same API, so a script can do
anything that the page does.

For example, start ghee with its bundled examples:

```shell
ghee serve --demo
```

It opens the first example, `assets`. From another terminal, set its param
`tile` to 3, change the words of its text layer `title`, and save the frame
as a PNG. Each `/set` answers `ok`, and the frame shows both changes:

```shell
curl -X POST 'http://127.0.0.1:8090/set?name=tile&value=3'
curl -X POST 'http://127.0.0.1:8090/set?layer=title&field=text&value=hello'
curl -o frame.png http://127.0.0.1:8090/frame.png
```

`ghee render --serve` and Python's `ghee.open(serve=8090)` serve the same
API. To pick the port and the address, pass `--port` and `--host`, and read
[From another computer](console.md#from-another-computer) before you open
it to a network.

## Requests

Read with GET, and change with POST. Every argument goes in the query
string. `POST /scene` and `POST /upload` also take a body.

To send a value with `#`, spaces, quotes or brackets, let curl encode it:
`-G --data-urlencode` puts each argument in the query string, and `-X POST`
keeps the method. This gives the text layer `title` a gradient:

```shell
curl -G -X POST http://127.0.0.1:8090/set --data-urlencode layer=title \
  --data-urlencode field=color --data-urlencode 'value={"from":"red","to":"yellow","angle":90}'
```

An error is a status and one line of text, such as:
```
bad /set: param 'nope' is not a param this scene declares (main.tile, tile, main.berry, berry, ...)
```

| Status | When |
| --- | --- |
| 304 | a GET whose `If-None-Match` names the current `ETag` ([Revisions](#revisions)) |
| 400 | a bad query or value |
| 403 | a change to a param or a layer key that the scene's program sets ([Locked](#locked)), a change that a [Python console](#from-python) refuses, a `Host` header other than `localhost`, a `127.x.x.x` address, `[::1]` or a name passed to `--allow-hosts`, with or without a port ([From another computer](console.md#from-another-computer)), or a POST from another site's web page |
| 404 | an unknown path, scene or item |
| 405 | a method that the path doesn't take, such as a GET to a path that changes something. `Allow` names the methods that the path takes |
| 409 | an edit for a scene other than the open one, or a scene name that's taken |
| 412 | the scene changed since the `ETag` in your `If-Match` ([Revisions](#revisions)) |
| 413 | a body over 1 MiB, or an upload over 32 MiB |
| 501 | a method that HTTP doesn't define, such as `BREW` |
| 503 | the scene is still loading ([Scenes](#scenes)). `Retry-After` gives the seconds to wait before you ask again |

## Changing a param

To change a param at runtime, POST `/set` with its `name` and `value`:

```shell
curl -X POST 'http://127.0.0.1:8090/set?name=tile&value=3'
```

ghee takes the scene's param of that name, else the param of the first pass
that declares it. To pick a pass, name it in `name`, as in
`/set?name=main.tile&value=3`, or add `pass=`, as in
`/set?pass=main&name=tile&value=3`.

| Param type | `value` |
| --- | --- |
| `float` | a number |
| `int` | a whole number |
| `bool` | `true`, `false`, `1` or `0` |
| `color` | a CSS hex colour such as `#f80` or `#ff8800`, a CSS colour name, or 3 numbers from 0 to 1, comma-separated |
| `vec2`, `vec3`, `vec4` | 2, 3 or 4 numbers, comma-separated |
| `string` | the text |

## Changing a layer

To change a layer at runtime, POST `/set` with its `layer`, the `field`,
and the `value`:

```shell
curl -X POST 'http://127.0.0.1:8090/set?layer=title&field=text&value=hello'
```

To change a key of one of the layer's effects, add `effect=`, as in
`/set?layer=title&effect=karaoke&field=sung&value=12`. A rect's or a
circle's border takes `effect=border`.

The keys that change at runtime, which `GET /format` lists for each kind
of layer ([Changing a layer at
runtime](layers.md#changing-a-layer-at-runtime)), take these values:

| Key | `value` |
| --- | --- |
| `text` | the words |
| `visible` | `true` or `false` |
| `x`, `y`, `opacity`, and a shape's sizes | a number |
| `color` | a colour, written as for a param, or a text's gradient as JSON, such as `{"from":"red","to":"yellow","angle":90}` |
| `image` | the name of an [image asset](layers.md#assets), or a `.png` or `.jpg` file in the scene's folder or in `$GHEE_ASSETS`, such as a file that `/upload` added |
| `points` | JSON, `[[x, y], ...]` |

To put `x`, `y`, `color`, a rect's `width` or `height`, or a circle's
`radius` back to its default, pass `value=null`.

Any other key takes a `/set` too, such as `font_size` or a whole effect, and
ghee builds the layer again. Its `value` is JSON, such as `14` or
`{"strength":0.5}`, or else text, such as `Pixel Operator 8`. `null` puts
the key back to its default.

### Locked

While a [procedural scene](procedural.md)'s program, or your Python
program, sets a param or a layer key, a `/set` of it answers 403, and so
does a `POST /scene` that gives it another value. `GET /stats` lists it
under [`locked`](#stats).

## Revisions

If your program shows the scene and edits it, as the console page does,
send each edit with an `If-Match` header that holds the `ETag` of the
scene that it last read, from `GET /scene`. ghee then refuses an edit with
412 when the scene changed since, so your program never overwrites a change
that it hasn't seen. Each edit answers the scene's new `ETag`: send your
next edit with that one. A change that the scene's program makes gives the
scene a new `ETag` too.

```shell
curl -X POST -H 'If-Match: "5"' 'http://127.0.0.1:8090/set?name=tile&value=3'
```

`GET /scene` answers the scene with every edit that answered before it. To
read the scene only when it changed, send `If-None-Match` with the `ETag`
that you hold: ghee answers 304, with no body, while it is current.

`/set`, `POST /scene`, `/save`, `/reset`, `/aside` and `/draft` edit the
open scene. To name it, pass `scene=` to `/set`, or `name=` to the others: a
name other than the open scene's answers 409. `/new`, `/rename`, `/delete` and
`/upload` work on any of your scenes, open or not.

## Reading

To read the open scene's state, GET `/stats`. Its keys are under
[Stats](#stats):

```shell
curl http://127.0.0.1:8090/stats
```

```
{"frame_ms":3.35,"error":"","text_errors":[],"display":{"w":320,"h":240,"format":"rgba","rotation":0},"error_line":null, ...}
```

| GET | What it answers |
| --- | --- |
| `/stats` | the state of the open scene, as JSON ([Stats](#stats)) |
| `/stats?usage=1` | the same, plus the memory and the GPU time that ghee uses |
| `/scenes` | `scenes`, every scene's name. `active`, the open scene. `procedural`, the ones that a program writes. `mine`, your own, and `my_scenes`, their folder ([Your scenes](console.md#your-scenes)) |
| `/scene?name=` | the scene as it runs, as JSON, with its revision in the `ETag` header. `defaults=1` gives it without your changes. With no name, the open scene |
| `/edits?name=` | your changes to the scene, as a save writes them. `unsaved` says whether they differ from what's saved, `aside` lists the changes waiting for an answer, `broken` the saved changes that no longer build, and `draft` says whether unsaved changes from before ghee stopped wait for an answer ([Saving](console.md#saving)) |
| `/frame.png` | a PNG of the next frame, at the render size, with every `/set` that answered before it. 409 while the scene fails to compile |
| `/format` | what a layer takes: each kind's keys, the keys that change at runtime, each effect's keys and defaults, and the fonts that ghee carries |
| `/filters` | the filters, by name |
| `/hwinfo` | the machine: `model`, `gpu`, `ram`, `ram_free`, and on a Pi `gpu_mem` and `temp` |

### Stats

| Key | What it holds |
| --- | --- |
| `active`, `rev` | the open scene and its revision |
| `fps`, `frame_ms` | frames a second, and the milliseconds of a frame |
| `frames`, `skipped` | the count of frames drawn, and of frames skipped |
| `out` | the outputs that take frames now, such as `kms` and `mjpeg` |
| `display` | the frame's `w` and `h` in pixels, its `format`, and its `rotation` in degrees |
| `error` | the compile error, if any. ghee keeps drawing the last code that compiled |
| `error_pass`, `error_line` | the index of the pass that holds the error, or `"common"`, and its line in that GLSL |
| `text_errors` | one line for each text layer's problem |
| `stale` | `true` when the scene's files changed since it loaded |
| `layer_changes` | how many times the scene's program changed its layers. Read `/scene` again when it moves |
| `locked` | what the scene's program sets: `params` and `fields`. In a [Python console](#from-python), `scene` is `true` too: the program owns the passes, the params' settings and the assets |
| `gpu` | `trilinear_npot`: `false` where a `trilinear` pass needs power-of-two sides |

`?usage=1` adds `rss_mib`, `gpu_mib`, `gpu_busy`, `machine_free_mib` and
`soc_celsius`. Each is `null` where the machine doesn't report it, such as
the GPU on a Pi Zero to Pi 3.

## Scenes

| POST | What it does |
| --- | --- |
| `/switch?name=` | opens the scene, with its time from 0. Answers `{"active": "assets"}` |
| `/next`, `/prev` | opens the next or the previous scene in the list |
| `/scene?name=` | applies the scene JSON in the body, and answers the scene as it runs. Its time keeps running |
| `/save?name=` | saves your changes ([Saving](console.md#saving)) |
| `/reset?name=` | takes the scene back to its files. What you saved stays saved until the next save |
| `/aside?name=&item=&answer=` | answers a change set aside, named as `/edits` lists it: `answer=keep` or `answer=default`. `item=edits` answers the saved changes that no longer build |
| `/draft?name=&answer=` | answers the unsaved changes from before ghee stopped: `answer=restore` puts them back, and `answer=drop` deletes them. 404 when none wait |

`/switch` answers as soon as the scene starts to load, and an edit before
it has loaded answers 503. To wait for the load, GET `/scene`, which
answers once the scene has loaded, within 35 seconds:

```shell
curl -X POST 'http://127.0.0.1:8090/switch?name=bars'
curl -s -o /dev/null 'http://127.0.0.1:8090/scene'
curl -X POST 'http://127.0.0.1:8090/set?name=gain&value=0.8'
```

## Your scenes

These change only the scenes in your own folder. Any other scene answers
403.

| POST | What it does |
| --- | --- |
| `/new?name=` | makes a blank scene, and leaves the open scene open. `from=` makes a copy of a scene, your changes included |
| `/rename?name=&to=` | renames a scene |
| `/delete?name=` | deletes a scene. When it was open, the console opens the first scene left |
| `/upload?name=&file=` | puts the body in the scene's folder as `file`: a `.png`, `.jpg`, `.ttf` or `.otf` file of up to 32 MiB. Answers `{"file": "dot.png"}`, with `-2` added before the extension when another file has that name |

A scene name takes letters, digits, `-` and `_`, starts with a letter or a
digit, and has 64 characters at most. A name that's taken answers 409.

This makes a blank scene `glow`, and adds an image to its folder:

```shell
curl -X POST 'http://127.0.0.1:8090/new?name=glow'
curl -X POST --data-binary @dot.png 'http://127.0.0.1:8090/upload?name=glow&file=dot.png'
```

## Streams

| Request | What it gives |
| --- | --- |
| `GET /stream?vid=` | an MJPEG stream, one JPEG for each frame |
| `GET /h264?vid=` | H264 for a browser's WebCodecs decoder, in the frames below |
| `POST /ping?vid=` | keeps your streams of that `vid` open. Send it every 3 seconds: ghee drops a viewer that sent none for 10 seconds |

Each frame of `/h264` is 1 byte of type, a 4-byte big-endian length, and
the payload:

| Type | Payload |
| --- | --- |
| 0 | the decoder's config, as JSON |
| 1 | a key frame, in AVCC |
| 2 | a delta frame, in AVCC |
| 3 | the reason that the stream ends, as UTF-8 text |

When no encoder can start, `/h264` answers 503.

## From Python

A console that `ghee.open(serve=8090)` serves shows your program's one
scene ([Python](python.md#the-console-from-python)). Two requests change
it, and anything else answers 403:

- `/set`.
- `POST /scene` with the running scene's layers changed: GET `/scene`,
  change its layers, and post it back. A posted scene that changes the
  passes, the params' settings or the assets answers 403, and its param
  values are ignored.
