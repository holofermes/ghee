# The console

`ghee serve <scenes_dir>` starts the web console on `127.0.0.1:8090`. The
console shows a hardware accelerated live preview of the running scene. It
edits each pass's layers and their effects, and the GLSL of each pass, with
compile errors inline. It lists a pass's layers front first: the last one in
the scene's `layers` paints on top and shows at the top. It shows each param
as a slider, hides passes, layers and effects, and picks filters. Its render
info shows the frame rate, and while it is open the memory and the GPU time
that ghee uses, where the kernel reports them. [docs/http.md](http.md)
holds its API.

An edit goes into the working copy of the scene that ghee holds, which every
open page shows. A live field applies as it changes, and every other edit
reloads into the running renderer. The working copy lasts until ghee stops,
and Save keeps its edits in the scene's [state file](#saved-edits).

`--fps N` caps the frame rate of `ghee serve` and of `ghee render --serve`.

The console draws while someone watches it: a page's preview, a viewer of
`/stream` or `/h264`, or a `GET /frame.png`. A console that drives a display
or a panel draws every frame. While nobody watches, the console draws once
for each edit and sleeps in between, and the scene's time keeps to the wall
clock, so the next frame draws at the time that the clock reached. With
`--timestep`, the scene's time keeps its fixed step per frame, however long
the console slept.

The console reads files from its scenes directory, the `$GHEE_ASSETS`
directories and the system's emoji font alone, so a scene that it serves
fails to load a path that is absolute or climbs out with `..`.

![the console editing a live scene](https://raw.githubusercontent.com/holofermes/ghee/main/media/console.png)

Careful: the console has no auth, and it compiles any GLSL posted to it, so it
binds to loopback by default. `--host 0.0.0.0` opens it to your network. Do
that only on a network you trust.

## Saved edits

Save writes the console's edits of a scene, what differs from the scene's
files, to the scene's state file. ghee lays the state file over the scene's
files each time that `ghee serve` or `ghee render --serve` loads the scene. A
new version of the scene's files leaves the state file alone: a saved value
stays, and every other setting follows the new files. A plain `ghee render`
reads no state file, and a console hosted by Python saves nothing.

The state files live in `$XDG_STATE_HOME/ghee/`, `~/.local/state/ghee/` when
the variable is unset, and `--state-dir DIR` puts them in DIR. A scene's state
file takes the scene file's absolute path under `scenes/`, with `.json` in
place of `.scene.json` or of a procedural scene's extension, and a bundled
demo's sits under `demo/`:

```
/srv/scenes/clocks/clocks.scene.json
  ~/.local/state/ghee/scenes/srv/scenes/clocks/clocks.json
ghee serve --demo, the scene wavy
  ~/.local/state/ghee/demo/wavy.json
```

This state file sets two params of the pass `main` and holds an edit of its
GLSL:

```json
{
  "version": 1,
  "scene": "/srv/scenes/clocks/clocks.scene.json",
  "values": { "passes": { "main": { "params": { "cols": 5, "rows": 4 } } } },
  "items": [
    { "item": "fragment", "pass": "main", "base": "4be1a2c97f3d0e55",
      "value": "void main() { ... }" }
  ]
}
```

`values` is a JSON merge patch over the scene by name: `params` holds the
scene params' values, `passes` each pass's `params`, `visible`, `size`,
`scale` and `mipmaps`, and `layers` each layer's settings and effects, as a
layer's JSON holds them. `null` removes a key. `items` holds each edit of code
or structure whole, with `base`, the hash of the same item in the files that it
was made against: a pass's `fragment`, `channels` and `layers`, the `common`
block, the `passes`, an added `param`, and the `filters`.

A value whose pass, layer or param is gone is skipped, with one line in the
log, and the next save drops it. An item whose `base` no longer matches the
files is set aside: the files' version runs, and the console asks once, keep
mine or update to default. A state file that ghee cannot read, such as one
with broken JSON, moves to `clocks.json.unreadable` beside it, with one line
in the log, and the scene loads without its edits.
