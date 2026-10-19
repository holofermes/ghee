# From Rust

ghee is a library as well as a command. Use it to render a scene into a
buffer of your own, or to run the console on a display from your own program.

## Depend on it

Add ghee to your `Cargo.toml` as a git dependency, with the two crates that
the examples here use:

```toml
[dependencies]
ghee = { git = "https://github.com/holofermes/ghee" }
anyhow = "1"
serde_json = "1"
```

To pin a release, add `tag = "vX.Y.Z"` with a version from the [releases
page](https://github.com/holofermes/ghee/releases). The build needs the
GLES and DRM headers that [Building](building.md) lists. To build for a Pi
from another computer, see [Cross compile for the
Pi](building.md#cross-compile-for-the-pi).

To read the documentation of every type and function, run `cargo doc -p
ghee --open` in your project.

## Render frames into a buffer

This program loads a scene, sets it up and renders one frame, RGBA at
320x240. The closure that `render` takes gives the data of each array that
the scene declares, by its name and size:

```rust,no_run
use ghee::engine::{Feed, Renderer};
use ghee::scene::{ArraySize, Scene};

fn main() -> anyhow::Result<()> {
    let scene = Scene::load("bars.scene.json")?;
    let mut r = Renderer::builder(320, 240).build()?;
    r.setup(&scene)?;
    let frame = r.render(Some(&mut |_name, size| match size {
        ArraySize::Linear(n) => Feed::Data(vec![0.5; n as usize]),
        ArraySize::Grid(w, h) => Feed::Data(vec![0.5; (w * h) as usize]),
    }))?;
    assert_eq!(frame.len(), 320 * 240 * 4);
    Ok(())
}
```

To find where a scene fails, match the `SetupError` that `setup` returns.
`SetupError::Compile` carries the pass and the line of your GLSL that the GPU
refused, and `SetupError::Scene` says what the scene asks for that the
renderer can't hold. After a failed `setup`, the scene before it keeps
drawing. Create and use a `Renderer` on one thread: GL is thread-affine.

`render` returns the frame's pixels row by row, the top row first. To read
the frames back as RGB, call `.components(3)` on the builder before `build`.
To render on one GPU of several, pass its node to `.device`, such as
`.device("/dev/dri/renderD129")`. To scan the frames out to a display, pass
its `Card` to `.display` ([Run the console on a
display](#run-the-console-on-a-display) opens one).

Give an array's data as `Feed::Data` for floats, or as `Feed::DataU8` for
bytes, each one read as byte / 255. Give `Feed::Unchanged` when the array
holds what it held last frame, and ghee skips the upload. After every
`setup`, give the data again.

## Change layers at runtime

A running scene takes new layers with no rebuild. Use `add_layer` to put a
layer, written as a scene declares it ([Layers](layers.md)), into a pass at
an index among its layers. Every other layer and pass keeps what it holds,
its phases and its arrays included. Use `move_layer` to put a layer at
another index of its pass, and `remove_layer` to take it out:

```rust,no_run
use ghee::engine::Renderer;
use ghee::scene::Scene;
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let mut r = Renderer::builder(128, 32).build()?;
    r.setup(&Scene::load("bars.scene.json")?)?;
    let bar = json!({"name": "bar", "shape": "rect", "x": 0, "y": 28, "width": 64, "height": 4});
    r.add_layer("main", 0, &bar)?;
    r.move_layer("bar", 1)?;
    r.remove_layer("bar")?;
    Ok(())
}
```

## Run the console on a display

This program runs the [console](console.md) on the display that the kernel
drives, over the scenes in `scenes/`, and feeds each scene's array `level`
from its own source. It draws at most 30 frames a second. Ctrl-C or SIGTERM
stops it and turns the display off, and the program then ends by that
signal, as `ghee` does:

```rust,no_run
use ghee::console::server::{self, ArraySource, Options};
use ghee::control::{self, Control};
use ghee::engine::{Card, Feed};
use ghee::scene::ArraySize;

fn main() -> anyhow::Result<()> {
    let level: ArraySource = Box::new(|size, frame| match size {
        ArraySize::Linear(n) => {
            Feed::Data((0..n).map(|i| ((frame as i64 + i) % 60) as f32 / 60.0).collect())
        }
        ArraySize::Grid(w, h) => Feed::DataU8(vec![255; (w * h) as usize]),
    });
    let stop = Control::new();
    control::stop_on_signals(&stop)?;
    server::run(Options {
        scenes_dir: Some("scenes".into()),
        display: Some(Card::open(None)?),
        arrays: vec![("level".into(), level)],
        fps: Some(30.0),
        control: stop.clone(),
        ..Options::default()
    })?;
    stop.reraise();
    Ok(())
}
```

`run` serves the console at http://127.0.0.1:8090 and draws every frame on
the thread that calls it. Every field of `Options` has a default, so set
only the ones you need. To reach the console from another machine, set
`host`, and read [From another computer](console.md#from-another-computer)
first.

`Card::open(None)` opens the first card with a connected display, and the
render takes the display's mode as its size. To pick a card, pass its node,
such as `Some("/dev/dri/card1")`. Its `CardError` says why it failed: a
node that won't open, no connected display, or GL that runs in software.
Leave `display` at None to render with no display, at `size` or 320x240.

An `ArraySource` gives the data of one array from the array's size and the
frame number, in a `Feed` as `render`'s closure does. An array with no
source gets ghee's placeholder data.

Set `fps` to cap the frame rate of the console and its display. With None,
the display's refresh paces the render.

Set `filters` to end every scene with the same [filters](filters.md), such
as `vec![Filter::named("frc")]` with `ghee::filters::Filter`. To start one
at settings of your own, make it with `Filter::from_json`, as in
`Filter::from_json(&json!({"name": "frc", "calm": 8}))?`. A filter that a
scene lists too keeps its place in the scene and takes your settings.

To find the scenes' files in folders of your own, call
`ghee::scene::set_asset_dirs` before `run`. ghee searches them after the
scene's own folder and before `$GHEE_ASSETS`.

A scene that a program writes ([Procedural scenes](procedural.md)) runs
under the console: list it in `scenes_dir`. A `Renderer` alone draws scene
files.

## Put the frames anywhere

This program hands every frame to a sink of its own, which keeps the newest
frame for another thread:

```rust,no_run
use std::sync::{Arc, Mutex};

use ghee::console::server::{self, Options};
use ghee::looper::{Frame, Output};

struct Latest(Arc<Mutex<Vec<u8>>>);

impl Output for Latest {
    fn write(&mut self, frame: Frame<'_>) -> anyhow::Result<()> {
        let mut held = self.0.lock().unwrap();
        held.clear();
        held.extend_from_slice(frame.data);
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let latest = Arc::new(Mutex::new(Vec::new()));
    server::run(Options {
        scenes_dir: Some("scenes".into()),
        sinks: vec![Box::new(Latest(Arc::clone(&latest)))],
        ..Options::default()
    })
}
```

`frame.data` holds the pixels row by row, the top row first, `frame.width`
by `frame.height`, `frame.components` bytes each. ghee calls `write` on the
thread that runs the console. Implement `persists` to return true when your
sink keeps showing its last frame, and ghee then skips a frame that would
come out the same. Implement `turn_on` and `turn_off` for a sink with a
power state, such as a backlight: ghee turns it on before the first frame
and before the first frame after each pause, and off while the render
pauses and when it ends.

## Stop and pause

This program pauses the console every ten seconds, and lets it go on ten
seconds later, from a thread of its own:

```rust,no_run
use std::thread;
use std::time::Duration;

use ghee::console::server::{self, Options};
use ghee::control::Control;

fn main() -> anyhow::Result<()> {
    let control = Control::new();
    let pauser = control.clone();
    thread::spawn(move || {
        while !pauser.stopped() {
            thread::sleep(Duration::from_secs(10));
            pauser.pause(!pauser.paused());
        }
    });
    server::run(Options {
        scenes_dir: Some("scenes".into()),
        control,
        ..Options::default()
    })
}
```

A paused console blanks its display, turns your sinks off and draws
nothing. It still takes requests, and they land on the first frame after
the pause. Use `pause` to rest the display while your program has nothing
to show.

`run` returns when its `Control` stops. Keep a clone of the `Control` that
you pass in `Options`, and call `stop` on it from any thread.
`stop_on_signals` stops it on SIGINT and SIGTERM, and `signal` then names
the signal, such as 15 for SIGTERM. When `run` returns, SIGINT and SIGTERM
go back to the handlers that they had before `stop_on_signals`, so a second
Ctrl-C ends your program as usual. To end your program by the signal that
stopped the console, as `ghee` does, call `reraise` on the `Control` once
`run` returns. A stopped `Control` stays stopped, so make a new one for each
run.

Careful: `stop` takes a lock. Never call it from a signal handler of your
own: use `stop_on_signals`.
