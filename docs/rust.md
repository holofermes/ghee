# From Rust

ghee is a library and a command. A render takes three calls: load or build a
scene, set it up, then render frames.

```rust
use ghee::engine::{Feed, Renderer};
use ghee::scene::{ArraySize, Scene};

let scene = Scene::load("bars.scene.json")?;
// width, height, components, DRI device (None probes), scanout, rgb565
let mut r = Renderer::new(320, 240, 4, None, false, false)?;
assert_eq!(r.setup(&scene)?, "");

let frame = r.render(Some(&mut |_name, size| match size {
    ArraySize::Linear(n) => Feed::Data(vec![0.5; n as usize]),
    ArraySize::Grid(w, h) => Feed::Data(vec![0.5; (w * h) as usize]),
}))?;
```

`render` takes one closure, which answers with the data for a named array. The
engine is thread-affine: build and use a Renderer on one thread.

A running scene grows with no rebuild. `add_layer` puts a layer, written as a
scene declares it, into a pass at an index among its layers, and every other
layer and pass keeps what it holds, its phases and its arrays included. A
shape compiles nothing, and a layer whose shaders come out the same as
another layer's draws through the same program:

```rust
r.add_layer("main", 0, &serde_json::json!(
    {"name": "bar", "shape": "rect", "x": 0, "y": 28, "width": 64, "height": 4}))?;
```

`remove_layer` takes a layer out the same way, and `move_layer` puts one at
another index among the layers of its pass, where it keeps its look and its
phases and compiles nothing:

```rust
r.move_layer("bar", 1)?;
r.remove_layer("bar")?;
```
