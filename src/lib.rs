//! ghee compiles GLSL scenes and renders them on a GPU through DRM/GBM,
//! headless.
//!
//! A [`Scene`](scene::Scene) is an ordered list of passes, each one a
//! fragment shader plus the arrays and params it reads. A
//! [`Renderer`](engine::Renderer) compiles a scene and renders frames into
//! a buffer that the host reads back.
//!
//! The `ghee` command is [`cli::main`]. The engine is thread-affine: build
//! and use a Renderer on one thread.
//!
//! The example opens a GL device, so the doc test only compiles it.
//! `tests/lib_api.rs` runs the same scene where a device is present.
//!
//! ```no_run
//! use ghee::engine::{Feed, Renderer};
//! use ghee::scene::{Param, Pass, PassBuilder, Scene, SceneBuilder};
//!
//! # fn main() -> anyhow::Result<()> {
//! let scene = Scene::of(vec![Pass::of(
//!     "main",
//!     "void main(){ gl_FragColor = vec4(tint * gain * level_at(0), 1.0); }",
//! )
//! .array("level", 8)
//! .param(Param::float("gain", 1.0))?])
//! .param(Param::color("tint", [1.0, 0.0, 0.0]))?;
//!
//! // width, height, components, DRI device (None probes), scanout, rgb565
//! let mut r = Renderer::new(64, 64, 4, None, false, false)?;
//! assert_eq!(r.setup(&scene)?, "");
//!
//! let frame = r.render(Some(&mut |_name, _size| Feed::Data(vec![1.0; 8])))?;
//! assert_eq!(frame.len(), 64 * 64 * 4);
//! assert_eq!(&frame[..4], &[255, 0, 0, 255]);
//! # Ok(())
//! # }
//! ```

pub mod assets;
pub mod bench;
pub mod cli;
pub mod console;
pub mod cparam;
pub mod drivers;
pub mod engine;
pub mod feeds;
pub mod ffi;
pub mod filters;
pub mod golden;
pub mod h264;
pub mod isf;
pub mod looper;
pub mod output;
pub mod procedural;
pub mod report;
pub mod scene;
pub mod sources;
pub mod stats;
