//! One loop for everything ghee renders. It draws a frame and hands it to
//! its sinks: an image file, a pipe, a panel, KMS scanout, H264, the
//! browser console, several at once. Each sink declares what it needs from
//! a frame (Output).
//!
//! GL is thread-affine, so the renderer belongs to the thread that runs the
//! loop. A host that runs the loop in a worker creates the renderer there.
//!
//! A host drives the loop through hooks: `poll` applies the host's queued
//! commands, `idle` says how long a skipped frame waits, `stopping` ends
//! the run, `on_frame` and `on_error` publish. A controller hooks the
//! active scene into the loop, and the loop reads its controller slot every
//! frame, so a host switches scenes mid-run by replacing it.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Result};

use crate::engine::{Feed, Given, Provider, Refused, Renderer};
use crate::layer::Address;
use crate::scene::{ArraySize, ParamValue};
use crate::stats::{monotonic, FrameRate};
use serde_json::Value;

/// a frame never hands a scene more than this, however long the loop idled
pub const MAX_DT: f64 = 0.25;

/// What the loop asks of a renderer. Renderer is the one implementation. The
/// loop's own tests drive a recording stand-in through it.
pub trait RenderTarget {
    fn width(&self) -> i32;
    fn height(&self) -> i32;
    fn components(&self) -> i32;
    fn set_dt(&self, value: f64);
    fn set_time(&self, t: f64);

    /// Set one live param. `pass` names a pass's param, and None takes the
    /// scene's. False when the pass or the uniform is missing. Required, a
    /// silent default would drop params with no error.
    fn set_param(
        &self,
        pass: Option<&str>,
        name: &str,
        ty: &str,
        value: &ParamValue,
    ) -> Result<bool>;

    /// Set one live field of a layer, as Renderer::set_layer() does. A
    /// target that draws no layers refuses every field.
    fn set_layer(&self, at: &Address, _given: Given) -> Result<()> {
        Err(Refused(format!("this renderer draws no layer '{}'", at.layer)).into())
    }

    /// Add a layer, written as a scene declares it, to the running scene at
    /// `index` among the layers of pass `pass`, as Renderer::add_layer()
    /// does. A target that draws no layers refuses every layer.
    fn add_layer(&mut self, pass: &str, _index: usize, _layer: &Value) -> Result<()> {
        Err(Refused(format!("this renderer adds no layer to pass '{pass}'")).into())
    }

    /// Give a layer of the running scene the look `layer`, written as a
    /// scene declares it, as Renderer::update_layer() does. False when the
    /// layer has that look already. A target that draws no layers refuses
    /// every look.
    fn update_layer(&mut self, layer: &Value) -> Result<bool> {
        let name = layer
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Err(Refused(format!("this renderer draws no layer '{name}'")).into())
    }

    fn draw(&mut self, provider: Option<Provider<'_>>) -> Result<()>;
    fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()>;
    fn present(
        &mut self,
        provider: Option<Provider<'_>>,
        preview: Option<(&mut [u8], (i32, i32))>,
    ) -> Result<()>;

    /// Blank the display, or light it again. A target that drives no display
    /// takes it as a no-op, so a sink can gate on an idle host either way.
    fn blank(&self, _on: bool) -> Result<()> {
        Ok(())
    }
}

impl RenderTarget for Renderer {
    fn width(&self) -> i32 {
        self.width
    }

    fn height(&self) -> i32 {
        self.height
    }

    fn components(&self) -> i32 {
        self.components
    }

    fn set_dt(&self, value: f64) {
        Renderer::set_dt(self, value);
    }

    fn set_time(&self, t: f64) {
        Renderer::set_time(self, Some(t));
    }

    fn set_param(
        &self,
        pass: Option<&str>,
        name: &str,
        ty: &str,
        value: &ParamValue,
    ) -> Result<bool> {
        match pass {
            Some(pass) => Renderer::set_param(self, pass, name, ty, value),
            None => Renderer::set_scene_param(self, name, ty, value),
        }
    }

    fn set_layer(&self, at: &Address, given: Given) -> Result<()> {
        Renderer::set_layer(self, at, given)
    }

    fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Result<()> {
        Renderer::add_layer(self, pass, index, layer)
    }

    fn update_layer(&mut self, layer: &Value) -> Result<bool> {
        Renderer::update_layer(self, layer)
    }

    fn draw(&mut self, provider: Option<Provider<'_>>) -> Result<()> {
        Renderer::draw(self, provider)
    }

    fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()> {
        Renderer::render_into(self, provider, out, out_size)
    }

    fn present(
        &mut self,
        provider: Option<Provider<'_>>,
        preview: Option<(&mut [u8], (i32, i32))>,
    ) -> Result<()> {
        Renderer::present(self, provider, preview)
    }

    fn blank(&self, on: bool) -> Result<()> {
        Renderer::blank(self, on)
    }
}

/// One frame as a sink sees it: the readback bytes, and their shape, row-major
/// (height, width, components).
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub data: &'a [u8],
    pub width: i32,
    pub height: i32,
    pub components: i32,
}

/// A frame buffer the loop renders into, and a pooled sink lends out.
pub struct OwnedFrame {
    pub data: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub components: i32,
}

impl OwnedFrame {
    /// (height, width, components), the layout order of the frame buffer.
    pub fn of(shape: (i32, i32, i32)) -> OwnedFrame {
        let (height, width, components) = shape;
        OwnedFrame {
            data: vec![
                0u8;
                (i64::from(height) * i64::from(width) * i64::from(components)) as usize
            ],
            width,
            height,
            components,
        }
    }

    #[allow(dead_code)]
    pub fn shape(&self) -> (i32, i32, i32) {
        (self.height, self.width, self.components)
    }

    pub fn view(&self) -> Frame<'_> {
        Frame {
            data: &self.data,
            width: self.width,
            height: self.height,
            components: self.components,
        }
    }
}

/// A sink takes an RGB(A) frame and puts it somewhere: an image file, a
/// hardware panel, a byte stream. Each sink declares what it needs from a
/// frame, through the flags and wants()/readback() below.
pub trait Output {
    /// what a host publishes as this sink's /stats destination
    fn dest(&self) -> &str {
        ""
    }

    /// the renderer presents to it (KMS), and nothing reads it back
    fn scanout(&self) -> bool {
        false
    }

    /// lends the loop its own buffers through acquire()/submit()
    fn pooled(&self) -> bool {
        false
    }

    /// write() is done with the buffer when it returns
    fn copies(&self) -> bool {
        true
    }

    /// keeps showing the last frame, so a clean one may be skipped
    fn persists(&self) -> bool {
        false
    }

    /// True when this sink takes a CPU frame this iteration. A GPU-fed sink
    /// (scanout, a hardware encoder) answers false. A sink that throttles
    /// itself answers false too.
    fn wants(&self, _frame: u64) -> bool {
        true
    }

    /// (w, h) to read the frame back at. None reads the render size back
    /// directly. A size is a GPU downscale: it costs a blit and saves the
    /// bytes.
    fn readback(&self) -> Option<(i32, i32)> {
        None
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()>;

    /// A buffer of `shape` to render into, pooled sinks only.
    fn acquire(&mut self, _shape: (i32, i32, i32)) -> Result<OwnedFrame> {
        bail!("not a pooled sink")
    }

    /// Take back a filled buffer, pooled sinks only.
    fn submit(&mut self, _frame: OwnedFrame) -> Result<()> {
        bail!("not a pooled sink")
    }

    /// optional hook: sample the GPU after the draw
    fn mirror(&mut self, _renderer: &dyn RenderTarget) -> Result<()> {
        Ok(())
    }

    /// optional hook: a sink with a power state (a backlight)
    fn turn_on(&mut self) -> Result<()> {
        Ok(())
    }

    fn turn_off(&mut self) -> Result<()> {
        Ok(())
    }

    /// optional hook: blank a surface that holds its frame
    fn clear(&mut self) -> Result<()> {
        Ok(())
    }

    /// optional hook: a sink overrides it only when it holds resources
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Paces a render loop, as output::FixedPacer does.
pub trait Pacer {
    /// Seconds to sleep after a frame. write_s is this frame's sink write.
    /// frame_s is the whole frame so far (render + write).
    fn sleep_for(&mut self, write_s: f64, frame_s: f64) -> f64;

    /// The current frame period, with no new measurement. A skipped (clean)
    /// frame sleeps this long before it polls again.
    fn idle_period(&self) -> f64;
}

/// Hooks a scene into the render loop. Every method is optional.
pub trait SceneController {
    /// Once an iteration, before the skip decision and before update(). The
    /// controller of a procedural scene reads the messages of the scene's
    /// process here and sends it this tick. An error ends the frame, so a
    /// process that died reports it here.
    fn tick(&mut self, _frame: u64, _t: f64, _dt: f64) -> Result<()> {
        Ok(())
    }

    /// Floats for a declared array this frame. None falls through to the
    /// host's source. Feed::Unchanged declares that the engine's texture
    /// already holds this array.
    fn provide(&mut self, _name: &str, _size: ArraySize, _frame: u64) -> Option<Feed> {
        None
    }

    /// A host-side (string) param changes. The next tick carries it.
    fn on_param(&mut self, _name: &str, _value: &str) {}

    /// Per-frame hook before the draw (e.g. set live params).
    fn update(&mut self, _renderer: &mut dyn RenderTarget, _frame: u64, _t: f64) -> Result<()> {
        Ok(())
    }

    /// False means this frame matches the last one, so the loop skips it
    /// whole. The display keeps scanning out the last flipped buffer. The
    /// default is true: every frame renders, which a scene animating on
    /// `iTime` needs. `iFrame` advances over a skipped frame too.
    fn dirty(&mut self, _frame: u64, _t: f64) -> bool {
        true
    }
}

/// The loop reads its controller every frame, so a host switches scenes mid-run
/// by putting a new one in here.
pub type ControllerSlot = RefCell<Option<Box<dyn SceneController>>>;

/// The loop draws on any of four demands. The demands are a queued command, the
/// first frame, a scene with no controller, and a controller that reports
/// dirty. A sink has to show something once, so the first frame always draws.
pub fn should_draw(
    forced: bool,
    controller: Option<&mut dyn SceneController>,
    frame: u64,
    t: f64,
) -> bool {
    let Some(controller) = controller else {
        return true;
    };
    if forced || frame == 0 {
        return true;
    }
    controller.dirty(frame, t)
}

fn sleep(seconds: f64) {
    if seconds > 0.0 {
        thread::sleep(Duration::from_secs_f64(seconds));
    }
}

/// The destination buffer for this frame's readback.
enum Target {
    /// no readback
    None,
    /// the index of the sink rendered straight into
    Pool(usize, OwnedFrame),
    /// a sink keeps the frame past write(), so this readback gets its own memory
    Fresh(OwnedFrame),
    /// the loop's one reused buffer
    Reuse,
}

pub type LoopProvider<'a> = Box<dyn FnMut(&str, ArraySize, u64) -> Feed + 'a>;
pub type OnFrame<'a> = Box<dyn FnMut(&Loop<'a>, f64) + 'a>;
pub type OnError<'a> = Box<dyn FnMut(&anyhow::Error) -> bool + 'a>;

/// Draw frames into a list of sinks. The run ends on the frame budget, or when
/// the host stops it.
///
/// frames=0 runs until `stopping`. Any other count is a finite render. The loop
/// turns its sinks on and off around the run, and never closes them. Their
/// lifetime belongs to their builder.
pub struct Loop<'a> {
    pub renderer: &'a mut dyn RenderTarget,
    pub sinks: Vec<&'a mut dyn Output>,
    pub controller: Option<&'a ControllerSlot>,
    /// (name, size, frame) -> this frame's data for one declared array
    pub provider: Option<LoopProvider<'a>>,
    pub frames: u64,
    /// pin iTime to frame * timestep and iTimeDelta to timestep
    pub timestep: Option<f64>,
    pub pacer: Option<Box<dyn Pacer + 'a>>,
    /// frames drawn, counted into a rate a host may read from its own thread
    pub rate: Option<Arc<Mutex<FrameRate>>>,
    /// seconds yielded after a frame, for a host's threads
    pub rest: f64,
    /// -> true to force a draw this iteration
    pub poll: Option<Box<dyn FnMut() -> bool + 'a>>,
    pub stopping: Option<Box<dyn FnMut() -> bool + 'a>>,
    pub on_frame: Option<OnFrame<'a>>,
    /// -> true to keep the loop alive past an error
    pub on_error: Option<OnError<'a>>,
    /// how long a skipped frame waits, given the measured period
    pub idle_hook: Option<Box<dyn FnMut(Option<f64>) -> f64 + 'a>>,
    /// iterations, skipped ones included
    pub frame: u64,
    pub drawn: u64,
    pub elapsed: f64,
    /// measured frame period, the iTimeDelta fed to a scene
    pub period: Option<f64>,
    prev: Option<f64>,
    buf: Vec<u8>,
    buf_shape: (i32, i32, i32),
}

impl<'a> Loop<'a> {
    pub fn new(renderer: &'a mut dyn RenderTarget, sinks: Vec<&'a mut dyn Output>) -> Loop<'a> {
        Loop {
            renderer,
            sinks,
            controller: None,
            provider: None,
            frames: 0,
            timestep: None,
            pacer: None,
            rate: None,
            rest: 0.0,
            poll: None,
            stopping: None,
            on_frame: None,
            on_error: None,
            idle_hook: None,
            frame: 0,
            drawn: 0,
            elapsed: 0.0,
            period: None,
            prev: None,
            buf: Vec::new(),
            buf_shape: (0, 0, 0),
        }
    }

    /// The frames reach these destinations, by name (stats::DEST_*).
    pub fn out(&self) -> Vec<&str> {
        self.sinks
            .iter()
            .map(|s| s.dest())
            .filter(|d| !d.is_empty())
            .collect()
    }

    /// How long a skipped (clean) frame sleeps before re-polling.
    pub fn idle(&mut self) -> f64 {
        let period = self.period;
        if let Some(f) = self.idle_hook.as_mut() {
            return f(period);
        }
        if let Some(p) = self.pacer.as_ref() {
            return p.idle_period();
        }
        match period {
            Some(p) if p != 0.0 => p,
            _ => 1.0 / 60.0,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        let scanout = self.sinks.iter().any(|s| s.scanout());
        // Only an endless run into sinks that all hold their last frame may
        // skip a clean frame. A file or a pipe loses it.
        let skippable =
            self.frames == 0 && !self.sinks.is_empty() && self.sinks.iter().all(|s| s.persists());
        let mut lit = false;
        let started = monotonic();
        let ran = self.pump(scanout, skippable, &mut lit, started);
        let shut = self.shutdown();
        shut.and(ran)
    }

    fn shutdown(&mut self) -> Result<()> {
        for s in self.sinks.iter_mut() {
            s.clear()?;
            s.turn_off()?;
        }
        Ok(())
    }

    fn pump(&mut self, scanout: bool, skippable: bool, lit: &mut bool, started: f64) -> Result<()> {
        while (self.frames == 0 || self.frame < self.frames) && !self.done() {
            if let Err(e) = self.step(scanout, skippable, lit, started) {
                let alive = match self.on_error.as_mut() {
                    Some(f) => f(&e),
                    None => false,
                };
                if !alive {
                    return Err(e);
                }
                sleep(0.25);
            }
        }
        Ok(())
    }

    fn done(&mut self) -> bool {
        match self.stopping.as_mut() {
            Some(f) => f(),
            None => false,
        }
    }

    fn step(&mut self, scanout: bool, skippable: bool, lit: &mut bool, started: f64) -> Result<()> {
        let t0 = monotonic();
        let forced = match self.poll.as_mut() {
            Some(p) => p(),
            None => false,
        };
        // the t handed to a controller. Unpinned it is the wall clock
        // since run() started: a scene synced to outside time (music) must
        // not drift with the loop's own pace.
        let t = match self.timestep {
            Some(step) => self.frame as f64 * step,
            None => t0 - started,
        };
        // The period since the last frame drawn, which iTimeDelta takes when
        // this iteration draws.
        let period = self.prev.map(|prev| t0 - prev);
        // before the skip decision: a scene fed from outside says whether this
        // tick needs a frame
        self.tick_controller(t, period)?;
        if skippable && !self.should_draw_now(forced, t) {
            self.frame += 1;
            let idle = self.idle();
            sleep(idle);
            return Ok(());
        }
        if period.is_some() {
            self.period = period;
        }
        self.prev = Some(t0);
        // The one place the loop feeds iTimeDelta: the measured frame period,
        // or the pinned step for a reproducible render. Before the first
        // measurement the engine's default stands.
        if let Some(step) = self.timestep {
            self.renderer.set_dt(step);
        } else if let Some(period) = self.period {
            // a draw after a skip stretch measures the whole gap: one frame
            // must never step a simulation by seconds
            self.renderer.set_dt(period.min(MAX_DT));
        }
        let drew = self.draw_frame(scanout, t)?;
        if !*lit {
            // the panel now holds a frame to light
            *lit = true;
            for s in self.sinks.iter_mut() {
                s.turn_on()?;
            }
        }
        self.frame += 1;
        self.drawn += 1;
        let end = monotonic();
        self.elapsed = end - started;
        if let Some(rate) = self.rate.as_ref() {
            rate.lock()
                .unwrap_or_else(|e| e.into_inner())
                .tick(Some(end));
        }
        if let Some(mut hook) = self.on_frame.take() {
            hook(self, end - t0);
            self.on_frame = Some(hook);
        }
        if let Some(pacer) = self.pacer.as_mut() {
            let now = monotonic();
            sleep(pacer.sleep_for(now - drew, now - t0));
        } else if self.rest != 0.0 {
            sleep(self.rest);
        }
        Ok(())
    }

    /// Hand the controller this iteration, drawn or skipped. Its dt is the
    /// iTimeDelta that the frame draws with: the pinned step, else `period`,
    /// the time since the last frame drawn, which MAX_DT caps, else the
    /// engine's first 1/60.
    fn tick_controller(&mut self, t: f64, period: Option<f64>) -> Result<()> {
        let Some(cell) = self.controller else {
            return Ok(());
        };
        let frame = self.frame;
        let dt = match self.timestep {
            Some(step) => step,
            None => period.map_or(1.0 / 60.0, |p| p.min(MAX_DT)),
        };
        let mut slot = cell.borrow_mut();
        match (*slot).as_mut() {
            Some(c) => c.tick(frame, t, dt),
            None => Ok(()),
        }
    }

    fn should_draw_now(&mut self, forced: bool, t: f64) -> bool {
        let frame = self.frame;
        match self.controller {
            None => should_draw(forced, None, frame, t),
            Some(cell) => {
                let mut slot = cell.borrow_mut();
                let ctrl: Option<&mut dyn SceneController> = match (*slot).as_mut() {
                    Some(c) => Some(&mut **c),
                    None => None,
                };
                should_draw(forced, ctrl, frame, t)
            }
        }
    }

    /// Update the controller, render, and hand the frame to its sinks. Returns
    /// the clock at the end of the render, before the sink writes.
    fn draw_frame(&mut self, scanout: bool, t: f64) -> Result<f64> {
        let frame = self.frame;
        let timestep = self.timestep;
        let Loop {
            renderer,
            sinks,
            controller,
            provider,
            buf,
            buf_shape,
            ..
        } = self;
        let renderer: &mut dyn RenderTarget = &mut **renderer;
        let mut slot = controller.map(|c| c.borrow_mut());
        let mut ctrl: Option<&mut dyn SceneController> = match slot.as_mut() {
            Some(s) => match (**s).as_mut() {
                Some(c) => Some(&mut **c),
                None => None,
            },
            None => None,
        };
        if let Some(c) = ctrl.as_mut() {
            c.update(renderer, frame, t)?;
        }
        if timestep.is_some() {
            renderer.set_time(t);
        }
        let readers: Vec<usize> = (0..sinks.len())
            .filter(|i| sinks[*i].wants(frame))
            .collect();
        let size = readback_size(sinks, &readers);
        let pool = pool_index(sinks, &readers, size);
        let shape = match size {
            None => (renderer.height(), renderer.width(), renderer.components()),
            Some((w, h)) => (h, w, renderer.components()),
        };
        let mut target = if readers.is_empty() {
            Target::None
        } else if let Some(i) = pool {
            Target::Pool(i, sinks[i].acquire(shape)?)
        } else if readers.iter().all(|i| sinks[*i].copies()) {
            if *buf_shape != shape || buf.len() != frame_len(shape) {
                *buf = vec![0u8; frame_len(shape)];
                *buf_shape = shape;
            }
            Target::Reuse
        } else {
            // a sink keeps the frame past write() and hands it to another
            // thread. The next readback needs its own memory.
            Target::Fresh(OwnedFrame::of(shape))
        };
        // The array provider for one frame. The controller answers first. The
        // host's source (or the placeholder) takes anything it leaves.
        let has_feed = ctrl.is_some() || provider.is_some();
        let mut feed = |name: &str, size: ArraySize| -> Feed {
            if let Some(c) = ctrl.as_mut() {
                if let Some(data) = c.provide(name, size, frame) {
                    return data;
                }
            }
            match provider.as_mut() {
                Some(p) => p(name, size, frame),
                None => Feed::Unchanged,
            }
        };
        let out: Option<&mut [u8]> = match &mut target {
            Target::None => None,
            Target::Pool(_, f) | Target::Fresh(f) => Some(&mut f.data),
            Target::Reuse => Some(&mut buf[..]),
        };
        {
            let provider: Option<Provider<'_>> = if has_feed { Some(&mut feed) } else { None };
            if scanout {
                // present() reads back through a downscale blit, or not at
                // all. It has no direct full-frame path.
                let preview = out.map(|o| {
                    let (w, h) = size.unwrap_or((renderer.width(), renderer.height()));
                    (o, (w, h))
                });
                renderer.present(provider, preview)?;
            } else if let Some(o) = out {
                renderer.render_into(provider, o, size)?;
            } else {
                renderer.draw(provider)?;
            }
        }
        let drew = monotonic();
        match target {
            Target::Pool(i, f) => sinks[i].submit(f)?,
            Target::Fresh(f) => {
                let view = f.view();
                for i in &readers {
                    sinks[*i].write(view)?;
                }
            }
            Target::Reuse => {
                let view = Frame {
                    data: buf,
                    width: shape.1,
                    height: shape.0,
                    components: shape.2,
                };
                for i in &readers {
                    sinks[*i].write(view)?;
                }
            }
            Target::None => {}
        }
        for s in sinks.iter_mut() {
            s.mirror(renderer)?;
        }
        Ok(drew)
    }
}

fn frame_len(shape: (i32, i32, i32)) -> usize {
    (i64::from(shape.0) * i64::from(shape.1) * i64::from(shape.2)) as usize
}

/// This set of readers reads the frame back at this size. None reads the render
/// size back directly. Any other answer is the largest downscale asked for, and
/// a smaller consumer downscales further itself.
fn readback_size(sinks: &[&mut dyn Output], readers: &[usize]) -> Option<(i32, i32)> {
    if readers.is_empty() {
        return None;
    }
    let mut size = (0, 0);
    for i in readers {
        let (w, h) = sinks[*i].readback()?;
        size = (size.0.max(w), size.1.max(h));
    }
    Some(size)
}

/// The one sink open to a direct render: a pooled sink that is the frame's
/// only consumer, at the render size. Anything else takes a copy.
fn pool_index(
    sinks: &[&mut dyn Output],
    readers: &[usize],
    size: Option<(i32, i32)>,
) -> Option<usize> {
    match readers {
        [only] if sinks[*only].pooled() && size.is_none() => Some(*only),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::stats::{DEST_FILE, DEST_KMS, DEST_PANEL};

    const W: i32 = 64;
    const H: i32 = 48;
    const COMPONENTS: i32 = 4;

    /// One recorded call on the renderer: (kind, readback bytes, out_size).
    type Call = (&'static str, Option<usize>, Option<(i32, i32)>);

    /// A Renderer stand-in that records what the loop asks of it.
    #[derive(Default)]
    struct Recorder {
        calls: Vec<Call>,
        /// the buffer handed to each readback
        buffers: Vec<usize>,
        dts: RefCell<Vec<f64>>,
        times: RefCell<Vec<f64>>,
        fed: Vec<Vec<f32>>,
    }

    impl Recorder {
        fn note(
            &mut self,
            kind: &'static str,
            out: Option<&mut [u8]>,
            out_size: Option<(i32, i32)>,
            provider: Option<Provider<'_>>,
        ) {
            self.calls
                .push((kind, out.as_ref().map(|o| o.len()), out_size));
            self.buffers.push(out.map_or(0, |o| o.as_ptr() as usize));
            if let Some(p) = provider {
                if let Feed::Data(vals) = p("vals", ArraySize::Linear(4)) {
                    self.fed.push(vals);
                }
            }
        }
    }

    impl RenderTarget for Recorder {
        fn width(&self) -> i32 {
            W
        }

        fn set_param(
            &self,
            _pass: Option<&str>,
            _name: &str,
            _ty: &str,
            _value: &ParamValue,
        ) -> Result<bool> {
            Ok(false) // the recorder holds no uniforms
        }

        fn height(&self) -> i32 {
            H
        }

        fn components(&self) -> i32 {
            COMPONENTS
        }

        fn set_dt(&self, value: f64) {
            self.dts.borrow_mut().push(value);
        }

        fn set_time(&self, t: f64) {
            self.times.borrow_mut().push(t);
        }

        fn draw(&mut self, provider: Option<Provider<'_>>) -> Result<()> {
            self.note("draw", None, None, provider);
            Ok(())
        }

        fn render_into(
            &mut self,
            provider: Option<Provider<'_>>,
            out: &mut [u8],
            out_size: Option<(i32, i32)>,
        ) -> Result<()> {
            self.note("render", Some(out), out_size, provider);
            Ok(())
        }

        fn present(
            &mut self,
            provider: Option<Provider<'_>>,
            preview: Option<(&mut [u8], (i32, i32))>,
        ) -> Result<()> {
            match preview {
                Some((out, size)) => self.note("present", Some(out), Some(size), provider),
                None => self.note("present", None, None, provider),
            }
            Ok(())
        }
    }

    /// What one sink write sees.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Got {
        at: usize,
        len: usize,
        width: i32,
        height: i32,
    }

    /// A recording sink, the flags under test are constructor fields.
    struct Sink {
        dest: &'static str,
        scanout: bool,
        pooled: bool,
        copies: bool,
        persists: bool,
        size: Option<(i32, i32)>,
        take: bool,
        boom: bool,
        frames: Vec<Got>,
        mirrors: u32,
        on: u32,
        off: u32,
        cleared: u32,
        lent: Vec<usize>,
        submitted: Vec<usize>,
    }

    impl Sink {
        fn new() -> Sink {
            Sink {
                dest: "test",
                scanout: false,
                pooled: false,
                copies: true,
                persists: false,
                size: None,
                take: true,
                boom: false,
                frames: Vec::new(),
                mirrors: 0,
                on: 0,
                off: 0,
                cleared: 0,
                lent: Vec::new(),
                submitted: Vec::new(),
            }
        }

        fn named(dest: &'static str) -> Sink {
            Sink {
                dest,
                ..Sink::new()
            }
        }

        /// A sink the loop may render straight into, like ImageSequence.
        fn pool() -> Sink {
            Sink {
                pooled: true,
                ..Sink::new()
            }
        }

        /// KMS: the renderer page-flips the final buffer itself, so this sink
        /// holds nothing and never takes a frame.
        fn scanout() -> Sink {
            Sink {
                dest: DEST_KMS,
                scanout: true,
                persists: true,
                take: false,
                ..Sink::new()
            }
        }
    }

    impl Output for Sink {
        fn dest(&self) -> &str {
            self.dest
        }

        fn scanout(&self) -> bool {
            self.scanout
        }

        fn pooled(&self) -> bool {
            self.pooled
        }

        fn copies(&self) -> bool {
            self.copies
        }

        fn persists(&self) -> bool {
            self.persists
        }

        fn wants(&self, _frame: u64) -> bool {
            self.take
        }

        fn readback(&self) -> Option<(i32, i32)> {
            self.size
        }

        fn write(&mut self, frame: Frame<'_>) -> Result<()> {
            if self.boom {
                bail!("boom");
            }
            self.frames.push(Got {
                at: frame.data.as_ptr() as usize,
                len: frame.data.len(),
                width: frame.width,
                height: frame.height,
            });
            Ok(())
        }

        fn acquire(&mut self, shape: (i32, i32, i32)) -> Result<OwnedFrame> {
            let f = OwnedFrame::of(shape);
            self.lent.push(f.data.as_ptr() as usize);
            Ok(f)
        }

        fn submit(&mut self, frame: OwnedFrame) -> Result<()> {
            self.submitted.push(frame.data.as_ptr() as usize);
            Ok(())
        }

        fn mirror(&mut self, _renderer: &dyn RenderTarget) -> Result<()> {
            self.mirrors += 1;
            Ok(())
        }

        fn turn_on(&mut self) -> Result<()> {
            self.on += 1;
            Ok(())
        }

        fn turn_off(&mut self) -> Result<()> {
            self.off += 1;
            Ok(())
        }

        fn clear(&mut self) -> Result<()> {
            self.cleared += 1;
            Ok(())
        }
    }

    struct Clean;

    impl SceneController for Clean {
        fn dirty(&mut self, _frame: u64, _t: f64) -> bool {
            false
        }
    }

    fn slot(c: impl SceneController + 'static) -> ControllerSlot {
        RefCell::new(Some(Box::new(c)))
    }

    /// A stopping hook that ends the run after `n` polls.
    fn after(n: u64) -> impl FnMut() -> bool {
        let mut seen = 0;
        move || {
            seen += 1;
            seen > n
        }
    }

    // what reaches a sink

    #[test]
    fn one_frame_reaches_every_sink() {
        let mut r = Recorder::default();
        let (mut a, mut b) = (Sink::named("a"), Sink::named("b"));
        {
            let mut lp = Loop::new(&mut r, vec![&mut a, &mut b]);
            lp.frames = 3;
            lp.run().unwrap();
        }
        assert_eq!(a.frames.len(), 3);
        assert_eq!(b.frames.len(), 3);
        assert_eq!(a.frames[0], b.frames[0]); // one readback, one buffer, both sinks
    }

    #[test]
    fn a_sink_that_takes_no_frame_gets_no_readback() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.take = false;
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 2;
            lp.run().unwrap();
        }
        assert_eq!(
            r.calls.iter().map(|c| c.0).collect::<Vec<&str>>(),
            vec!["draw", "draw"]
        );
        assert!(s.frames.is_empty());
    }

    #[test]
    fn the_readback_is_the_largest_downscale_asked_for() {
        let mut r = Recorder::default();
        let (mut a, mut b) = (Sink::new(), Sink::new());
        a.size = Some((32, 24));
        b.size = Some((20, 40));
        {
            let mut lp = Loop::new(&mut r, vec![&mut a, &mut b]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert_eq!(
            r.calls[0],
            ("render", Some((40 * 32 * 4) as usize), Some((32, 40)))
        );
    }

    #[test]
    fn a_sink_needing_the_whole_frame_wins_over_a_downscale() {
        // None is the render size read back directly. A smaller consumer
        // downscales the big frame itself.
        let mut r = Recorder::default();
        let (mut a, mut b) = (Sink::new(), Sink::new());
        b.size = Some((32, 24));
        {
            let mut lp = Loop::new(&mut r, vec![&mut a, &mut b]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert_eq!(r.calls[0], ("render", Some((H * W * 4) as usize), None));
    }

    #[test]
    fn a_pooled_sink_alone_is_rendered_straight_into() {
        let mut r = Recorder::default();
        let mut p = Sink::pool();
        {
            let mut lp = Loop::new(&mut r, vec![&mut p]);
            lp.frames = 3;
            lp.run().unwrap();
        }
        assert_eq!(p.submitted.len(), 3);
        assert_eq!(p.lent, p.submitted); // the loop draws into the sink's buffers
        assert_eq!(r.buffers[0], p.lent[0]);
        assert!(p.frames.is_empty()); // write() never runs: nothing copies
    }

    #[test]
    fn a_pooled_sink_beside_another_takes_a_copy() {
        let mut r = Recorder::default();
        let (mut p, mut s) = (Sink::pool(), Sink::new());
        {
            let mut lp = Loop::new(&mut r, vec![&mut p, &mut s]);
            lp.frames = 2;
            lp.run().unwrap();
        }
        assert!(p.submitted.is_empty()); // the pool cannot own a buffer two sinks read
        assert_eq!(p.frames.len(), 2);
        assert_eq!(s.frames.len(), 2);
    }

    #[test]
    fn a_pooled_sink_asked_for_a_downscale_takes_a_copy() {
        let mut r = Recorder::default();
        let mut p = Sink::pool();
        p.size = Some((16, 12));
        {
            let mut lp = Loop::new(&mut r, vec![&mut p]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert!(p.submitted.is_empty());
        assert_eq!((p.frames[0].width, p.frames[0].height), (16, 12));
        assert_eq!(p.frames[0].len, (12 * 16 * 4) as usize);
    }

    #[test]
    fn a_sink_that_copies_gets_one_buffer_reused() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 3;
            lp.run().unwrap();
        }
        assert_eq!(s.frames.len(), 3);
        assert!(s.frames.iter().all(|f| f.at == s.frames[0].at));
    }

    #[test]
    fn a_sink_that_keeps_the_frame_gets_a_buffer_of_its_own() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.copies = false;
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 3;
            lp.run().unwrap();
            // the loop keeps no buffer to reuse, so nothing overwrites a held frame
            assert!(lp.buf.is_empty());
        }
        assert_eq!(s.frames.len(), 3);
    }

    // the scanout

    #[test]
    fn a_scanout_presents_and_never_takes_a_frame() {
        let mut r = Recorder::default();
        let mut k = Sink::scanout();
        let out;
        {
            let mut lp = Loop::new(&mut r, vec![&mut k]);
            lp.frames = 2;
            lp.run().unwrap();
            out = lp.out().iter().map(|d| d.to_string()).collect::<Vec<_>>();
        }
        assert_eq!(
            r.calls.iter().map(|c| c.0).collect::<Vec<&str>>(),
            vec!["present", "present"]
        );
        assert_eq!(r.calls[0], ("present", None, None));
        assert_eq!(out, vec![DEST_KMS]);
    }

    #[test]
    fn a_scanout_beside_a_reader_reads_the_frame_back_too() {
        // present() has no direct full-frame readback, so the size is explicit.
        let mut r = Recorder::default();
        let (mut k, mut s) = (Sink::scanout(), Sink::new());
        {
            let mut lp = Loop::new(&mut r, vec![&mut k, &mut s]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert_eq!(
            r.calls[0],
            ("present", Some((H * W * 4) as usize), Some((W, H)))
        );
    }

    #[test]
    fn a_scanout_beside_a_downscaling_reader_downscales_on_the_gpu() {
        let mut r = Recorder::default();
        let (mut k, mut s) = (Sink::scanout(), Sink::new());
        s.size = Some((32, 24));
        {
            let mut lp = Loop::new(&mut r, vec![&mut k, &mut s]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert_eq!(
            r.calls[0],
            ("present", Some((24 * 32 * 4) as usize), Some((32, 24)))
        );
    }

    // a mirror sink samples the GPU itself

    #[test]
    fn a_mirror_sink_runs_after_the_draw() {
        let mut r = Recorder::default();
        let mut m = Sink::new();
        m.take = false;
        {
            let mut lp = Loop::new(&mut r, vec![&mut m]);
            lp.frames = 3;
            lp.run().unwrap();
        }
        assert_eq!(m.mirrors, 3);
        assert!(m.frames.is_empty());
    }

    // iTimeDelta and iTime, fed in one place

    #[test]
    fn dt_is_the_measured_frame_period() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 4;
            lp.run().unwrap();
        }
        let dts = r.dts.borrow();
        assert_eq!(dts.len(), 3); // frame 0 has no previous frame to measure against
        assert!(dts.iter().all(|v| *v > 0.0 && *v < 5.0));
        assert!(r.times.borrow().is_empty()); // the engine's own clock still runs the scene
    }

    #[test]
    fn a_pinned_timestep_is_both_the_dt_and_the_time() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 3;
            lp.timestep = Some(0.5);
            lp.run().unwrap();
        }
        assert_eq!(*r.dts.borrow(), vec![0.5, 0.5, 0.5]);
        assert_eq!(*r.times.borrow(), vec![0.0, 0.5, 1.0]);
    }

    /// A scene fed from outside reads in its tick the iTimeDelta that the
    /// tick's frame draws with.
    #[test]
    fn a_tick_carries_the_dt_that_its_frame_draws_with() {
        struct Ticker(Arc<Mutex<Vec<f64>>>);

        impl SceneController for Ticker {
            fn tick(&mut self, _frame: u64, _t: f64, dt: f64) -> Result<()> {
                self.0.lock().unwrap().push(dt);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut r = Recorder::default();
        let mut s = Sink::new();
        let ctrl = slot(Ticker(seen.clone()));
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.frames = 4;
            lp.rest = 0.01;
            lp.run().unwrap();
        }
        let seen = seen.lock().unwrap();
        let dts = r.dts.borrow();
        // frame 0 draws with the engine's own 1/60
        assert_eq!(seen[0], 1.0 / 60.0);
        assert_eq!(seen[1..], dts[..]);
    }

    #[test]
    fn the_first_frame_keeps_the_engine_default_dt() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert!(r.dts.borrow().is_empty());
    }

    // skipping a clean frame

    #[test]
    fn a_clean_frame_is_skipped_when_every_sink_holds_its_last() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.persists = true;
        let ctrl = slot(Clean);
        let (frame, drawn);
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.stopping = Some(Box::new(after(4)));
            lp.idle_hook = Some(Box::new(|_| 0.0));
            lp.run().unwrap();
            (frame, drawn) = (lp.frame, lp.drawn);
        }
        assert!(frame > 1);
        assert_eq!(drawn, 1); // the first frame draws, the rest are skipped
        assert_eq!(s.frames.len(), 1);
    }

    #[test]
    fn a_sink_that_does_not_hold_its_last_frame_is_never_skipped() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        let ctrl = slot(Clean);
        let drawn;
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.frames = 4;
            lp.run().unwrap();
            drawn = lp.drawn;
        }
        assert_eq!(drawn, 4);
        assert_eq!(s.frames.len(), 4);
    }

    #[test]
    fn a_finite_render_draws_every_frame_even_for_a_clean_scene() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.persists = true;
        let ctrl = slot(Clean);
        let drawn;
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.frames = 5;
            lp.run().unwrap();
            drawn = lp.drawn;
        }
        assert_eq!(drawn, 5);
    }

    #[test]
    fn a_skipped_frame_advances_the_frame_number() {
        struct Watcher(Arc<Mutex<Vec<u64>>>);

        impl SceneController for Watcher {
            fn dirty(&mut self, frame: u64, _t: f64) -> bool {
                self.0.lock().unwrap().push(frame);
                false
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.persists = true;
        let ctrl = slot(Watcher(seen.clone()));
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.stopping = Some(Box::new(after(4)));
            lp.idle_hook = Some(Box::new(|_| 0.0));
            lp.run().unwrap();
        }
        let seen = seen.lock().unwrap();
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(*seen, sorted);
        assert!(seen.len() > 1);
    }

    #[test]
    fn the_idle_wait_falls_back_to_the_measured_period() {
        // A scanout render with no --fps has no pacer and no host interval.
        // idle() still answers.
        let mut r = Recorder::default();
        let mut k = Sink::scanout();
        let mut lp = Loop::new(&mut r, vec![&mut k]);
        assert!((lp.idle() - 1.0 / 60.0).abs() < 1e-12);
        lp.period = Some(0.02);
        assert_eq!(lp.idle(), 0.02);
    }

    // the frame budget and what it publishes

    #[test]
    fn frames_is_the_budget_and_drawn_is_what_landed() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        let mut lp = Loop::new(&mut r, vec![&mut s]);
        lp.frames = 7;
        lp.run().unwrap();
        assert_eq!((lp.frame, lp.drawn), (7, 7));
        assert!(lp.elapsed > 0.0);
    }

    #[test]
    fn an_endless_loop_runs_until_the_host_stops_it() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        let mut lp = Loop::new(&mut r, vec![&mut s]);
        lp.stopping = Some(Box::new(after(3)));
        lp.run().unwrap();
        assert_eq!(lp.drawn, 3);
    }

    #[test]
    fn out_names_every_sinks_destination_and_skips_a_nameless_one() {
        let mut r = Recorder::default();
        let (mut k, mut p, mut f) = (
            Sink::scanout(),
            Sink::named(DEST_PANEL),
            Sink::named(DEST_FILE),
        );
        {
            let lp = Loop::new(&mut r, vec![&mut k, &mut p, &mut f]);
            assert_eq!(lp.out(), vec![DEST_KMS, DEST_PANEL, DEST_FILE]);
        }
        let mut nameless = Sink::named("");
        let lp = Loop::new(&mut r, vec![&mut nameless]);
        assert!(lp.out().is_empty());
    }

    #[test]
    fn on_frame_reports_each_frames_own_cost() {
        let costs: Cell<Vec<f64>> = Cell::new(Vec::new());
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 3;
            lp.on_frame = Some(Box::new(|_, cost| {
                let mut seen = costs.take();
                seen.push(cost);
                costs.set(seen);
            }));
            lp.run().unwrap();
        }
        let seen = costs.take();
        assert_eq!(seen.len(), 3);
        assert!(seen.iter().all(|c| *c >= 0.0));
    }

    // the sinks' power state

    #[test]
    fn a_sink_is_turned_on_once_there_is_a_frame_to_show() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 3;
            lp.run().unwrap();
        }
        assert_eq!(s.on, 1);
        // the backlight comes up over a drawn frame
        assert!(!s.frames.is_empty());
    }

    #[test]
    fn a_sink_is_blanked_and_turned_off_on_the_way_out() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 2;
            lp.run().unwrap();
        }
        assert_eq!((s.cleared, s.off), (1, 1));
    }

    #[test]
    fn a_run_that_drew_nothing_never_turns_a_sink_on() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.stopping = Some(Box::new(|| true));
            lp.run().unwrap();
        }
        assert_eq!((s.on, s.cleared, s.off), (0, 1, 1));
    }

    // the controller

    #[test]
    fn the_controller_can_be_swapped_mid_run() {
        struct First;
        struct Second;

        impl SceneController for First {
            fn update(
                &mut self,
                renderer: &mut dyn RenderTarget,
                _frame: u64,
                _t: f64,
            ) -> Result<()> {
                renderer.set_time(1.0);
                Ok(())
            }
        }

        impl SceneController for Second {
            fn update(
                &mut self,
                renderer: &mut dyn RenderTarget,
                _frame: u64,
                _t: f64,
            ) -> Result<()> {
                renderer.set_time(2.0);
                Ok(())
            }
        }

        let mut r = Recorder::default();
        let mut s = Sink::new();
        let ctrl = slot(First);
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.frames = 4;
            lp.on_frame = Some(Box::new(|_, _| {
                *ctrl.borrow_mut() = Some(Box::new(Second));
            }));
            lp.run().unwrap();
        }
        // update() writes through the renderer, so each recorded time names
        // the controller of that frame
        assert_eq!(*r.times.borrow(), vec![1.0, 2.0, 2.0, 2.0]);
    }

    /// The loop ticks a scene fed from outside even for a skipped frame. The
    /// scene answers only after a tick.
    #[test]
    fn the_controller_is_ticked_once_an_iteration_drawn_or_skipped() {
        struct Ticker(Arc<Mutex<Vec<(u64, f64)>>>);

        impl SceneController for Ticker {
            fn tick(&mut self, frame: u64, _t: f64, dt: f64) -> Result<()> {
                self.0.lock().unwrap().push((frame, dt));
                Ok(())
            }

            fn dirty(&mut self, _frame: u64, _t: f64) -> bool {
                false
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.persists = true;
        let ctrl = slot(Ticker(seen.clone()));
        let (frame, drawn);
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.timestep = Some(0.25);
            lp.stopping = Some(Box::new(after(4)));
            lp.idle_hook = Some(Box::new(|_| 0.0));
            lp.run().unwrap();
            (frame, drawn) = (lp.frame, lp.drawn);
        }
        let seen = seen.lock().unwrap();
        assert_eq!(drawn, 1); // the first frame draws, the rest are clean
        assert_eq!(seen.len() as u64, frame);
        assert_eq!(seen[0], (0, 0.25));
        assert_eq!(
            seen.iter().map(|(f, _)| *f).collect::<Vec<u64>>(),
            (0..frame).collect::<Vec<u64>>()
        );
    }

    #[test]
    fn a_tick_that_fails_ends_the_frame() {
        struct Gone;

        impl SceneController for Gone {
            fn tick(&mut self, _frame: u64, _t: f64, _dt: f64) -> Result<()> {
                bail!("the scene process exited")
            }
        }

        let mut r = Recorder::default();
        let mut s = Sink::new();
        let ctrl = slot(Gone);
        let mut lp = Loop::new(&mut r, vec![&mut s]);
        lp.controller = Some(&ctrl);
        lp.frames = 2;
        assert_eq!(
            lp.run().unwrap_err().to_string(),
            "the scene process exited"
        );
        assert_eq!(lp.drawn, 0);
    }

    #[test]
    fn a_controller_answers_for_its_own_arrays_and_the_provider_for_the_rest() {
        struct Half;

        impl SceneController for Half {
            fn provide(&mut self, name: &str, _size: ArraySize, _frame: u64) -> Option<Feed> {
                (name == "mine").then(|| Feed::Data(vec![1.0]))
            }
        }

        let mut r = Recorder::default();
        let mut s = Sink::new();
        let ctrl = slot(Half);
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.controller = Some(&ctrl);
            lp.provider = Some(Box::new(|_, _, _| Feed::Data(vec![9.0])));
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert_eq!(r.fed, vec![vec![9.0]]); // the recorder asks for "vals"
    }

    #[test]
    fn no_provider_and_no_controller_leaves_the_arrays_alone() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 1;
            lp.run().unwrap();
        }
        assert!(r.fed.is_empty()); // Renderer::feed(None) skips every upload
    }

    // errors

    #[test]
    fn an_error_ends_the_run_by_default() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.boom = true;
        let mut lp = Loop::new(&mut r, vec![&mut s]);
        lp.frames = 2;
        let e = lp.run().unwrap_err();
        assert_eq!(e.to_string(), "boom");
    }

    #[test]
    fn a_host_may_keep_the_loop_alive_through_an_error() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.boom = true;
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.stopping = Some(Box::new(after(2)));
            let noted = seen.clone();
            lp.on_error = Some(Box::new(move |e| {
                noted.lock().unwrap().push(e.to_string());
                true
            }));
            lp.run().unwrap();
        }
        assert_eq!(*seen.lock().unwrap(), vec!["boom", "boom"]);
    }

    #[test]
    fn a_sink_is_turned_off_when_a_frame_raises() {
        let mut r = Recorder::default();
        let mut s = Sink::new();
        s.boom = true;
        {
            let mut lp = Loop::new(&mut r, vec![&mut s]);
            lp.frames = 1;
            assert!(lp.run().is_err());
        }
        assert_eq!((s.cleared, s.off), (1, 1));
    }

    // the skip decision itself

    #[test]
    fn should_draw_is_the_one_rule_for_it() {
        let mut clean = Clean;
        assert!(should_draw(true, Some(&mut clean), 5, 0.1)); // a queued command
        assert!(should_draw(false, Some(&mut clean), 0, 0.0)); // the first frame
        assert!(should_draw(false, None, 5, 0.1)); // no controller
        assert!(!should_draw(false, Some(&mut clean), 5, 0.1));
    }
}
