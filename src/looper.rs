//! Port of ghee/loop.py (see that file for the reference implementation).
//!
//! One loop for everything ghee renders: image files, a pipe, a panel, KMS
//! scanout, H264, the browser console, or several of those at once. It draws a
//! frame and hands it to its sinks; what each sink needs from a frame the sink
//! declares itself (Output).
//!
//! GL is thread-affine, so the renderer belongs to the thread that runs the
//! loop: a host running the loop in a worker creates it there.
//!
//! A host drives the loop through callables instead of subclassing it: `poll`
//! drains its command queue, `idle` says how long a skipped frame waits,
//! `stopping` ends the run, `on_frame`/`on_error` publish. The controller slot
//! is read every frame, so replacing what it holds switches scenes mid-run.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Result};

use crate::engine::{Feed, Provider, Renderer};
use crate::scene::ArraySize;
use crate::stats::{monotonic, FrameRate};

/// What the loop asks of a renderer. Renderer is the one implementation; the
/// loop's own tests drive a recording stand-in through it.
pub trait RenderTarget {
    fn width(&self) -> i32;
    fn height(&self) -> i32;
    fn components(&self) -> i32;
    fn set_dt(&self, value: f64);
    fn set_time(&self, t: f64);
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
}

/// One frame as a sink sees it: the readback bytes and the shape they came back
/// at, row-major (height, width, components).
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
    /// (height, width, components), numpy's shape order.
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

/// A sink takes an RGB(A) frame and puts it somewhere: image files on disk, a
/// hardware panel, a byte stream. What each one needs from a frame it declares
/// itself, through the flags and wants()/readback() below.
pub trait Output {
    /// what a host publishes as this sink's /stats destination
    fn dest(&self) -> &str {
        ""
    }

    /// the renderer presents to it (KMS); no readback for it
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

    /// Whether this sink takes a CPU frame this iteration. False for a sink fed
    /// by the GPU (scanout, a hardware encoder) or one throttling itself.
    fn wants(&self, _frame: u64) -> bool {
        true
    }

    /// (w, h) to read the frame back at, or None for the render size read back
    /// directly. A size is a GPU downscale, which costs a blit and saves the
    /// bytes.
    fn readback(&self) -> Option<(i32, i32)> {
        None
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()>;

    /// A buffer of `shape` to render into; pooled sinks only.
    fn acquire(&mut self, _shape: (i32, i32, i32)) -> Result<OwnedFrame> {
        bail!("not a pooled sink")
    }

    /// Take back a filled buffer; pooled sinks only.
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

    /// optional hook, sinks override only if they hold resources
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Paces a render loop: ghee.output's AdaptivePacer and FixedPacer.
pub trait Pacer {
    /// Seconds to sleep after a frame: write_s = this frame's sink write,
    /// frame_s = the whole frame so far (render + write).
    fn sleep_for(&mut self, write_s: f64, frame_s: f64) -> f64;

    /// Current frame period without a new measurement: what a skipped
    /// (dirty=False) frame sleeps before re-polling.
    fn idle_period(&self) -> f64;
}

/// Hooks a scene into the render loop. Every method is optional.
pub trait SceneController {
    /// Floats for a declared array this frame, None to fall through to the
    /// host's source, or Feed::Unchanged to declare the engine's texture
    /// already holds this array.
    fn provide(&mut self, _name: &str, _size: ArraySize, _frame: u64) -> Option<Feed> {
        None
    }

    /// Per-frame hook before the draw (e.g. set live params).
    fn update(&mut self, _renderer: &mut dyn RenderTarget, _frame: u64, _t: f64) -> Result<()> {
        Ok(())
    }

    /// False = this frame would look identical to the last one, so the loop may
    /// skip it entirely (the display keeps scanning out the last flipped
    /// buffer). Default true: render every frame, as any scene animating on
    /// `time` needs. `frame` advances on skipped frames too.
    fn dirty(&mut self, _frame: u64, _t: f64) -> bool {
        true
    }
}

/// The loop reads its controller every frame, so a host switches scenes mid-run
/// by putting a new one in here.
pub type ControllerSlot = RefCell<Option<Box<dyn SceneController>>>;

/// A frame draws when something demands it: a queued command, the very first
/// frame (the sink must show something once), a scene with no controller
/// (free-running), or a controller reporting dirty.
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

/// The buffer this frame is read back into.
enum Target {
    /// no readback
    None,
    /// the sink the loop renders straight into, by index
    Pool(usize, OwnedFrame),
    /// a sink keeps the frame past write(), so this readback gets its own memory
    Fresh(OwnedFrame),
    /// the loop's one reused buffer
    Reuse,
}

pub type LoopProvider<'a> = Box<dyn FnMut(&str, ArraySize, u64) -> Feed + 'a>;
pub type OnFrame<'a> = Box<dyn FnMut(&Loop<'a>, f64) + 'a>;
pub type OnError<'a> = Box<dyn FnMut(&anyhow::Error) -> bool + 'a>;

/// Draw frames into a list of sinks until the frame budget runs out or the host
/// stops it.
///
/// frames=0 runs until `stopping`; anything else is a finite render. The loop
/// turns its sinks on and off around the run but never closes them: their
/// lifetime belongs to whoever built them.
pub struct Loop<'a> {
    pub renderer: &'a mut dyn RenderTarget,
    pub sinks: Vec<&'a mut dyn Output>,
    pub controller: Option<&'a ControllerSlot>,
    /// (name, size, frame) -> this frame's data for one declared array
    pub provider: Option<LoopProvider<'a>>,
    pub frames: u64,
    /// pin time and DT to frame * timestep
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
    /// measured frame period, the DT a scene is fed
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

    /// The destination names the frames reach (ghee.stats DEST_*).
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
        // Only a run that never ends, into sinks that all keep showing their
        // last frame, may skip a frame a controller reports clean. A file or a
        // pipe would lose it.
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
        // the t a controller is handed; unpinned it assumes a 60Hz cadence, so a
        // slower panel gets a proportionally scaled t
        let t = match self.timestep {
            Some(step) => self.frame as f64 * step,
            None => self.frame as f64 / 60.0,
        };
        if skippable && !self.should_draw_now(forced, t) {
            self.frame += 1;
            let idle = self.idle();
            sleep(idle);
            return Ok(());
        }
        if let Some(prev) = self.prev {
            self.period = Some(t0 - prev);
        }
        self.prev = Some(t0);
        // The one place DT is fed: the real frame period, or the pinned step
        // when the render has to be reproducible. Until a period has been
        // measured the engine's default stands.
        if let Some(step) = self.timestep {
            self.renderer.set_dt(step);
        } else if let Some(period) = self.period {
            self.renderer.set_dt(period);
        }
        let drew = self.draw_frame(scanout, t)?;
        if !*lit {
            // nothing was on the panel to light up before
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
    /// the clock at the end of the render, before the sinks were written.
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
            // a sink keeps the frame past write() (hands it to another thread),
            // so the next readback must not land in the same memory
            Target::Fresh(OwnedFrame::of(shape))
        };
        // The array provider for one frame: the controller first, the host's
        // source (or the placeholder) for anything it does not answer.
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
                // present() reads back through a downscale blit or not at all;
                // there is no direct full-frame path here.
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

/// The size the frame is read back at for this set of readers: None for the
/// render size read back directly, else the largest downscale asked for (a
/// smaller consumer downscales further itself).
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

/// The one sink the loop may render straight into: a pooled sink alone on the
/// frame, at the render size. Anything else takes a copy.
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

    /// One call the loop made on the renderer: (kind, readback bytes, out_size).
    type Call = (&'static str, Option<usize>, Option<(i32, i32)>);

    /// A Renderer stand-in that records what the loop asked of it.
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

    /// What one sink write saw.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Got {
        at: usize,
        len: usize,
        width: i32,
        height: i32,
    }

    /// A recording sink; the flags under test are constructor fields.
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
        // None is the render size read back directly; a smaller consumer
        // downscales the big frame itself rather than costing it detail.
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
        assert_eq!(p.lent, p.submitted); // the loop drew into the sink's own buffers
        assert_eq!(r.buffers[0], p.lent[0]);
        assert!(p.frames.is_empty()); // write() never ran: no copy anywhere
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
            // nothing overwrote a held frame: the loop kept no buffer to reuse
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

    // DT and time, fed in one place

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
    fn a_skipped_frame_still_advances_the_frame_number() {
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
        // Without a pacer or a host's own interval there is still an answer: a
        // scanout render with no --fps has neither.
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
    fn out_names_every_destination_the_frames_reach() {
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
        // the backlight came up over a drawn frame, not a blank panel
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
        // update() writes through the renderer, so the recorded times name the
        // controller that ran each frame
        assert_eq!(*r.times.borrow(), vec![1.0, 2.0, 2.0, 2.0]);
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
    fn a_sink_is_still_turned_off_when_a_frame_raises() {
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
