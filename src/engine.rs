//! Renderer drives the C engine over the ghee_* ABI.
//!
//! $GHEE_DRI selects the DRI device.

use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::cparam::create_c_parameter;
use crate::ffi;
use crate::scene::{ArraySize, ParamValue, Scene};

/// error_pass for a compile error in the scene's common block.
/// A pass error reports the pass index.
pub const COMMON_PASS: &str = "common";

/// A vecN param reaches the shader as float uniforms with these suffixes.
const VEC_SUFFIXES: [&str; 4] = ["_x", "_y", "_z", "_w"];

/// The shader holding a setup() compile error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorPass {
    Common,
    Index(usize),
}

impl ErrorPass {
    /// The wire form. A common-block error carries the name.
    /// A pass error carries the index.
    pub fn to_json(self) -> Value {
        match self {
            ErrorPass::Common => Value::from(COMMON_PASS),
            ErrorPass::Index(i) => Value::from(i),
        }
    }
}

/// What a provider returns for one declared array.
///
/// DataU8 is the same array from bytes, where the value is byte / 255. A mask
/// that starts as bytes reaches the texture without a float detour.
/// Unchanged states that the texture already holds this array. The engine then
/// skips the upload. The engine never caches or compares data.
/// setup() rebuilds a scene with empty textures. A controller pushes data again
/// after every setup().
pub enum Feed {
    Data(Vec<f32>),
    DataU8(Vec<u8>),
    Unchanged,
}

/// provider(name, size) returns this frame's data for one declared array.
pub type Provider<'a> = &'a mut dyn FnMut(&str, ArraySize) -> Feed;

/// The pixels of one asset: tightly packed rows of `components` values.
///
/// U8 uploads the bytes as they lie. F32 needs a GL with float textures.
pub enum AssetData<'a> {
    U8(&'a [u8]),
    F32(&'a [f32]),
}

/// The (width, rows) of one array upload, and its data.
///
/// A short 1xN feed uploads the values it holds, and a longer one is cut. A
/// short mask zero-pads to the full shape, because the rows must line up.
fn shape<T: Copy + Default>(size: ArraySize, data: Vec<T>) -> (c_int, c_int, Vec<T>) {
    match size {
        ArraySize::Linear(n) => (data.len().min(n as usize) as c_int, 1, data),
        ArraySize::Grid(w, h) => {
            let want = (w * h) as usize;
            let data = if data.len() < want {
                let mut buf = vec![T::default(); want];
                buf[..data.len()].copy_from_slice(&data);
                buf
            } else {
                data
            };
            (w as c_int, h as c_int, data)
        }
    }
}

fn dri_nodes(prefix: &str) -> Vec<String> {
    let Ok(entries) = fs::read_dir("/dev/dri") else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(prefix))
        .map(|n| format!("/dev/dri/{n}"))
        .collect();
    out.sort();
    out
}

fn env_dri() -> Option<String> {
    env::var("GHEE_DRI").ok().filter(|d| !d.is_empty())
}

fn dedupe(devices: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for d in devices {
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

/// The DRI nodes to try for offscreen rendering, in order.
///
/// The order is $GHEE_DRI, then the render nodes, then the primary cards.
/// The first node that creates a renderer wins.
/// Every caller reads this one list, so every caller lands on the same GPU.
/// A primary node carries DRM master, so the render nodes come first.
/// A render node has no master to contend for.
pub fn probe_dri_devices() -> Vec<String> {
    let mut devices: Vec<String> = env_dri().into_iter().collect();
    devices.push("/dev/dri/renderD128".to_string());
    devices.extend(dri_nodes("renderD"));
    devices.push("/dev/dri/card0".to_string());
    devices.extend(dri_nodes("card"));
    dedupe(devices)
}

/// The primary DRI nodes, $GHEE_DRI first.
/// Display work needs a primary node. A render node has no connector.
/// A render node cannot take DRM master.
pub fn probe_dri_cards() -> Vec<String> {
    let mut devices: Vec<String> = env_dri().into_iter().collect();
    devices.extend(dri_nodes("card"));
    let devices = dedupe(devices);
    if devices.is_empty() {
        vec!["/dev/dri/card0".to_string()]
    } else {
        devices
    }
}

/// The card node that drives a connected display, as (device, width, height).
/// None means no display is connected.
pub fn probe_display_device(dri_device: Option<&str>) -> Option<(String, i32, i32)> {
    let devices = match dri_device.filter(|d| !d.is_empty()) {
        Some(d) => vec![d.to_string()],
        None => probe_dri_cards(),
    };
    for dev in devices {
        let Ok(cdev) = CString::new(dev.as_str()) else {
            continue;
        };
        let (mut w, mut h): (c_int, c_int) = (0, 0);
        if unsafe { ffi::ghee_display_probe(cdev.as_ptr(), &mut w, &mut h) } == 0 {
            return Some((dev, w, h));
        }
    }
    None
}

/// The mode size of the connected display, as (width, height).
/// None means no card node, or no connected display.
pub fn probe_display(dri_device: Option<&str>) -> Option<(i32, i32)> {
    probe_display_device(dri_device).map(|(_, w, h)| (w, h))
}

/// The size of a packed frame in bytes.
/// bw packs 1bpp. Otherwise the frame packs RGB565.
pub fn packed_size(width: i32, height: i32, bw: bool) -> usize {
    let (w, h) = (i64::from(width), i64::from(height));
    (if bw { w * ((h + 7) / 8) } else { w * h * 2 }) as usize
}

/// Pack an RGB(A) frame to 1bpp (bw) or RGB565 without a Renderer.
/// An output sink that holds a frame and no engine calls this.
#[allow(dead_code)]
pub fn pack_frame(rgb: &[u8], width: i32, height: i32, components: i32, bw: bool) -> Vec<u8> {
    let mut out = vec![0u8; packed_size(width, height, bw)];
    pack_frame_into(rgb, width, height, components, bw, &mut out);
    out
}

/// pack_frame into a caller-held buffer. The call zeros the buffer first.
pub fn pack_frame_into(
    rgb: &[u8],
    width: i32,
    height: i32,
    components: i32,
    bw: bool,
    out: &mut [u8],
) {
    let frame = (i64::from(width) * i64::from(height) * i64::from(components)) as usize;
    let packed = packed_size(width, height, bw);
    // The packer reads three channels of every pixel. Fewer than 3 components
    // reads outside the span checked below.
    assert!(
        components >= 3,
        "packing needs 3+ components, got {components}"
    );
    // The C side reads and writes these spans. It carries no length of its own.
    assert!(
        rgb.len() >= frame && out.len() >= packed,
        "packing {width}x{height}x{components} reads {frame} bytes and writes {packed}"
    );
    out.fill(0);
    unsafe {
        ffi::ghee_pack_frame(
            width,
            height,
            components,
            c_int::from(bw),
            rgb.as_ptr(),
            out.as_mut_ptr(),
        );
    }
}

pub(crate) fn err_text(err: *mut c_char) -> String {
    if err.is_null() {
        "unknown".to_string()
    } else {
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }
}

fn or_else(s: String, fallback: &str) -> String {
    if s.is_empty() {
        fallback.to_string()
    } else {
        s
    }
}

/// The line a GL compile error names. The line is the first `:<digits>(` in
/// the error text.
fn error_line_number(err: &str) -> Option<i64> {
    let b = err.as_bytes();
    for (i, _) in err.match_indices(':') {
        let mut j = i + 1;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > i + 1 && j < b.len() && b[j] == b'(' {
            return Some(err[i + 1..j].parse::<i64>().unwrap_or(i64::MAX));
        }
    }
    None
}

/// Map a GL compile-error line back to the author's own line.
///
/// The error names a line of the assembled shader. Returns (line, in_common).
/// A line inside the scene's prepended common block counts from the start of
/// that block. Any other line is the pass's own code, less the preamble length.
fn user_line(err: &str, glsl: &str, code: &str, common: Option<&str>) -> (Option<i64>, bool) {
    let Some(line) = error_line_number(err) else {
        return (None, false);
    };
    let pos = glsl.rfind(code); // The pass code is the tail of the assembled source.
    let at = pos.map_or(0, |p| glsl[..p].matches('\n').count() as i64);
    if let Some(c) = common.filter(|c| !c.is_empty()) {
        let cpos = glsl[..pos.unwrap_or(glsl.len())].rfind(c);
        let cat = cpos.map_or(-1, |p| glsl[..p].matches('\n').count() as i64);
        if cat >= 0 && cat < line && line <= at {
            return (Some((line - cat).max(1)), true);
        }
    }
    (Some((line - at).max(1)), false)
}

fn size_repr(size: ArraySize) -> String {
    match size {
        ArraySize::Linear(n) => n.to_string(),
        ArraySize::Grid(w, h) => format!("({w}, {h})"),
    }
}

/// A multipass renderer bound to one engine.
///
/// ghee is thread-affine: create a Renderer on one thread and use it there.
/// The engine makes the GL context current at create time.
/// A worker that renders in its own thread creates its own Renderer there.
/// The raw handle keeps this type off Send.
pub struct Renderer {
    pub width: i32,
    pub height: i32,
    pub components: i32,
    pub scene: Option<Scene>,
    /// The last setup() compile error text. "" means the setup succeeded.
    pub error: String,
    /// The user-code line of the last setup() compile error.
    pub error_line: Option<i64>,
    /// The pass holding the error. Common is the scene's common block.
    pub error_pass: Option<ErrorPass>,
    #[allow(dead_code)]
    pub display: bool,
    /// pass name -> engine pass index, in the order setup() adds the passes.
    mp: Vec<(String, c_int)>,
    /// array name -> size, in declaration order.
    feeds: Vec<(String, ArraySize)>,
    rgb: Vec<u8>,
    handle: *mut ffi::GheeScene,
}

impl Renderer {
    pub fn new(
        width: i32,
        height: i32,
        components: i32,
        dri_device: Option<&str>,
        display: bool,
        rgb565: bool,
    ) -> Result<Renderer> {
        // The engine reads back GL_RGB for 3 components and GL_RGBA otherwise.
        // Any other count sizes the frame buffer under what the engine writes.
        assert!(
            components == 3 || components == 4,
            "components must be 3 or 4, got {components}"
        );
        let mut r = Renderer {
            width,
            height,
            components,
            scene: None,
            error: String::new(),
            error_line: None,
            error_pass: None,
            display,
            mp: Vec::new(),
            feeds: Vec::new(),
            rgb: vec![0u8; (i64::from(height) * i64::from(width) * i64::from(components)) as usize],
            handle: ptr::null_mut(),
        };
        let device = dri_device.filter(|d| !d.is_empty());
        if display {
            // Scanout to a KMS connector needs a card node and DRM master.
            // A render node offers neither.
            let dev = match device.map(str::to_string).or_else(env_dri) {
                Some(d) => d,
                None => probe_display_device(None)
                    .map_or_else(|| "/dev/dri/card0".to_string(), |(d, _, _)| d),
            };
            let cdev = CString::new(dev.as_str())?;
            let mut err: *mut c_char = ptr::null_mut();
            r.handle = unsafe {
                ffi::ghee_display_create(
                    width,
                    height,
                    cdev.as_ptr(),
                    c_int::from(rgb565),
                    &mut err,
                )
            };
            if r.handle.is_null() {
                bail!("ghee_display_create failed: {}", err_text(err));
            }
            return Ok(r);
        }
        let devices = match device {
            Some(d) => vec![d.to_string()],
            None => probe_dri_devices(),
        };
        let mut errs: Vec<String> = Vec::new();
        for dev in devices {
            let cdev = CString::new(dev.as_str())?;
            let mut err: *mut c_char = ptr::null_mut();
            r.handle =
                unsafe { ffi::ghee_create(width, height, components, cdev.as_ptr(), &mut err) };
            if !r.handle.is_null() {
                break;
            }
            errs.push(format!("{dev}: {}", err_text(err)));
        }
        if r.handle.is_null() {
            bail!("ghee_create failed: {}", errs.join("; "));
        }
        Ok(r)
    }

    /// The engine handle. A host passes it to the ghee_* calls it makes itself,
    /// such as the encoder calls.
    pub fn handle(&self) -> *mut ffi::GheeScene {
        self.handle
    }

    fn pass_index(&self, name: &str) -> Option<c_int> {
        self.mp.iter().find(|(n, _)| n == name).map(|(_, i)| *i)
    }

    /// Build the scene's passes.
    ///
    /// Returns "" on success. Returns an error string for a compile error, or
    /// for a scene the engine refuses.
    /// A compile error sets error_line and error_pass to the user-code line.
    /// An error in the scene's common block sets error_pass to Common.
    /// error_line then counts from the start of that block.
    pub fn setup(&mut self, scene: &Scene) -> Result<String> {
        self.error = String::new();
        self.error_line = None;
        self.error_pass = None;
        if scene.passes.is_empty() {
            self.error = "scene has no passes".to_string();
            return Ok(self.error.clone());
        }
        // An array name is one input, so it holds one size.
        // This check runs before the engine sees the scene.
        let mut seen: Vec<(&str, ArraySize)> = Vec::new();
        for p in &scene.passes {
            for a in &p.arrays {
                match seen.iter().find(|(n, _)| *n == a.name) {
                    Some((_, s)) if *s != a.size => {
                        self.error = format!(
                            "array '{}' declared at conflicting sizes ({} and {})",
                            a.name,
                            size_repr(*s),
                            size_repr(a.size)
                        );
                        return Ok(self.error.clone());
                    }
                    Some(_) => {}
                    None => seen.push((&a.name, a.size)),
                }
            }
        }
        // The output pass renders at the renderer size. Its buffer feeds the
        // readback, the display and the encoder.
        // The check reads the resolved pass list, so a filter pass appended
        // after a sized pass keeps the scene valid.
        let last = scene.output();
        if last.size.is_some() {
            self.error = format!(
                "output pass '{}' must not set a size; it renders at the renderer \
                 size ({}x{})",
                last.name, self.width, self.height
            );
            return Ok(self.error.clone());
        }
        // A channel binds to a pass by name. A name that names no pass is an
        // error. buffer=None is the unbound channel, and it reads black.
        let names: Vec<&str> = scene.passes.iter().map(|p| p.name.as_str()).collect();
        for p in &scene.passes {
            for (c, ch) in p.channels.iter().enumerate() {
                if let Some(b) = ch.buffer.as_deref().filter(|b| !names.contains(b)) {
                    self.error = format!(
                        "pass '{}': channel {c} binds to buffer '{b}', which is not a \
                         pass in this scene",
                        p.name
                    );
                    return Ok(self.error.clone());
                }
            }
        }
        let units = self.max_texture_units();
        for p in &scene.passes {
            let mut used: Vec<&str> = p.arrays.iter().map(|a| a.name.as_str()).collect();
            used.sort_unstable();
            used.dedup();
            let want = p.channels.len() + used.len();
            if want as i64 > i64::from(units) {
                self.error = format!(
                    "pass '{}' needs {want} texture units; this GPU has {units}",
                    p.name
                );
                return Ok(self.error.clone());
            }
        }
        let mut mp: Vec<(String, c_int)> = Vec::new();
        let hw_linear = self.data_texture_linear();
        let total = scene.passes.len();
        for (i, p) in scene.passes.iter().enumerate() {
            // A VC4 takes a while over a big shader, and a scene of several
            // compiles them one after another with nothing to show for it.
            crate::report::progress("compile", format!("pass {} of {total}, {}", i + 1, p.name));
            let glsl = p.glsl(Some(&scene.params), scene.common.as_deref(), hw_linear);
            let cglsl = CString::new(glsl.as_str())?;
            let mut idx = unsafe { ffi::ghee_mp_add_pass(self.handle, cglsl.as_ptr()) };
            if idx >= 0 {
                if let Some((w, h)) = p.size {
                    if !unsafe {
                        ffi::ghee_mp_set_pass_size(self.handle, idx, w as c_int, h as c_int)
                    } {
                        idx = -1;
                    }
                }
            }
            if idx < 0 {
                let err = or_else(self.last_error(), "compile error");
                unsafe { ffi::ghee_mp_abort(self.handle) };
                self.error = err.clone();
                let (line, in_common) = user_line(&err, &glsl, &p.code, scene.common.as_deref());
                self.error_line = line;
                self.error_pass = Some(if in_common {
                    ErrorPass::Common
                } else {
                    ErrorPass::Index(i)
                });
                crate::report::progress_done();
                return Ok(err);
            }
            mp.push((p.name.clone(), idx));
        }
        crate::report::progress_done();
        if unsafe { ffi::ghee_mp_commit(self.handle) } < 0 {
            self.error = or_else(self.last_error(), "commit failed");
            return Ok(self.error.clone());
        }
        self.mp = mp;
        // The output aspect reaches every pass that declares it.
        // A pass with its own size reads that size off `resolution`.
        // Such a pass cannot derive the shape of the frame.
        // A scene param of this name overwrites the value below.
        self.set_scene_param(
            "aspect",
            "float",
            &ParamValue::Float(f64::from(self.width) / f64::from(self.height.max(1))),
        )?;
        for (i, p) in scene.passes.iter().enumerate() {
            for (c, ch) in p.channels.iter().enumerate() {
                let src = match ch.buffer.as_deref() {
                    None => -1,
                    Some(b) => self.pass_index(b).ok_or_else(|| {
                        anyhow!(
                            "pass '{}': channel {c} binds to buffer '{b}', which is \
                             not a pass in this scene",
                            p.name
                        )
                    })?,
                };
                if !unsafe { ffi::ghee_mp_set_channel(self.handle, self.mp[i].1, c as c_int, src) }
                {
                    self.error = or_else(
                        self.last_error(),
                        &format!("pass '{}': bad channel", p.name),
                    );
                    return Ok(self.error.clone());
                }
            }
            for (nm, pm) in p.params.iter() {
                self.set_param(&p.name, nm, &pm.ty, &pm.value)?;
            }
        }
        // A scene param sits in every preamble, so it reaches every pass.
        for (nm, pm) in scene.params.iter() {
            self.set_scene_param(nm, &pm.ty, &pm.value)?;
        }
        self.scene = Some(scene.clone());

        let mut feeds: Vec<(String, ArraySize)> = Vec::new();
        for (i, p) in scene.passes.iter().enumerate() {
            for a in &p.arrays {
                let cname = CString::new(a.name.as_str())?;
                if unsafe { ffi::ghee_mp_declare_array(self.handle, self.mp[i].1, cname.as_ptr()) }
                    < 0
                {
                    self.error = or_else(
                        self.last_error(),
                        &format!("pass '{}': array '{}' refused", p.name, a.name),
                    );
                    return Ok(self.error.clone());
                }
                match feeds.iter_mut().find(|(n, _)| *n == a.name) {
                    Some(slot) => slot.1 = a.size,
                    None => feeds.push((a.name.clone(), a.size)),
                }
            }
        }
        self.feeds = feeds;
        Ok(String::new())
    }

    /// Render all passes. The final pass is the returned RGB(A) frame.
    /// The frame is row-major (height, width, components).
    /// `provider` fills each declared array for this frame.
    /// None leaves the arrays as they are.
    pub fn render(&mut self, provider: Option<Provider<'_>>) -> Result<&[u8]> {
        self.feed(provider)?;
        unsafe { ffi::ghee_mp_render(self.handle, self.rgb.as_mut_ptr(), 0, 0) };
        Ok(&self.rgb)
    }

    /// render() into a host-held buffer.
    /// out_size=(w, h) reads back a GPU-downscaled frame at that size.
    /// None reads back the full frame. present() downscales the same way.
    pub fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()> {
        self.feed(provider)?;
        let (pw, ph) = out_size.unwrap_or((0, 0));
        // The engine downscales only when both dimensions are positive.
        // Any other out_size falls through to the full-frame readback.
        let (w, h) = if pw > 0 && ph > 0 {
            (pw, ph)
        } else {
            (self.width, self.height)
        };
        let want = (i64::from(w) * i64::from(h) * i64::from(self.components)) as usize;
        // The readback writes this span. It carries no length of its own.
        assert!(
            out.len() >= want,
            "readback of {w}x{h}x{} needs {want} bytes, the buffer holds {}",
            self.components,
            out.len()
        );
        unsafe { ffi::ghee_mp_render(self.handle, out.as_mut_ptr(), pw, ph) };
        Ok(())
    }

    /// render() reading back a GPU-downscaled frame into a fresh buffer.
    #[allow(dead_code)]
    pub fn render_scaled(
        &mut self,
        provider: Option<Provider<'_>>,
        out_size: (i32, i32),
    ) -> Result<Vec<u8>> {
        let (w, h) = out_size;
        let mut out =
            vec![0u8; (i64::from(w) * i64::from(h) * i64::from(self.components)) as usize];
        self.render_into(provider, &mut out, Some(out_size))?;
        Ok(out)
    }

    /// Render all passes with no readback.
    pub fn draw(&mut self, provider: Option<Provider<'_>>) -> Result<()> {
        self.feed(provider)?;
        unsafe { ffi::ghee_mp_render(self.handle, ptr::null_mut(), 0, 0) };
        Ok(())
    }

    /// Upload each declared array for this frame, once per unique name.
    ///
    /// A render uploads what the provider returns. An Unchanged answer states
    /// that the texture already holds this array, and the upload is skipped.
    /// The engine never caches or compares data.
    fn feed(&self, provider: Option<Provider<'_>>) -> Result<()> {
        let Some(provider) = provider else {
            return Ok(());
        };
        for (name, size) in &self.feeds {
            match provider(name, *size) {
                Feed::Unchanged => {}
                Feed::Data(data) => {
                    let (w, h, data) = shape(*size, data);
                    let cname = CString::new(name.as_str())?;
                    unsafe {
                        ffi::ghee_mp_set_array(self.handle, cname.as_ptr(), data.as_ptr(), w, h);
                    }
                }
                Feed::DataU8(data) => {
                    let (w, h, data) = shape(*size, data);
                    let cname = CString::new(name.as_str())?;
                    unsafe {
                        ffi::ghee_mp_set_array_u8(self.handle, cname.as_ptr(), data.as_ptr(), w, h);
                    }
                }
            }
        }
        Ok(())
    }

    /// Hand one named image asset to the engine.
    ///
    /// Every pass that reads the name gets it as a sampler uniform `name`,
    /// plus a vec2 `name_size` of (width, height). An asset outlives the
    /// scene: setup() keeps it, and the same name again replaces its pixels.
    /// `linear` samples with GL_LINEAR, f32 pixels fall back to nearest
    /// without data_texture_linear().
    pub fn set_asset(
        &self,
        name: &str,
        width: u32,
        height: u32,
        components: u32,
        linear: bool,
        data: AssetData<'_>,
    ) -> Result<()> {
        let want = width as usize * height as usize * components as usize;
        let (ptr, len, is_f32) = match data {
            AssetData::U8(d) => (d.as_ptr().cast::<c_void>(), d.len(), false),
            AssetData::F32(d) => (d.as_ptr().cast::<c_void>(), d.len(), true),
        };
        // The engine reads this span. It carries no length of its own.
        assert!(
            len >= want,
            "asset '{name}' at {width}x{height}x{components} needs {want} values, \
             the data holds {len}"
        );
        let cname = CString::new(name)?;
        unsafe {
            ffi::ghee_mp_set_asset(
                self.handle,
                cname.as_ptr(),
                ptr,
                width as c_int,
                height as c_int,
                components as c_int,
                c_int::from(is_f32),
                c_int::from(linear),
            );
        }
        Ok(())
    }

    /// Display mode only. This call feeds the arrays and renders the passes.
    /// It then scans the final buffer out to the connected display.
    /// `preview` also reads back a GPU-downscaled frame.
    /// None skips that readback.
    pub fn present(
        &mut self,
        provider: Option<Provider<'_>>,
        preview: Option<(&mut [u8], (i32, i32))>,
    ) -> Result<()> {
        self.feed(provider)?;
        let (buf, pw, ph) = match preview {
            Some((out, (pw, ph))) => {
                let want = (i64::from(pw) * i64::from(ph) * i64::from(self.components)) as usize;
                assert!(
                    out.len() >= want,
                    "preview of {pw}x{ph}x{} needs {want} bytes, the buffer holds {}",
                    self.components,
                    out.len()
                );
                (out.as_mut_ptr(), pw, ph)
            }
            None => (ptr::null_mut(), 0, 0),
        };
        unsafe { ffi::ghee_mp_present(self.handle, buf, pw, ph) };
        Ok(())
    }

    /// Set one uniform on a pass live, with no recompile.
    ///
    /// Returns false when the pass or the uniform is absent.
    /// A "color" param is [r, g, b]. It sets the float uniforms name_r, name_g
    /// and name_b, because the engine has no color type.
    /// A "vecK" param is K floats. It sets the float uniforms name_x, name_y,
    /// name_z and name_w.
    /// A "string" param is host-side. The scene's controller consumes it, so
    /// this call sets nothing.
    pub fn set_param(
        &self,
        pass_name: &str,
        name: &str,
        ptype: &str,
        value: &ParamValue,
    ) -> Result<bool> {
        if ptype == "string" {
            return Ok(false);
        }
        let Some(idx) = self.pass_index(pass_name) else {
            return Ok(false);
        };
        if ptype == "color" {
            let ParamValue::Color(rgb) = value else {
                bail!("color param '{name}' needs three numbers [r, g, b]");
            };
            let oks = ["_r", "_g", "_b"]
                .iter()
                .zip(rgb)
                .map(|(suffix, comp)| {
                    self.set_uniform(idx, &format!("{name}{suffix}"), "float", Value::from(*comp))
                })
                .collect::<Result<Vec<bool>>>()?;
            return Ok(oks.into_iter().all(|o| o));
        }
        if matches!(ptype, "vec2" | "vec3" | "vec4") {
            let ParamValue::Vec(components) = value else {
                bail!("{ptype} param '{name}' needs numbers");
            };
            let oks = VEC_SUFFIXES
                .iter()
                .zip(components)
                .map(|(suffix, comp)| {
                    self.set_uniform(idx, &format!("{name}{suffix}"), "float", Value::from(*comp))
                })
                .collect::<Result<Vec<bool>>>()?;
            return Ok(oks.into_iter().all(|o| o));
        }
        self.set_uniform(idx, name, ptype, value.to_json())
    }

    fn set_uniform(&self, idx: c_int, name: &str, ptype: &str, value: Value) -> Result<bool> {
        let mut cp = create_c_parameter(ptype, &value)?;
        cp.set_c_name(name)?;
        Ok(unsafe { ffi::ghee_mp_set_uniform(self.handle, idx, cp.as_ptr()) })
    }

    /// Set one scene-scoped uniform on every pass live, with no recompile.
    ///
    /// A scene param sits in every pass's preamble, so the set reaches every
    /// pass. A pass that never reads the param holds an inactive uniform, and
    /// the engine skips it.
    /// Returns true when at least one pass takes the value.
    pub fn set_scene_param(&self, name: &str, ptype: &str, value: &ParamValue) -> Result<bool> {
        let oks = self
            .mp
            .iter()
            .map(|(pn, _)| self.set_param(pn, name, ptype, value))
            .collect::<Result<Vec<bool>>>()?;
        Ok(oks.into_iter().any(|o| o))
    }

    /// Pack an RGB(A) frame to 1bpp (bw) or RGB565.
    #[allow(dead_code)]
    pub fn pack(&self, rgb: &[u8], bw: bool) -> Vec<u8> {
        pack_frame(rgb, self.width, self.height, self.components, bw)
    }

    /// pack() into a caller-held buffer. The call zeros the buffer first.
    #[allow(dead_code)]
    pub fn pack_into(&self, rgb: &[u8], bw: bool, out: &mut [u8]) {
        pack_frame_into(rgb, self.width, self.height, self.components, bw, out);
    }

    /// The refresh rate of the connected display mode, in Hz.
    /// A renderer outside display mode returns 0.
    pub fn display_vrefresh(&self) -> i32 {
        unsafe { ffi::ghee_display_vrefresh(self.handle) }
    }

    /// Blank the display, or light it again.
    /// Blanking stops the CRTC driving the connector, and the kernel driver
    /// takes its panel from there: the display goes off and the backlight
    /// with it. A blanked display draws nothing, so present returns at once
    /// and the caller paces its own loop.
    /// A renderer that drives no display returns an error.
    pub fn blank(&self, on: bool) -> Result<()> {
        if unsafe { ffi::ghee_display_blank(self.handle, i32::from(on)) } != 0 {
            bail!("{}", self.last_error());
        }
        Ok(())
    }

    /// True while the display is blanked.
    pub fn blanked(&self) -> bool {
        unsafe { ffi::ghee_display_blanked(self.handle) != 0 }
    }

    /// Enable per-frame draw and readback timing.
    /// The timing adds a glFinish. Timing off costs nothing.
    #[allow(dead_code)]
    pub fn set_profiling(&self, on: bool) {
        unsafe { ffi::ghee_set_profiling(self.handle, on) };
    }

    /// Pin the `time` uniform to a fixed value, for a reproducible render.
    /// The default is the monotonic clock since the engine start.
    /// None returns the uniform to that clock.
    pub fn set_time(&self, t: Option<f64>) {
        match t {
            None => unsafe { ffi::ghee_clear_time(self.handle) },
            Some(t) => unsafe { ffi::ghee_set_time(self.handle, t as f32) },
        }
    }

    /// Set the `DT` uniform, this frame's advance in seconds of scene time.
    /// DT rides the draw like `time`: one store here, one upload per pass.
    /// The default is 1/60 until a caller feeds a measured period.
    pub fn set_dt(&self, value: f64) {
        unsafe { ffi::ghee_set_dt(self.handle, value as f32) };
    }

    /// The fragment texture units this GPU exposes. VC4 exposes 8.
    /// A pass fits its channels and its array inputs in this budget.
    pub fn max_texture_units(&self) -> i32 {
        unsafe { ffi::ghee_max_texture_units(self.handle) }
    }

    /// True when this GPU linear-filters the float textures that carry arrays.
    /// V3D returns false. It has no OES_texture_float_linear.
    /// A filtered fetch there returns the nearest texel.
    pub fn data_texture_linear(&self) -> bool {
        unsafe { ffi::ghee_data_texture_linear(self.handle) }
    }

    /// The last frame's split, as (draw_ms, read_ms).
    /// draw_ms is the GPU render of every pass.
    /// read_ms is the glReadPixels readback.
    /// Both stay 0 until a frame renders with profiling on.
    #[allow(dead_code)]
    pub fn timings(&self) -> (f64, f64) {
        let (mut draw, mut read): (c_double, c_double) = (0.0, 0.0);
        unsafe { ffi::ghee_get_timings(self.handle, &mut draw, &mut read) };
        (draw, read)
    }

    /// The GL this renderer runs on: GL_RENDERER and GL_VERSION in one string.
    pub fn renderer(&self) -> String {
        let r = unsafe { ffi::ghee_renderer(self.handle) };
        if r.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(r) }.to_string_lossy().into_owned()
        }
    }

    pub fn last_error(&self) -> String {
        let e = unsafe { ffi::ghee_last_error(self.handle) };
        if e.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        unsafe { ffi::ghee_mp_reset(self.handle) };
        self.mp = Vec::new();
        self.feeds = Vec::new();
    }

    pub fn close(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::ghee_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::os::unix::fs::MetadataExt;
    use std::sync::{Mutex, MutexGuard};
    use std::thread::sleep;
    use std::time::Duration;

    use serde_json::json;

    use super::*;

    const W: usize = 64;
    const H: usize = 48;
    const COMPONENTS: usize = 4;

    const FLAT: &str = "void main(){ gl_FragColor = vec4(0.5, 0.25, 0.75, 1.0); }";
    const BROKEN: &str = "void main(){ gl_FragColor = no_such_symbol; }";
    const RAW: &str =
        "#version 100\nprecision mediump float;\nvoid main(){ gl_FragColor = vec4(1.0); }";

    /// A GL device takes one Renderer at a time, and $GHEE_DRI is process-wide.
    /// A test that touches either one holds this lock.
    static GPU: Mutex<()> = Mutex::new(());

    fn lock() -> MutexGuard<'static, ()> {
        GPU.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One renderer for these tests, with the device lock.
    /// None means the host has no GL device.
    fn gpu() -> Option<(MutexGuard<'static, ()>, Renderer)> {
        let guard = lock();
        let mut errs = Vec::new();
        for dev in probe_dri_devices() {
            match Renderer::new(
                W as i32,
                H as i32,
                COMPONENTS as i32,
                Some(&dev),
                false,
                false,
            ) {
                Ok(r) => return Some((guard, r)),
                Err(e) => errs.push(format!("{dev}: {e}")),
            }
        }
        eprintln!("no GL device for rendering: {}", errs.join("; "));
        None
    }

    fn scene_of(passes: Value) -> Scene {
        Scene::from_dict(&json!({"version": 1, "passes": passes}), None, None).unwrap()
    }

    fn flat_scene() -> Scene {
        scene_of(json!([{"name": "main", "code": FLAT}]))
    }

    /// One channel of every pixel, 0..255 as floats.
    fn channel(frame: &[u8], c: usize) -> Vec<f64> {
        frame
            .iter()
            .skip(c)
            .step_by(COMPONENTS)
            .map(|v| f64::from(*v))
            .collect()
    }

    fn mean(vals: &[f64]) -> f64 {
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    fn px(frame: &[u8], w: usize, x: usize, y: usize, c: usize) -> f64 {
        f64::from(frame[(y * w + x) * COMPONENTS + c])
    }

    #[test]
    fn probe_dri_devices_puts_the_override_first() {
        let _g = lock();
        env::set_var("GHEE_DRI", "/dev/dri/card9");
        let devs = probe_dri_devices();
        env::remove_var("GHEE_DRI");
        assert_eq!(devs[0], "/dev/dri/card9");
        assert_eq!(devs.len(), devs.iter().collect::<HashSet<_>>().len());
        assert!(devs.iter().any(|d| d == "/dev/dri/card0"));
    }

    #[test]
    fn probe_dri_devices_prefers_render_nodes() {
        // A primary node carries DRM master. A second process on a Pi 5 card0
        // renders black. A render node has no master to contend for.
        let _g = lock();
        env::remove_var("GHEE_DRI");
        let devs = probe_dri_devices();
        let cards: Vec<usize> = (0..devs.len())
            .filter(|i| devs[*i].contains("/card"))
            .collect();
        let renders: Vec<usize> = (0..devs.len())
            .filter(|i| devs[*i].contains("/renderD"))
            .collect();
        assert_eq!(devs[0], "/dev/dri/renderD128");
        assert!(!cards.is_empty() && !renders.is_empty());
        assert!(renders.iter().max() < cards.iter().min());
    }

    #[test]
    fn probe_dri_cards_are_primary_nodes() {
        let _g = lock();
        env::set_var("GHEE_DRI", "/dev/dri/card9");
        let cards = probe_dri_cards();
        env::remove_var("GHEE_DRI");
        assert_eq!(cards[0], "/dev/dri/card9");
        assert!(probe_dri_cards().iter().all(|d| !d.contains("/renderD")));
    }

    #[test]
    fn probe_missing_device_is_none() {
        let _g = lock();
        assert_eq!(probe_display(Some("/dev/dri/does-not-exist")), None);
    }

    #[test]
    fn probe_display_device_names_a_probed_card_and_its_size() {
        // A headless host probes to None.
        // Any other host reports a positive width and height on a card node.
        let _g = lock();
        env::remove_var("GHEE_DRI");
        let Some((dev, w, h)) = probe_display_device(None) else {
            return;
        };
        // The connectors do not always sit on card0. A Pi 5 puts them on card1.
        assert!(probe_dri_cards().contains(&dev));
        assert!(w > 0 && h > 0);
        assert_eq!(probe_display(None), Some((w, h)));
    }

    #[test]
    fn display_create_fails_once_without_drm_master() {
        let _g = lock();
        if fs::metadata("/proc/self").is_ok_and(|m| m.uid() == 0) {
            return; // As root this test takes the session's card.
        }
        let Some((dev, w, h)) = probe_display_device(None) else {
            return;
        };
        let before_the_master_attempt = ["cannot open DRI device", "eglInitialize failed"];
        // An idle host holds no DRM master. This test takes the master itself.
        let holder = Renderer::new(w, h, 4, Some(&dev), true, false);
        let msg = match &holder {
            Err(e) => e.to_string(),
            Ok(_) => match Renderer::new(w, h, 4, Some(&dev), true, false) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("{dev}: two display renderers at once"),
            },
        };
        if before_the_master_attempt.iter().any(|s| msg.contains(s)) {
            return;
        }
        assert!(msg.contains("DRM master"), "{msg}");
    }

    #[test]
    fn an_unfinished_flip_on_a_slow_board_is_still_waited_for() {
        // The kernel holds a page flip until the GPU finishes the frame. The
        // wait scales with the render. The panel refresh does not bound it.
        // A bound near one frame time leaves a slow board's flip pending.
        // The engine refuses every submit while a flip stays pending.
        let src = fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/ghee/_engine/engine.c"
        ))
        .unwrap();
        let bound = src
            .split("#define FLIP_STUCK_MS ")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap();
        assert!(bound >= 1000); // 1000 ms is slower than any frame, so only a stuck flip hits it.

        let present = src.split("void ghee_mp_present").nth(1).unwrap();
        let unreaped = present
            .split("!wait_flip(g)")
            .nth(1)
            .unwrap()
            .split("} else")
            .next()
            .unwrap();
        assert!(unreaped.contains("gbm_surface_release_buffer(g->gbm_surface, bo)"));
        assert!(!unreaped.contains("drmModePageFlip"));

        let flipped = present
            .split("if (!drmModePageFlip")
            .nth(1)
            .unwrap()
            .split("} else")
            .next()
            .unwrap();
        assert!(flipped.contains("g->flip_failed = false;"));
    }

    #[test]
    fn a_compile_error_line_maps_back_to_the_authors_own_line() {
        let glsl = "#version 100\nprecision mediump float;\nfloat sat(float x){ return x; }\n\nvoid main(){\n  nope();\n}";
        let common = "float sat(float x){ return x; }";
        let code = "void main(){\n  nope();\n}";
        // Line 6 of the assembled source is line 2 of the pass.
        assert_eq!(
            user_line("0:6(3): error: no function nope", glsl, code, Some(common)),
            (Some(2), false)
        );
        // Line 3 sits in the common block. It counts from the block start.
        assert_eq!(
            user_line("0:3(3): error: bad", glsl, code, Some(common)),
            (Some(1), true)
        );
        // With no common block the same line is the pass's, clamped at 1.
        assert_eq!(
            user_line("0:3(3): error: bad", glsl, code, None),
            (Some(1), false)
        );
        // An error with no line attributes nothing.
        assert_eq!(user_line("out of memory", glsl, code, None), (None, false));
    }

    #[test]
    fn packed_size_is_1bpp_by_column_or_rgb565() {
        assert_eq!(packed_size(128, 32, true), 128 * 4);
        assert_eq!(packed_size(128, 33, true), 128 * 5); // A 33rd row fills a partial byte column.
        assert_eq!(packed_size(128, 32, false), 128 * 32 * 2);
    }

    #[test]
    fn setup_rejects_a_scene_with_no_passes() {
        let Some((_g, mut r)) = gpu() else { return };
        let empty = Scene::of(Vec::new()).unwrap();
        assert!(r.setup(&empty).unwrap().contains("no passes"));
    }

    #[test]
    fn channel_naming_no_pass_is_rejected() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([
            {"name": "a", "code": FLAT},
            {"name": "b", "channels": [{"buffer": "a"}, {"buffer": "typo"}], "code": FLAT},
        ]));
        assert_eq!(
            r.setup(&sc).unwrap(),
            "pass 'b': channel 1 binds to buffer 'typo', which is not a pass in this scene"
        );
    }

    #[test]
    fn channel_bound_to_nothing_is_allowed() {
        // buffer=None is the unbound channel. The scene stays valid.
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([
            {"name": "a", "channels": [{"buffer": null}], "code": FLAT},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
    }

    #[test]
    fn setup_rejects_a_scene_in_a_fixed_order() {
        let Some((_g, mut r)) = gpu() else { return };
        let sized = json!([{"name": "a", "code": FLAT, "size": [8, 8]}]);
        assert_eq!(
            r.setup(&scene_of(sized)).unwrap(),
            "output pass 'a' must not set a size; it renders at the renderer size (64x48)"
        );
        let clash = json!([
            {"name": "a", "code": FLAT, "arrays": [{"name": "v", "size": 4}]},
            {"name": "b", "code": FLAT, "arrays": [{"name": "v", "size": [2, 2]}]},
        ]);
        assert_eq!(
            r.setup(&scene_of(clash)).unwrap(),
            "array 'v' declared at conflicting sizes (4 and (2, 2))"
        );
        // The size clash outranks the sized output pass below it.
        let both = json!([
            {"name": "a", "code": FLAT, "arrays": [{"name": "v", "size": 4}]},
            {"name": "b", "code": FLAT, "size": [8, 8],
             "arrays": [{"name": "v", "size": [2, 2]}]},
        ]);
        assert!(r.setup(&scene_of(both)).unwrap().starts_with("array 'v'"));
        // The sized output pass outranks the bad channel below it.
        let sized_and_bad = json!([
            {"name": "a", "code": FLAT},
            {"name": "b", "code": FLAT, "size": [8, 8], "channels": [{"buffer": "typo"}]},
        ]);
        assert!(r
            .setup(&scene_of(sized_and_bad))
            .unwrap()
            .starts_with("output pass 'b'"));
    }

    #[test]
    fn unbound_channel_reads_black() {
        // 'leak' leaves a white texture bound on unit 0.
        // Channel 0 of 'main' is unbound, and it reads black.
        let Some((_g, mut r)) = gpu() else { return };
        let sample = "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }";
        let sc = scene_of(json!([
            {"name": "bright", "code": "void main(){ gl_FragColor = vec4(1.0); }"},
            {"name": "leak", "channels": [{"buffer": "bright"}], "code": sample},
            {"name": "main", "channels": [{"buffer": null}], "code": sample},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        for c in 0..3 {
            assert!(channel(&f, c).iter().fold(0.0, |a: f64, b| a.max(*b)) <= 2.0);
        }
    }

    /// A sized pass renders at its own size, so its `resolution` is not the
    /// frame size. `aspect` carries the shape of the output.
    #[test]
    fn every_pass_reads_the_output_aspect() {
        let Some((_g, mut r)) = gpu() else { return };
        let red = "uniform float aspect;\nvoid main(){ \
                   gl_FragColor = vec4(aspect * 0.5, 0.0, 0.0, 1.0); }";
        let through = "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }";
        let sc = scene_of(json!([
            {"name": "small", "code": red, "size": [8, 8]},
            {"name": "main", "code": through, "channels": [{"buffer": "small"}]},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        let want = 255.0 * (W as f64 / H as f64) * 0.5;
        assert!((mean(&channel(&f, 0)) - want).abs() <= 2.0, "{}", f[0]);
        // A scene that declares the name keeps its own value.
        let mine = Scene::from_dict(
            &json!({"version": 1,
                    "params": {"aspect": {"type": "float", "value": 0.5}},
                    "passes": [{"name": "main", "code":
                                "void main(){ gl_FragColor = \
                                 vec4(aspect, 0.0, 0.0, 1.0); }"}]}),
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.setup(&mine).unwrap(), "");
        let f = r.render(None).unwrap().to_vec();
        assert!((mean(&channel(&f, 0)) - 128.0).abs() <= 2.0);
    }

    #[test]
    fn compile_error_leaves_the_running_scene() {
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();

        let bad = scene_of(json!([
            {"name": "main", "code": FLAT},
            {"name": "broken", "channels": [{"buffer": "main"}], "code": BROKEN},
        ]));
        assert!(!r.setup(&bad).unwrap().is_empty());
        assert_eq!(r.error_pass, Some(ErrorPass::Index(1)));
        assert_eq!(r.render(None).unwrap(), before);
    }

    #[test]
    fn unallocatable_pass_buffer_fails_loudly() {
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();

        let raw = CString::new(RAW).unwrap();
        let idx = unsafe { ffi::ghee_mp_add_pass(r.handle(), raw.as_ptr()) };
        assert_eq!(idx, 0);
        assert!(!unsafe { ffi::ghee_mp_set_pass_size(r.handle(), idx, 1 << 17, 1 << 17) });
        assert!(r.last_error().contains("pass buffer failed"));
        unsafe { ffi::ghee_mp_abort(r.handle()) };
        assert_eq!(r.render(None).unwrap(), before);
    }

    #[test]
    fn render_on_an_empty_scene_reports_instead_of_indexing() {
        let Some((_g, mut r)) = gpu() else { return };
        r.reset();
        r.render(None).unwrap();
        assert!(r.last_error().contains("no passes"));
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
    }

    #[test]
    fn a_degenerate_out_size_cannot_undersize_the_readback_buffer() {
        // The engine downscales only for positive dimensions.
        // A zero or negative dimension falls through to the full-frame readback.
        // The buffer check sizes for that readback.
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        for bad in [(0, 6), (8, 0), (-4, -6)] {
            let mut out: Vec<u8> = vec![0u8; 8 * 6 * COMPONENTS];
            let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = r.render_into(None, &mut out, Some(bad));
            }));
            let msg = *hit.unwrap_err().downcast::<String>().unwrap();
            assert!(
                msg.contains(&format!("readback of {W}x{H}x{COMPONENTS}")),
                "{msg}"
            );
        }
        // A full-size buffer satisfies the fall-through readback.
        let mut out = vec![0u8; H * W * COMPONENTS];
        r.render_into(None, &mut out, Some((0, 6))).unwrap();
        assert_eq!(out, r.render(None).unwrap());
    }

    #[test]
    fn downscaled_readback_matches_the_full_frame() {
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let full = r.render(None).unwrap().to_vec();
        let small = r.render_scaled(None, (8, 6)).unwrap();
        assert_eq!(small.len(), 6 * 8 * COMPONENTS);
        for c in 0..3 {
            let diff = mean(&channel(&small, c)) as i64 - mean(&channel(&full, c)) as i64;
            assert!(diff.abs() <= 2);
        }
        let mut buf = vec![0u8; 6 * 8 * COMPONENTS];
        r.render_into(None, &mut buf, Some((8, 6))).unwrap();
        assert_eq!(buf, small);
    }

    #[test]
    fn render_into_a_caller_buffer() {
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let full = r.render(None).unwrap().to_vec();
        let mut buf = vec![0u8; H * W * COMPONENTS];
        r.render_into(None, &mut buf, None).unwrap();
        assert_eq!(buf, full);
    }

    #[test]
    fn draw_renders_with_no_readback() {
        // The scene feeds back: each frame adds to the last.
        // A draw() that renders shows up in the next readback.
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([{
            "name": "main",
            "channels": [{"buffer": "main"}],
            "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) \
                     + vec4(0.1, 0.0, 0.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let one = r.render(None).unwrap().to_vec();
        assert_eq!(r.setup(&sc).unwrap(), ""); // setup() gives fresh buffers, cleared to black.
        r.draw(None).unwrap();
        let two = r.render(None).unwrap().to_vec();
        assert!(mean(&channel(&two, 0)) as i64 > mean(&channel(&one, 0)) as i64);
    }

    #[test]
    fn one_upload_per_array_name() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([
            {
                "name": "a",
                "arrays": [{"name": "shared", "size": 4}],
                "code": "void main(){ gl_FragColor = vec4(shared_at(0), 0.0, 0.0, 1.0); }",
            },
            {
                "name": "b",
                "channels": [{"buffer": "a"}],
                "arrays": [{"name": "shared", "size": 4}],
                "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) \
                         + vec4(0.0, shared_at(1), 0.0, 0.0); }",
            },
            {
                "name": "c",
                "channels": [{"buffer": "b"}],
                "arrays": [{"name": "shared", "size": 4}],
                "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) \
                         + vec4(0.0, 0.0, shared_at(2), 0.0); }",
            },
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut calls: Vec<String> = Vec::new();
        let f = {
            let mut provider = |name: &str, size: ArraySize| {
                calls.push(name.to_string());
                let n = match size {
                    ArraySize::Linear(n) => n as usize,
                    ArraySize::Grid(w, h) => (w * h) as usize,
                };
                Feed::Data(vec![0.5; n])
            };
            r.render(Some(&mut provider)).unwrap().to_vec()
        };
        assert_eq!(calls, ["shared"]); // One upload feeds the three passes that read it.
        for c in 0..3 {
            assert!((mean(&channel(&f, c)) as i64 - 128).abs() <= 2);
        }
    }

    #[test]
    fn texture_unit_budget_rejects_a_greedy_pass() {
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();

        // The pass asks for one channel past this GPU's budget, plus an array.
        let units = r.max_texture_units() as usize;
        let channels: Vec<Value> = (0..units).map(|_| json!({"buffer": "a"})).collect();
        let greedy = scene_of(json!([
            {"name": "a", "code": FLAT},
            {
                "name": "main",
                "channels": channels,
                "arrays": [{"name": "v", "size": 4}],
                "code": "void main(){ gl_FragColor = vec4(v_at(0)); }",
            },
        ]));
        assert_eq!(
            r.setup(&greedy).unwrap(),
            format!(
                "pass 'main' needs {} texture units; this GPU has {units}",
                units + 1
            )
        );
        assert_eq!(r.render(None).unwrap(), before);
    }

    #[test]
    fn channel_source_pass_is_bound_checked() {
        let Some((_g, mut r)) = gpu() else { return };
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        assert!(!unsafe { ffi::ghee_mp_set_channel(r.handle(), 0, 0, 7) });
        assert!(unsafe { ffi::ghee_mp_set_channel(r.handle(), 0, 0, -1) });
        assert!(unsafe { ffi::ghee_mp_set_channel(r.handle(), 0, 0, 0) });
    }

    #[test]
    fn set_time_none_returns_to_the_wall_clock() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([{
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(mod(time * 37.0, 1.0), 0.0, 0.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(1.0));
        let pinned = r.render(None).unwrap().to_vec();
        assert_eq!(r.render(None).unwrap(), pinned);
        r.set_time(None);
        let a = r.render(None).unwrap().to_vec();
        sleep(Duration::from_millis(20));
        assert_ne!(r.render(None).unwrap(), a);
    }

    #[test]
    fn set_dt_reaches_every_pass() {
        // DT rides the draw like time, so one store lands in every pass.
        // Red carries the first pass's DT through a channel.
        // Green carries the second pass's DT.
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([
            {"name": "src", "code": "void main(){ gl_FragColor = vec4(DT, 0.0, 0.0, 1.0); }"},
            {
                "name": "main",
                "code": "void main(){ gl_FragColor = vec4(\
                         texture2D(iChannel0, v_texCoord).r, DT, 0.0, 1.0); }",
                "channels": [{"buffer": "src"}],
            },
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let near = |frame: &[u8], want: f64| {
            (0..2).all(|c| channel(frame, c).iter().all(|v| (v - want).abs() <= 1.0))
        };
        let f = r.render(None).unwrap().to_vec();
        assert!(near(&f, (255.0f64 / 60.0).round())); // 1/60 is the engine default.
        r.set_dt(0.5);
        let f = r.render(None).unwrap().to_vec();
        assert!(near(&f, 128.0));
        r.set_dt(0.25);
        let f = r.render(None).unwrap().to_vec();
        assert!(near(&f, 64.0));

        r.set_dt(0.5);
        let same = r.render(None).unwrap().to_vec();
        assert_eq!(r.render(None).unwrap(), same);
    }

    #[test]
    fn closing_a_second_renderer_leaves_the_first_alive() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = flat_scene();
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();
        // A GPU node can refuse a second renderer. The test then stops.
        let Ok(mut other) = Renderer::new(16, 16, COMPONENTS as i32, None, false, false) else {
            return;
        };
        assert_eq!(other.setup(&sc).unwrap(), "");
        other.render(None).unwrap();
        other.close();
        assert_eq!(r.render(None).unwrap(), before);
    }

    #[test]
    fn a_short_2d_feed_is_zero_padded_and_a_long_1d_one_is_cut() {
        let Some((_g, mut r)) = gpu() else { return };
        // The shader reads the second half of mask.
        // The provider feeds vals past its declared size.
        let sc = scene_of(json!([{
            "name": "main",
            "arrays": [{"name": "mask", "size": [2, 2]}, {"name": "vals", "size": 2}],
            "code": "void main(){ gl_FragColor = vec4(mask_sample2(vec2(0.75, 0.75)), \
                     vals_at(1), 0.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut provider = |name: &str, _size: ArraySize| match name {
            // This feed holds half a 2x2 mask. The second row zero-pads.
            "mask" => Feed::Data(vec![1.0, 1.0]),
            _ => Feed::Data(vec![0.0, 0.5, 1.0, 1.0]),
        };
        let f = r.render(Some(&mut provider)).unwrap().to_vec();
        assert!(mean(&channel(&f, 0)) < 2.0);
        assert!((mean(&channel(&f, 1)) - 128.0).abs() <= 2.0);
    }

    #[test]
    fn an_unchanged_feed_leaves_the_texture_alone() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([{
            "name": "main",
            "arrays": [{"name": "vals", "size": 4}],
            "code": "void main(){ gl_FragColor = vec4(vals_at(0), 0.0, 0.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut half = |_: &str, _: ArraySize| Feed::Data(vec![0.5; 4]);
        let fed = r.render(Some(&mut half)).unwrap().to_vec();
        let mut hold = |_: &str, _: ArraySize| Feed::Unchanged;
        assert_eq!(r.render(Some(&mut hold)).unwrap(), fed);
        assert!((mean(&channel(&fed, 0)) - 128.0).abs() <= 2.0);
    }

    #[test]
    fn an_error_in_the_common_reports_the_common_line() {
        let Some((_g, mut r)) = gpu() else { return };
        let bad = "float sat(float x){ return clamp(x, 0.0, 1.0); }\nfloat b(){ return nope(); }";
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "common": {"code": bad},
                "passes": [{
                    "name": "main",
                    "code": "void main(){ gl_FragColor = vec4(sat(v_texCoord.x), 0.0, 0.0, 1.0); }",
                }],
            }),
            None,
            None,
        )
        .unwrap();
        assert!(!r.setup(&sc).unwrap().is_empty());
        assert_eq!(r.error_pass, Some(ErrorPass::Common));
        assert_eq!(r.error_line, Some(2)); // The author wrote nope() on line 2.
    }

    #[test]
    fn an_error_in_a_pass_still_reports_that_pass() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "common": {"code": "float sat(float x){ return clamp(x, 0.0, 1.0); }"},
                "passes": [
                    {"name": "a", "code": FLAT},
                    {"name": "b", "code": "void main(){\n  float v = 1.0;\n  nope(v);\n}"},
                ],
            }),
            None,
            None,
        )
        .unwrap();
        assert!(!r.setup(&sc).unwrap().is_empty());
        assert_eq!(r.error_pass, Some(ErrorPass::Index(1)));
        assert_eq!(r.error_line, Some(3));
        // A good compile clears the attribution.
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        assert!(r.error_pass.is_none() && r.error_line.is_none());
    }

    // Data arrays reach the shader as 1xN textures.

    const N: usize = 64; // 64 is past the cap of about 48 a VC4 uniform array holds.

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
    }

    fn array_scene(code: &str, name: &str, size: Value) -> Scene {
        scene_of(json!([{
            "name": "main",
            "code": code,
            "arrays": [{"name": name, "size": size}],
        }]))
    }

    #[test]
    fn array_samples_correctly() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = array_scene(
            "void main(){ vec2 uv=gl_FragCoord.xy/resolution; \
             gl_FragColor=vec4(vec3(vals_at(int(uv.x*float(vals_size-1)))),1.0); }",
            "vals",
            json!(N),
        );
        assert_eq!(r.setup(&sc).unwrap(), "");
        let vals = ramp(N);
        let mut provider = |_: &str, _: ArraySize| Feed::Data(vals.clone());
        let f = r.render(Some(&mut provider)).unwrap().to_vec();
        let column = |x: usize| mean(&(0..H).map(|y| px(&f, W, x, y, 0)).collect::<Vec<_>>());
        assert!(column(0) < 40.0); // The left column holds vals[0] = 0.
        assert!(column(W - 1) > 215.0); // The right column holds vals[N-1] = 1.
        let row: Vec<f64> = (0..W).map(|x| px(&f, W, x, H / 2, 0)).collect();
        // The row runs dark to bright across x.
        assert!(row.windows(2).all(|w| w[1] - w[0] >= -1.0));
    }

    #[test]
    fn sample_interpolates() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = array_scene(
            "void main(){ vec2 uv=gl_FragCoord.xy/resolution; \
             gl_FragColor=vec4(vec3(vals_sample(uv.x)),1.0); }",
            "vals",
            json!(N),
        );
        assert_eq!(r.setup(&sc).unwrap(), "");
        let vals = ramp(N);
        let mut provider = |_: &str, _: ArraySize| Feed::Data(vals.clone());
        let f = r.render(Some(&mut provider)).unwrap().to_vec();
        // A ramp gives sample(0.5) = 0.5.
        let mid = mean(&(0..H).map(|y| px(&f, W, W / 2, y, 0)).collect::<Vec<_>>()) / 255.0;
        assert!(0.4 < mid && mid < 0.6);
    }

    #[test]
    fn a_u8_feed_renders_what_the_same_floats_render() {
        // The two wire forms name the same values: byte / 255. The frames must
        // match to the pixel, on a float data texture and on an 8-bit one.
        use sha2::{Digest, Sha256};
        let Some((_g, mut r)) = gpu() else { return };
        let bytes: Vec<u8> = (0..N).map(|i| (i * 4) as u8).collect();
        let floats: Vec<f32> = bytes.iter().map(|b| f32::from(*b) / 255.0).collect();
        for (code, size) in [
            (
                "void main(){ gl_FragColor=vec4(vec3(vals_sample(v_texCoord.x)),1.0); }",
                json!(N),
            ),
            (
                "void main(){ gl_FragColor=vec4(vec3(vals_sample2(v_texCoord)),1.0); }",
                json!([8, 8]),
            ),
        ] {
            let sc = array_scene(code, "vals", size.clone());
            assert_eq!(r.setup(&sc).unwrap(), "");
            let fed = floats.clone();
            let mut as_f32 = |_: &str, _: ArraySize| Feed::Data(fed.clone());
            let want = Sha256::digest(r.render(Some(&mut as_f32)).unwrap());
            let fed = bytes.clone();
            let mut as_u8 = |_: &str, _: ArraySize| Feed::DataU8(fed.clone());
            let got = Sha256::digest(r.render(Some(&mut as_u8)).unwrap());
            assert_eq!(got, want, "{size}");
        }
    }

    #[test]
    fn an_asset_binds_by_name_and_carries_its_size() {
        let Some((_g, mut r)) = gpu() else { return };
        // logo lands before setup, so the commit resolves it.
        // late lands on the running scene.
        let code = "uniform sampler2D logo; uniform vec2 logo_size; \
                    uniform sampler2D late; \
                    void main(){ gl_FragColor = vec4(\
                    texture2D(logo, v_texCoord).r, \
                    texture2D(late, v_texCoord).r, \
                    logo_size.x / 8.0, 1.0); }";
        r.set_asset("logo", 2, 2, 4, false, AssetData::U8(&[128u8; 16]))
            .unwrap();
        let sc = scene_of(json!([{"name": "main", "code": code}]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        r.set_asset("late", 1, 1, 1, false, AssetData::U8(&[64u8]))
            .unwrap();
        let f = r.render(None).unwrap().to_vec();
        for (c, want) in [(0, 128.0), (1, 64.0), (2, 64.0)] {
            assert!((mean(&channel(&f, c)) - want).abs() <= 2.0, "channel {c}");
        }
        // The same name again replaces the pixels, at a new size.
        r.set_asset("logo", 1, 1, 4, false, AssetData::U8(&[255u8; 4]))
            .unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!((mean(&channel(&f, 0)) - 255.0).abs() <= 2.0);
        assert!((mean(&channel(&f, 2)) - 32.0).abs() <= 2.0);
    }

    /// GL LINEAR filtering with CLAMP_TO_EDGE on a 1xN row.
    fn lerp1(vals: &[f32], t: f64) -> f64 {
        let last = vals.len() as f64 - 1.0;
        let x = t * vals.len() as f64 - 0.5;
        let i = x.floor();
        let at = |k: f64| f64::from(vals[k.clamp(0.0, last) as usize]);
        at(i) + (at(i + 1.0) - at(i)) * (x - i)
    }

    #[test]
    fn sample_matches_texel_lerp() {
        // The test compares every column of a rendered _sample(t) ramp against
        // the lerp above. It holds on a GL that linear-filters float textures.
        // It also holds on V3D, which has no float linear filtering.
        // The renderer emits a reader that makes the two agree.
        let Some((_g, mut r)) = gpu() else { return };
        let vals: Vec<f32> = vec![0.10, 0.90, 0.30, 0.70, 0.50];
        let sc = array_scene(
            "void main(){ gl_FragColor=vec4(vec3(vals_sample(v_texCoord.x)),1.0); }",
            "vals",
            json!(vals.len()),
        );
        assert_eq!(r.setup(&sc).unwrap(), "");
        let fed = vals.clone();
        let mut provider = |_: &str, _: ArraySize| Feed::Data(fed.clone());
        let f = r.render(Some(&mut provider)).unwrap().to_vec();
        for x in 0..W {
            let got = px(&f, W, x, H / 2, 0) / 255.0;
            let want = lerp1(&vals, (x as f64 + 0.5) / W as f64);
            assert!(
                (got - want).abs() < 4.0 / 255.0,
                "column {x}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn sample2_matches_bilinear() {
        // _sample2 reads points mid-texel in one axis, in both, and in neither.
        let Some((_g, mut r)) = gpu() else { return };
        let m: Vec<f32> = vec![0.0, 1.0, 0.5, 0.25]; // A 2x2 mask, indexed [y][x].
        for (uv, want) in [
            ((0.5, 0.5), 0.4375),             // The centre reads all four texels.
            ((0.25, 0.5), (0.0 + 0.5) / 2.0), // Column 0, between the rows.
            ((0.5, 0.25), (0.0 + 1.0) / 2.0), // Row 0, between the columns.
        ] {
            let code = format!(
                "void main(){{ gl_FragColor=vec4(vec3(mask_sample2(vec2({}, {}))),1.0); }}",
                uv.0, uv.1
            );
            let sc = array_scene(&code, "mask", json!([2, 2]));
            assert_eq!(r.setup(&sc).unwrap(), "");
            let fed = m.clone();
            let mut provider = |_: &str, _: ArraySize| Feed::Data(fed.clone());
            let f = r.render(Some(&mut provider)).unwrap().to_vec();
            let got = channel(&f, 0);
            let scaled: Vec<f64> = got.iter().map(|v| v / 255.0).collect();
            assert!((mean(&scaled) - want).abs() < 4.0 / 255.0, "{uv:?}");
            let (lo, hi) = (
                got.iter().cloned().fold(f64::MAX, f64::min),
                got.iter().cloned().fold(f64::MIN, f64::max),
            );
            assert_eq!(lo, hi); // One value covers the whole frame.
        }
    }

    // The reference produced the values below for these inputs.

    #[test]
    fn user_line_edges_match_the_reference() {
        let glsl = "#version 100\nprecision mediump float;\nfloat sat(float x){ return x; }\n\nvoid main(){\n  nope();\n}";
        let common = Some("float sat(float x){ return x; }");
        let code = "void main(){\n  nope();\n}";
        // The pass starts on line 5. The common block starts on line 3.
        assert_eq!(user_line("0:5(1): x", glsl, code, common), (Some(1), false));
        // A line at or above the common block start belongs to the pass.
        assert_eq!(user_line("0:2(1): x", glsl, code, common), (Some(1), false));
        assert_eq!(
            user_line("0:99(1): x", glsl, code, common),
            (Some(95), false)
        );
        // The parser takes the first ":<digits>(" in the text.
        assert_eq!(
            user_line("abc:12x(3) then :45( here", glsl, code, None),
            (Some(41), false)
        );
        assert_eq!(user_line("::7(2): x", glsl, code, None), (Some(3), false));
        assert_eq!(user_line("no digits :(", glsl, code, None), (None, false));
        // Pass code absent from the assembled source subtracts nothing.
        assert_eq!(
            user_line("0:6(3): e", glsl, "nowhere-in-glsl", common),
            (Some(6), false)
        );
        // An empty common block counts as no common block.
        assert_eq!(
            user_line("0:6(3): e", glsl, code, Some("")),
            (Some(2), false)
        );
    }

    #[test]
    fn probe_dri_devices_dedupes_an_override_that_repeats_a_default() {
        let _g = lock();
        env::set_var("GHEE_DRI", "/dev/dri/renderD128");
        let with = probe_dri_devices();
        env::set_var("GHEE_DRI", "");
        let empty = probe_dri_devices();
        env::remove_var("GHEE_DRI");
        let without = probe_dri_devices();
        assert_eq!(with, without);
        assert_eq!(empty, without); // An empty override counts as no override.
    }

    #[test]
    fn setup_error_strings_match_the_reference() {
        let Some((_g, mut r)) = gpu() else { return };
        // A 2D size prints as a pair. A 1D size prints as an int.
        let clash = json!([
            {"name": "a", "code": FLAT, "arrays": [{"name": "v", "size": [3, 5]}]},
            {"name": "b", "code": FLAT, "arrays": [{"name": "v", "size": 7}]},
        ]);
        assert_eq!(
            r.setup(&scene_of(clash)).unwrap(),
            "array 'v' declared at conflicting sizes ((3, 5) and 7)"
        );
        // The bad channel outranks the texture-unit budget of the same pass.
        let units = r.max_texture_units() as usize;
        let typos: Vec<Value> = (0..units).map(|_| json!({"buffer": "typo"})).collect();
        let both = json!([
            {"name": "a", "code": FLAT},
            {"name": "main", "channels": typos, "arrays": [{"name": "v", "size": 4}],
             "code": "void main(){ gl_FragColor = vec4(v_at(0)); }"},
        ]);
        assert_eq!(
            r.setup(&scene_of(both)).unwrap(),
            "pass 'main': channel 0 binds to buffer 'typo', which is not a pass in this scene"
        );
    }

    #[test]
    fn the_texture_budget_counts_an_array_name_once() {
        // A name declared twice on one pass takes one texture unit, so the
        // budget passes it. The compiler stops it on the redeclaration.
        let Some((_g, mut r)) = gpu() else { return };
        let units = r.max_texture_units() as usize;
        let channels: Vec<Value> = (0..units - 1).map(|_| json!({"buffer": "a"})).collect();
        let sc = scene_of(json!([
            {"name": "a", "code": FLAT},
            {"name": "main", "channels": channels,
             "arrays": [{"name": "v", "size": 4}, {"name": "v", "size": 4}],
             "code": "void main(){ gl_FragColor = vec4(v_at(0)); }"},
        ]));
        let err = r.setup(&sc).unwrap();
        assert!(!err.contains("texture units"), "{err}");
        if !err.is_empty() {
            assert_eq!(r.error_pass, Some(ErrorPass::Index(1)));
        }
    }

    #[test]
    fn a_param_reaches_the_shader_decomposed() {
        // A color param sets name_r, name_g and name_b.
        // A vecK param sets name_x, name_y, name_z and name_w.
        // This shader reads tint.r, off.x and off.y, so GL drops tint_g and
        // tint_b. The color set then reports false, and its red still lands.
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([{
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(tint.r, off.x, off.y, 1.0); }",
            "params": {"tint": {"type": "color", "value": [0.0, 0.0, 0.0]},
                       "off": {"type": "vec2", "value": [0.0, 0.0]},
                       "label": {"type": "string", "value": "hi"}},
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let color = ParamValue::Color([0.5, 0.1, 0.2]);
        assert!(!r.set_param("main", "tint", "color", &color).unwrap());
        let off = ParamValue::Vec(vec![0.25, 0.75]);
        assert!(r.set_param("main", "off", "vec2", &off).unwrap());
        // A string param is host-side. It sets nothing, on any pass.
        let label = ParamValue::Str("x".to_string());
        assert!(!r.set_param("main", "label", "string", &label).unwrap());
        assert!(!r.set_param("nope", "label", "string", &label).unwrap());
        assert!(!r
            .set_param("nope", "tint", "float", &ParamValue::Float(1.0))
            .unwrap());
        // An undeclared name sets nothing.
        assert!(!r
            .set_param("main", "ghost", "color", &ParamValue::Color([1.0; 3]))
            .unwrap());
        // A type with more suffixes than components stops at the components.
        assert!(r.set_param("main", "off", "vec4", &off).unwrap());
        let f = r.render(None).unwrap();
        for (c, want) in [(0, 128.0), (1, 64.0), (2, 191.0), (3, 255.0)] {
            assert!(
                (f64::from(f[c]) - want).abs() <= 2.0,
                "channel {c}: {}",
                f[c]
            );
        }
    }

    #[test]
    fn a_scene_param_reaches_every_pass_not_just_the_first() {
        // 'a' writes P to red. 'main' adds P to green through the channel.
        // A set that stops at the first pass leaves green at its default.
        let Some((_g, mut r)) = gpu() else { return };
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "params": {"P": {"type": "float", "value": 0.0}},
                "passes": [
                    {"name": "a", "code": "void main(){ gl_FragColor = vec4(P, 0.0, 0.0, 1.0); }"},
                    {"name": "main", "channels": [{"buffer": "a"}],
                     "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord) \
                              + vec4(0.0, P, 0.0, 0.0); }"},
                ],
            }),
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        assert!(r
            .set_scene_param("P", "float", &ParamValue::Float(0.5))
            .unwrap());
        let f = r.render(None).unwrap();
        assert!((f64::from(f[0]) - 128.0).abs() <= 2.0, "{}", f[0]);
        assert!((f64::from(f[1]) - 128.0).abs() <= 2.0, "{}", f[1]);
        // No pass holds this uniform. A string param sets nothing anywhere.
        assert!(!r
            .set_scene_param("ghost", "float", &ParamValue::Float(0.5))
            .unwrap());
        assert!(!r
            .set_scene_param("P", "string", &ParamValue::Str("x".to_string()))
            .unwrap());
    }

    #[test]
    fn feed_lengths_that_miss_the_declared_size() {
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([{
            "name": "main",
            "arrays": [{"name": "mask", "size": [2, 2]}, {"name": "vals", "size": 3}],
            "code": "void main(){ gl_FragColor = vec4(mask_sample2(vec2(0.25, 0.25)), \
                     mask_sample2(vec2(0.75, 0.75)), vals_at(2), 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut run = |mask: Vec<f32>, vals: Vec<f32>| -> [u8; 4] {
            let mut provider = |name: &str, _: ArraySize| {
                Feed::Data(if name == "mask" {
                    mask.clone()
                } else {
                    vals.clone()
                })
            };
            let f = r.render(Some(&mut provider)).unwrap();
            [f[0], f[1], f[2], f[3]]
        };
        let near = |got: [u8; 4], want: [f64; 4]| {
            for c in 0..4 {
                assert!(
                    (f64::from(got[c]) - want[c]).abs() <= 2.0,
                    "channel {c}: {got:?} vs {want:?}"
                );
            }
        };
        near(
            run(vec![1., 0., 0., 1.], vec![0., 0., 1.]),
            [255., 255., 255., 255.],
        );
        // A 2D feed past the mask keeps the declared w x h and drops the tail.
        near(
            run(vec![0.25, 0.5, 0.75, 1.0, 9.0, 9.0], vec![0., 0., 0.5]),
            [64., 255., 128., 255.],
        );
        // An empty 2D feed zero-pads the whole mask.
        near(run(Vec::new(), vec![0., 0., 0.25]), [0., 0., 64., 255.]);
        // A 1D feed past the declared size uploads min(len, size) values.
        near(
            run(vec![1., 1., 1., 1.], vec![0.1, 0.2, 0.3, 0.9, 0.9]),
            [255., 255., 76., 255.],
        );
        // An empty 1D feed uploads zero width. The texture keeps its data.
        near(
            run(vec![1., 1., 1., 1.], Vec::new()),
            [255., 255., 76., 255.],
        );
    }

    #[test]
    fn a_readback_size_is_w_times_h_times_components() {
        // out_size reads back a GPU-scaled frame at that exact size.
        // The size scales up or down, and on either axis alone.
        let Some((_g, mut r)) = gpu() else { return };
        let sc = scene_of(json!([{
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(v_texCoord.x, v_texCoord.y, 0.25, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        for (w, h) in [(W * 2, H * 2), (W, H), (1, 1), (3, 7)] {
            let out = r.render_scaled(None, (w as i32, h as i32)).unwrap();
            assert_eq!(out.len(), w * h * COMPONENTS);
            let mut buf = vec![0u8; w * h * COMPONENTS];
            r.render_into(None, &mut buf, Some((w as i32, h as i32)))
                .unwrap();
            assert_eq!(buf, out);
        }
    }
}
