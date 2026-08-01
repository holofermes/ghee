//! Port of ghee/engine.py (see that file for the reference implementation).
//!
//! Renderer drives the C engine over the ghee_* ABI. The engine is linked into
//! this binary, so Python's lib_path / $GHEE_LIB / bundled .so lookup has no
//! counterpart here; $GHEE_DRI still picks the device.

use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_double, c_int};
use std::ptr;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::cparam::create_c_parameter;
use crate::ffi;
use crate::scene::{ArraySize, ParamValue, Scene};

/// error_pass for a compile error in the scene's common block; a pass error
/// reports the pass's index instead.
pub const COMMON_PASS: &str = "common";

/// vecN params reach the shader as decomposed float uniforms named by these.
const VEC_SUFFIXES: [&str; 4] = ["_x", "_y", "_z", "_w"];

/// Which shader a setup() compile error sits in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorPass {
    Common,
    Index(usize),
}

impl ErrorPass {
    /// The wire form: a common-block error by name, a pass error by index.
    pub fn to_json(self) -> Value {
        match self {
            ErrorPass::Common => Value::from(COMMON_PASS),
            ErrorPass::Index(i) => Value::from(i),
        }
    }
}

/// What a provider hands back for one declared array. Unchanged is the scene's
/// explicit statement that the texture already holds this array, so the upload
/// is skipped entirely; the engine never caches or compares data to guess.
/// After setup() rebuilds a scene the textures are empty, so a controller must
/// push real data again first.
pub enum Feed {
    Data(Vec<f32>),
    Unchanged,
}

/// provider(name, size) -> this frame's data for one declared array.
pub type Provider<'a> = &'a mut dyn FnMut(&str, ArraySize) -> Feed;

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

/// DRI nodes to try for offscreen rendering, in order: $GHEE_DRI, the render
/// nodes, then the primary cards. First one that creates a renderer wins. One
/// list for every caller, so the console, the tests and the Renderer itself all
/// land on the same GPU.
///
/// Render nodes come first because a primary node carries DRM master. A render
/// node has no master to contend for.
pub fn probe_dri_devices() -> Vec<String> {
    let mut devices: Vec<String> = env_dri().into_iter().collect();
    devices.push("/dev/dri/renderD128".to_string());
    devices.extend(dri_nodes("renderD"));
    devices.push("/dev/dri/card0".to_string());
    devices.extend(dri_nodes("card"));
    dedupe(devices)
}

/// Primary DRI nodes, $GHEE_DRI first. Display work needs one of these: a
/// render node has no connectors and cannot take DRM master.
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

/// The card node driving a connected display, as (device, width, height), or
/// None when nothing is connected.
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

/// Size of the mode the connected display is driving, as (width, height), or
/// None when there is no card or nothing connected.
pub fn probe_display(dri_device: Option<&str>) -> Option<(i32, i32)> {
    probe_display_device(dri_device).map(|(_, w, h)| (w, h))
}

/// Bytes a frame packs down to: 1bpp (bw) vs RGB565.
pub fn packed_size(width: i32, height: i32, bw: bool) -> usize {
    let (w, h) = (i64::from(width), i64::from(height));
    (if bw { w * ((h + 7) / 8) } else { w * h * 2 }) as usize
}

/// Pack an RGB(A) frame to 1bpp (bw) or RGB565, with no Renderer. For output
/// sinks that hold a frame but not an engine.
#[allow(dead_code)]
pub fn pack_frame(rgb: &[u8], width: i32, height: i32, components: i32, bw: bool) -> Vec<u8> {
    let mut out = vec![0u8; packed_size(width, height, bw)];
    pack_frame_into(rgb, width, height, components, bw, &mut out);
    out
}

/// pack_frame into a buffer the caller holds; zeros it first.
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
    // the packer reads three channels of every pixel, so under 3 components its
    // last pixel would read past the frame span checked below
    assert!(
        components >= 3,
        "packing needs 3+ components, got {components}"
    );
    // the C side reads and writes these spans with no length of its own
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

fn err_text(err: *mut c_char) -> String {
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

/// The line a GL compile error names: the first `:<digits>(` in it.
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

/// Map a GL compile-error line (full shader) to the line its author wrote.
/// Returns (line, in_common): a line inside the scene's prepended common block
/// counts from the start of that block, anything else is the pass's own code
/// with the generated preamble's length subtracted.
fn user_line(err: &str, glsl: &str, code: &str, common: Option<&str>) -> (Option<i64>, bool) {
    let Some(line) = error_line_number(err) else {
        return (None, false);
    };
    let pos = glsl.rfind(code); // the pass code is the tail of the assembled source
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
/// ghee is thread-affine: construct and use a Renderer on a single thread (the
/// context is made current at create time). A worker that renders in its own
/// thread must create the Renderer in that thread; the raw handle keeps this
/// type off Send.
pub struct Renderer {
    pub width: i32,
    pub height: i32,
    pub components: i32,
    pub scene: Option<Scene>,
    /// last setup() compile error text ("" = ok)
    pub error: String,
    /// last setup() compile error: the line of user code it sits on
    pub error_line: Option<i64>,
    /// the pass it sits in, or Common for the scene's common block
    pub error_pass: Option<ErrorPass>,
    #[allow(dead_code)]
    pub display: bool,
    /// pass name -> engine pass index, in the order the passes were added
    mp: Vec<(String, c_int)>,
    /// array name -> size, in declaration order
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
        // the engine reads back GL_RGB for 3 and GL_RGBA for anything else, so
        // any other count would size the frame buffers under what it writes
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
            // scan out to a KMS connector: needs the card (KMS) node, not a
            // render node, and DRM master.
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

    /// The engine handle, for the ghee_* calls a host makes for itself (the
    /// encoder).
    pub fn handle(&self) -> *mut ffi::GheeScene {
        self.handle
    }

    fn pass_index(&self, name: &str) -> Option<c_int> {
        self.mp.iter().find(|(n, _)| n == name).map(|(_, i)| *i)
    }

    /// Build the scene's passes. Returns "" on success or an error string (a
    /// compile error, or a scene the engine refuses). On a compile error
    /// error_line/error_pass point at the offending user-code line; an error in
    /// the scene's common block reports error_pass=Common and the line within
    /// that block.
    pub fn setup(&mut self, scene: &Scene) -> Result<String> {
        self.error = String::new();
        self.error_line = None;
        self.error_pass = None;
        if scene.passes.is_empty() {
            self.error = "scene has no passes".to_string();
            return Ok(self.error.clone());
        }
        // An array name is one logical input, so it must have a single size.
        // Reject a scene that declares the same name at two sizes before
        // touching the engine.
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
        // The output pass renders at the renderer size (its buffer feeds
        // readback/display/encoder). Checked here against the final resolved
        // pass list, so a scene whose own last pass is sized stays valid once
        // filters append a full-size pass after it.
        let last = scene.output();
        if last.size.is_some() {
            self.error = format!(
                "output pass '{}' must not set a size; it renders at the renderer \
                 size ({}x{})",
                last.name, self.width, self.height
            );
            return Ok(self.error.clone());
        }
        // A channel binds to a pass by name. A name that is no pass is an error;
        // buffer=None is the deliberate unbound channel and reads black.
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
        for (i, p) in scene.passes.iter().enumerate() {
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
                return Ok(err);
            }
            mp.push((p.name.clone(), idx));
        }
        if unsafe { ffi::ghee_mp_commit(self.handle) } < 0 {
            self.error = or_else(self.last_error(), "commit failed");
            return Ok(self.error.clone());
        }
        self.mp = mp;
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
        // scene params reach every pass at once (they sit in every preamble)
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

    /// Render all passes; the final pass is the returned RGB(A) frame, row-major
    /// (height, width, components). `provider` fills each declared array for
    /// this frame; None leaves the arrays untouched.
    pub fn render(&mut self, provider: Option<Provider<'_>>) -> Result<&[u8]> {
        self.feed(provider)?;
        unsafe { ffi::ghee_mp_render(self.handle, self.rgb.as_mut_ptr(), 0, 0) };
        Ok(&self.rgb)
    }

    /// render() into a buffer the host holds. out_size=(w, h) reads back a
    /// GPU-downscaled frame at that size instead of the full one, as present()
    /// does.
    pub fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()> {
        self.feed(provider)?;
        let (pw, ph) = out_size.unwrap_or((0, 0));
        // the engine downscales only when both dims are positive; any other
        // out_size falls through to the full-frame readback
        let (w, h) = if pw > 0 && ph > 0 {
            (pw, ph)
        } else {
            (self.width, self.height)
        };
        let want = (i64::from(w) * i64::from(h) * i64::from(self.components)) as usize;
        // the readback writes this span with no length of its own
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

    /// Upload each declared array for this frame (once per unique name). A
    /// render uploads whatever the provider returns; the one exception is the
    /// explicit Unchanged answer, a scene's declaration that the texture already
    /// holds this array (a partially dynamic scene skips its static mask that
    /// way). The engine never caches or diffs data on its own.
    fn feed(&self, provider: Option<Provider<'_>>) -> Result<()> {
        let Some(provider) = provider else {
            return Ok(());
        };
        for (name, size) in &self.feeds {
            let Feed::Data(data) = provider(name, *size) else {
                continue;
            };
            let (w, h, data) = match *size {
                ArraySize::Linear(n) => (data.len().min(n as usize) as c_int, 1, data),
                ArraySize::Grid(gw, gh) => {
                    let want = (gw * gh) as usize;
                    let data = if data.len() < want {
                        // short feed: zero-pad to the full mask
                        let mut buf = vec![0f32; want];
                        buf[..data.len()].copy_from_slice(&data);
                        buf
                    } else {
                        data
                    };
                    (gw as c_int, gh as c_int, data)
                }
            };
            let cname = CString::new(name.as_str())?;
            unsafe {
                ffi::ghee_mp_set_array(self.handle, cname.as_ptr(), data.as_ptr(), w, h);
            }
        }
        Ok(())
    }

    /// Display mode only: feed arrays, render the passes, and scan the final
    /// buffer straight out to the connected display. `preview` also reads back a
    /// GPU-downscaled frame (a few hundred KB, not the full frame); None skips
    /// the readback.
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

    /// Set one uniform on a pass live (no recompile). False if the pass/uniform
    /// is absent. A "color" param is [r, g, b]: it sets the three decomposed
    /// float uniforms name_r/_g/_b (the engine has no color type). A "vecK"
    /// param is K floats: it sets the decomposed float uniforms name_x/_y/_z/_w.
    /// A "string" param is host-side (the scene's controller consumes it):
    /// nothing to set here.
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

    /// Set one scene-scoped uniform on every pass live (no recompile). A scene
    /// param sits in every pass's preamble, so the set fans out to all of them;
    /// a pass that never reads it has an inactive uniform and is skipped. True
    /// if at least one pass held the uniform and took the value. Every pass is
    /// set, not just up to the first that holds the uniform.
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

    /// pack() into a buffer the caller holds; zeros it first.
    #[allow(dead_code)]
    pub fn pack_into(&self, rgb: &[u8], bw: bool, out: &mut [u8]) {
        pack_frame_into(rgb, self.width, self.height, self.components, bw, out);
    }

    /// The connected display mode's refresh rate in Hz, 0 when not in display
    /// mode.
    pub fn display_vrefresh(&self) -> i32 {
        unsafe { ffi::ghee_display_vrefresh(self.handle) }
    }

    /// Enable per-frame draw/readback timing (adds a glFinish; off = zero cost).
    #[allow(dead_code)]
    pub fn set_profiling(&self, on: bool) {
        unsafe { ffi::ghee_set_profiling(self.handle, on) };
    }

    /// Pin the `time` uniform to a fixed value, for reproducible renders.
    /// Default is the monotonic clock since the engine was created; None hands
    /// it back to that clock.
    pub fn set_time(&self, t: Option<f64>) {
        match t {
            None => unsafe { ffi::ghee_clear_time(self.handle) },
            Some(t) => unsafe { ffi::ghee_set_time(self.handle, t as f32) },
        }
    }

    /// Set the `DT` uniform: the seconds a scene should advance by this frame.
    /// Rides the draw like `time`, one store here and one upload per pass.
    /// Defaults to 1/60 until a caller feeds a measured period.
    pub fn set_dt(&self, value: f64) {
        unsafe { ffi::ghee_set_dt(self.handle, value as f32) };
    }

    /// Fragment texture units this GPU exposes: a pass's channels plus its array
    /// inputs have to fit (8 on VC4).
    pub fn max_texture_units(&self) -> i32 {
        unsafe { ffi::ghee_max_texture_units(self.handle) }
    }

    /// Whether this GPU linear-filters the textures array inputs are bound as.
    /// False on V3D (float textures, no OES_texture_float_linear), where a
    /// filtered fetch silently comes back nearest.
    pub fn data_texture_linear(&self) -> bool {
        unsafe { ffi::ghee_data_texture_linear(self.handle) }
    }

    /// Last frame's split as (draw_ms, read_ms): GPU render of all passes vs the
    /// glReadPixels readback. Both 0 until a frame renders with profiling on.
    #[allow(dead_code)]
    pub fn timings(&self) -> (f64, f64) {
        let (mut draw, mut read): (c_double, c_double) = (0.0, 0.0);
        unsafe { ffi::ghee_get_timings(self.handle, &mut draw, &mut read) };
        (draw, read)
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

    /// One GL device, one Renderer at a time, and $GHEE_DRI is process-wide: the
    /// tests that touch either run one after another.
    static GPU: Mutex<()> = Mutex::new(());

    fn lock() -> MutexGuard<'static, ()> {
        GPU.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The shared renderer of tests/conftest.py, held with the device lock. None
    /// where that fixture skips: no GL device.
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
        // a primary node carries DRM master: a second concurrent process on a
        // Pi 5's card0 renders black, a render node has no master to contend for
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
    fn probe_display_device_names_the_card_it_found() {
        // Headless hosts (CI, render-node-only cards) legitimately probe to
        // None; anything else must be a positive WxH on a card node.
        let _g = lock();
        env::remove_var("GHEE_DRI");
        let Some((dev, w, h)) = probe_display_device(None) else {
            return;
        };
        // the connectors are not always on card0: on a Pi 5 they are on card1
        assert!(probe_dri_cards().contains(&dev));
        assert!(w > 0 && h > 0);
        assert_eq!(probe_display(None), Some((w, h)));
    }

    #[test]
    fn display_create_fails_once_without_drm_master() {
        let _g = lock();
        if fs::metadata("/proc/self").is_ok_and(|m| m.uid() == 0) {
            return; // not worth contending for the session's card as root
        }
        let Some((dev, w, h)) = probe_display_device(None) else {
            return;
        };
        let before_the_master_attempt = ["cannot open DRI device", "eglInitialize failed"];
        // An idle box has no master to lose the race against, so take it here
        // rather than relying on whatever happens to be running.
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
    fn a_flip_the_board_is_too_slow_to_finish_is_still_waited_for() {
        // A page flip is fenced on the GPU finishing the frame, so its wait
        // scales with the render rather than with the panel's refresh. Bound
        // that wait near a frame time and a slow board leaves the flip pending,
        // and every submit made while it stays pending is refused.
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
        assert!(bound >= 1000); // slower than any frame, i.e. only a stuck flip hits it

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
    fn a_compile_error_line_maps_back_to_the_code_its_author_wrote() {
        let glsl = "#version 100\nprecision mediump float;\nfloat sat(float x){ return x; }\n\nvoid main(){\n  nope();\n}";
        let common = "float sat(float x){ return x; }";
        let code = "void main(){\n  nope();\n}";
        // line 6 of the assembled source is line 2 of the pass
        assert_eq!(
            user_line("0:6(3): error: no function nope", glsl, code, Some(common)),
            (Some(2), false)
        );
        // line 3 sits inside the common block, counted from its own start
        assert_eq!(
            user_line("0:3(3): error: bad", glsl, code, Some(common)),
            (Some(1), true)
        );
        // the same line with no common block is the pass's, clamped at 1
        assert_eq!(
            user_line("0:3(3): error: bad", glsl, code, None),
            (Some(1), false)
        );
        // an error with no line at all attributes nothing
        assert_eq!(user_line("out of memory", glsl, code, None), (None, false));
    }

    #[test]
    fn packed_size_is_1bpp_by_column_or_rgb565() {
        assert_eq!(packed_size(128, 32, true), 128 * 4);
        assert_eq!(packed_size(128, 33, true), 128 * 5); // partial byte column
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
        // buffer=None is the deliberate unbound channel, not a broken reference
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
        // a size clash outranks the sized output pass below it
        let both = json!([
            {"name": "a", "code": FLAT, "arrays": [{"name": "v", "size": 4}]},
            {"name": "b", "code": FLAT, "size": [8, 8],
             "arrays": [{"name": "v", "size": [2, 2]}]},
        ]);
        assert!(r.setup(&scene_of(both)).unwrap().starts_with("array 'v'"));
        // and the sized output pass outranks the bad channel below it
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
        // 'leak' leaves a white texture bound on unit 0; 'main' has an unbound
        // channel 0, whose sampler used to default to that unit and sample the leak
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
        // the engine only downscales for positive dims; a zero or negative one
        // falls through to the full-frame readback, which the buffer check has
        // to size for (it used to demand 0 bytes and let C write the frame)
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
        // a full-size buffer satisfies the fall-through readback
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
        // feedback: each frame adds to the last, so a draw() that really
        // rendered shows up in the next readback
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
        assert_eq!(r.setup(&sc).unwrap(), ""); // fresh buffers, cleared to black
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
        assert_eq!(calls, ["shared"]); // one upload, three passes reading it
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

        // one channel past what this GPU has, plus an array on top
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
        // DT rides the draw like time, so one store has to land in the uniform
        // of every pass. Red carries the first pass's DT through a channel,
        // green the second's.
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
        assert!(near(&f, (255.0f64 / 60.0).round())); // engine default
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
        // one GPU node, one renderer at a time
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
        // mask reads its second half, vals is fed past its declared size
        let sc = scene_of(json!([{
            "name": "main",
            "arrays": [{"name": "mask", "size": [2, 2]}, {"name": "vals", "size": 2}],
            "code": "void main(){ gl_FragColor = vec4(mask_sample2(vec2(0.75, 0.75)), \
                     vals_at(1), 0.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut provider = |name: &str, _size: ArraySize| match name {
            // half a 2x2 mask: the second row zero-pads
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
        assert_eq!(r.error_line, Some(2)); // the line the author wrote nope() on
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
        // a good compile clears the attribution
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        assert!(r.error_pass.is_none() && r.error_line.is_none());
    }

    // data arrays, fed as 1xN textures (tests/test_texture.py)

    const N: usize = 64; // past VC4's ~48 uniform-array cap

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
        assert!(column(0) < 40.0); // left column ~ vals[0] = 0
        assert!(column(W - 1) > 215.0); // right column ~ vals[N-1] = 1
        let row: Vec<f64> = (0..W).map(|x| px(&f, W, x, H / 2, 0)).collect();
        // non-decreasing dark->bright across x
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
        // sample(0.5) ~ 0.5 for a ramp
        let mid = mean(&(0..H).map(|y| px(&f, W, W / 2, y, 0)).collect::<Vec<_>>()) / 255.0;
        assert!(0.4 < mid && mid < 0.6);
    }

    /// GL's LINEAR + CLAMP_TO_EDGE on a 1xN row.
    fn lerp1(vals: &[f32], t: f64) -> f64 {
        let last = vals.len() as f64 - 1.0;
        let x = t * vals.len() as f64 - 0.5;
        let i = x.floor();
        let at = |k: f64| f64::from(vals[k.clamp(0.0, last) as usize]);
        at(i) + (at(i + 1.0) - at(i)) * (x - i)
    }

    #[test]
    fn sample_matches_texel_lerp() {
        // Every column of a rendered _sample(t) ramp against the lerp computed
        // here. Holds on a GL that linear-filters float textures and on one that
        // cannot (V3D); which reader the renderer emitted is what makes the
        // values agree.
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
        // _sample2 at points that land mid-texel in one axis, both, or neither
        let Some((_g, mut r)) = gpu() else { return };
        let m: Vec<f32> = vec![0.0, 1.0, 0.5, 0.25]; // 2x2, [y][x]
        for (uv, want) in [
            ((0.5, 0.5), 0.4375),             // centre: all four texels
            ((0.25, 0.5), (0.0 + 0.5) / 2.0), // on column 0, between rows
            ((0.5, 0.25), (0.0 + 1.0) / 2.0), // on row 0, between columns
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
            assert_eq!(lo, hi); // one value, the whole frame
        }
    }

    // Behaviours checked against ghee/engine.py by running both sides on the
    // same input; the values below are what the Python one produced.

    #[test]
    fn user_line_edges_match_the_reference() {
        let glsl = "#version 100\nprecision mediump float;\nfloat sat(float x){ return x; }\n\nvoid main(){\n  nope();\n}";
        let common = Some("float sat(float x){ return x; }");
        let code = "void main(){\n  nope();\n}";
        // the pass starts on line 5, the common block on line 3
        assert_eq!(user_line("0:5(1): x", glsl, code, common), (Some(1), false));
        // a line at or above the common block's own start is the pass's
        assert_eq!(user_line("0:2(1): x", glsl, code, common), (Some(1), false));
        assert_eq!(
            user_line("0:99(1): x", glsl, code, common),
            (Some(95), false)
        );
        // the first ":<digits>(" in the text, not the first colon
        assert_eq!(
            user_line("abc:12x(3) then :45( here", glsl, code, None),
            (Some(41), false)
        );
        assert_eq!(user_line("::7(2): x", glsl, code, None), (Some(3), false));
        assert_eq!(user_line("no digits :(", glsl, code, None), (None, false));
        // code the assembled source does not contain subtracts nothing
        assert_eq!(
            user_line("0:6(3): e", glsl, "nowhere-in-glsl", common),
            (Some(6), false)
        );
        // an empty common block is no common block
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
        assert_eq!(empty, without); // an empty override is no override
    }

    #[test]
    fn setup_error_strings_match_the_reference() {
        let Some((_g, mut r)) = gpu() else { return };
        // a 2D size prints as the pair it is stored as, a 1D one as the int
        let clash = json!([
            {"name": "a", "code": FLAT, "arrays": [{"name": "v", "size": [3, 5]}]},
            {"name": "b", "code": FLAT, "arrays": [{"name": "v", "size": 7}]},
        ]);
        assert_eq!(
            r.setup(&scene_of(clash)).unwrap(),
            "array 'v' declared at conflicting sizes ((3, 5) and 7)"
        );
        // the bad channel outranks the texture-unit budget the same pass blows
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
        // a name declared twice on one pass is one texture unit, so the budget
        // lets it through; what stops it is the compiler, on the redeclaration
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
        // a color sets name_r/_g/_b and a vecK name_x/_y/_z/_w; the shader here
        // reads tint.r, off.x and off.y, so the unread tint_g/tint_b are
        // optimised out and the color set reports false while its red lands
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
        // a string param is host-side: nothing to set, on any pass or none
        let label = ParamValue::Str("x".to_string());
        assert!(!r.set_param("main", "label", "string", &label).unwrap());
        assert!(!r.set_param("nope", "label", "string", &label).unwrap());
        assert!(!r
            .set_param("nope", "tint", "float", &ParamValue::Float(1.0))
            .unwrap());
        // a name no pass declares sets nothing
        assert!(!r
            .set_param("main", "ghost", "color", &ParamValue::Color([1.0; 3]))
            .unwrap());
        // more suffixes than components stops at the components
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
        // 'a' writes P to red, 'main' adds P to green through the channel. Stop
        // at the first pass that took the value and green stays at its default.
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
        // no pass holds the uniform, and a string param sets nothing anywhere
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
        // a 2D feed past the mask keeps the declared w x h and drops the tail
        near(
            run(vec![0.25, 0.5, 0.75, 1.0, 9.0, 9.0], vec![0., 0., 0.5]),
            [64., 255., 128., 255.],
        );
        // an empty 2D feed zero-pads the whole mask
        near(run(Vec::new(), vec![0., 0., 0.25]), [0., 0., 64., 255.]);
        // a 1D feed past the declared size uploads min(len, size) of it
        near(
            run(vec![1., 1., 1., 1.], vec![0.1, 0.2, 0.3, 0.9, 0.9]),
            [255., 255., 76., 255.],
        );
        // and an empty 1D feed is a zero-wide upload: the texture keeps its data
        near(
            run(vec![1., 1., 1., 1.], Vec::new()),
            [255., 255., 76., 255.],
        );
    }

    #[test]
    fn a_readback_size_is_w_times_h_times_components() {
        // out_size reads back a GPU-scaled frame at exactly that size, up or
        // down from the renderer's and on either axis alone
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
