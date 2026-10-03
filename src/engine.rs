//! The renderer: one GPU context, and the scene that it draws.
//!
//! The C engine in ghee/_engine owns everything GL: the EGL context on a DRI
//! device, each pass's program and its two buffers, the quads that a pass
//! draws after its own, the textures, the readback and the scanout.
//! ghee/_engine/engine.h lists its calls, and `crate::ffi` declares them. A
//! [`Renderer`] holds one engine and drives it from a [`Scene`]:
//!
//! - [`Renderer::new`] opens the engine on the device named by `$GHEE_DRI`,
//!   else on the first render node that works, or on a card for a KMS
//!   display.
//! - [`Renderer::setup`] writes each pass's GLSL, compiles the passes and
//!   their layers beside the running scene, and swaps them in at once. A
//!   compile error leaves the running scene drawing, and names the line of
//!   the author's own GLSL, counted without the preamble ghee puts in front.
//! - [`Renderer::render`] asks a [`Provider`] for the data of each declared
//!   array, uploads it, draws every pass in order, and reads the frame back.
//!   [`Renderer::present`] scans it out to the display.
//!
//! The Renderer draws the scene's texts too. [`Renderer::set_text`] lays the
//! words out through `crate::text`, uploads the texture of curves that
//! `_cover` and the slug fill read, and hands each layer one quad a glyph. A
//! layer of the MTSDF fill reads its glyphs from an atlas that a
//! `crate::atlas` generator fills on its own threads, and every render
//! uploads the glyphs that the atlas gained since the last.

use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::assets::Pixels;
use crate::atlas;
use crate::cparam::create_c_parameter;
use crate::ffi;
use crate::scene::{ArraySize, Layer, ParamValue, Pass, Scene};
use crate::text::{self, Fault, Fill, Layout, Text};

/// The passes that the engine holds, filter passes included. engine.c holds
/// the same number as MP_MAX_PASSES.
const MAX_PASSES: usize = 16;

/// error_pass for a compile error in the scene's common block.
/// A pass error reports the pass index.
pub const COMMON_PASS: &str = "common";

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

/// A layer's GLSL for one stage, under the layer's key of the same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerPart {
    Vertex,
    Fragment,
}

impl LayerPart {
    /// The layer's key that holds this GLSL.
    pub fn key(self) -> &'static str {
        match self {
            LayerPart::Vertex => "vertex",
            LayerPart::Fragment => "fragment",
        }
    }
}

/// The layer holding a setup() compile error, in the pass of error_pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ErrorLayer {
    /// The layer's index in its pass's layers.
    pub index: usize,
    /// The GLSL holding the error. error_line counts from its first line.
    pub part: LayerPart,
}

impl ErrorLayer {
    /// The wire form, `{"index": 0, "part": "fragment"}`.
    pub fn to_json(self) -> Value {
        serde_json::json!({"index": self.index, "part": self.part.key()})
    }
}

/// What a provider returns for one declared array.
///
/// DataU8 is the same array from bytes, where the value is byte / 255. A mask
/// that starts as bytes reaches the texture without a float detour.
/// Unchanged states that the texture already holds this array. The engine then
/// skips the upload. The engine never caches or compares data.
/// setup() rebuilds a scene with empty textures, so a provider sends its data
/// again at the first render after every setup(): Unchanged there leaves the
/// texture empty.
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

/// How a batch's colour lands on its pass's buffer. The colour is
/// premultiplied by its alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// src + dst * (1 - src.a)
    Over,
    /// src + dst
    Add,
    /// src
    Replace,
}

impl Blend {
    fn engine(self) -> c_int {
        match self {
            Blend::Over => ffi::GHEE_BLEND_OVER,
            Blend::Add => ffi::GHEE_BLEND_ADD,
            Blend::Replace => ffi::GHEE_BLEND_REPLACE,
        }
    }
}

/// Quads that a pass draws after its own quad, with a vertex and a fragment
/// shader of their own and a blend, for setup_batches() to hand the engine
/// through ghee_mp_add_batch. A scene layer takes two or three such batches of
/// its own: the layer, its colour glyphs when it paints them, and its error
/// boxes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Batch {
    /// The name of the pass that the batch draws into.
    pub pass: String,
    pub vertex: String,
    pub fragment: String,
    pub blend: Blend,
}

/// A scene layer that the engine holds as a batch, and the batches of its
/// colour glyphs and of its error boxes.
struct LayerBatches {
    /// The engine's index of the batch.
    batch: c_int,
    /// The engine's index of the colour batch, -1 for a layer that paints no
    /// colour glyphs or a text whose fonts hold no colour bitmaps.
    colour: c_int,
    /// The engine's index of the error-box batch.
    boxes: c_int,
    /// The scene text of the layer.
    text: String,
    /// The layer in an error, as "pass 'main' layer 'glow'".
    name: String,
}

/// One atlas generator, and the engine textures that hold its atlas.
struct AtlasSlot {
    fonts: Vec<text::Font>,
    px_em: u32,
    range: u32,
    gen: atlas::Generator,
    /// The engine asset `<name>_atlas` of each text drawn from this atlas, and
    /// whether it still waits for its first upload, which takes all of it.
    textures: Vec<(String, bool)>,
}

/// New words of a text that wait for the atlas to hold their glyphs.
struct Held {
    words: String,
    laid: Layout,
}

/// The font that an atlas generator reads, from a text's fonts.
type AtlasFont = Box<dyn Fn(&[text::Font]) -> Arc<dyn atlas::Font>>;

/// The colour atlas of one text, the engine asset `<name>_emoji`.
#[derive(Default)]
struct ColourAtlas {
    atlas: atlas::Atlas,
    /// The atlas's mip levels below level 0.
    mips: crate::colour::Mips,
    /// The texels per em of each glyph's bitmap, by atlas id.
    scale: HashMap<u32, f64>,
    /// Why each glyph that failed took no cell, by atlas id.
    failed: HashMap<u32, String>,
}

/// The problems of one scene text, a line each.
#[derive(Clone, Default, PartialEq)]
struct TextErrors {
    /// The problems of the words on screen.
    drawn: Vec<String>,
    /// Why the last words failed to lay out, when they did.
    refused: Option<String>,
}

/// The (width, rows) of one array upload, and its data.
///
/// A 1xN array uploads as many values as the feed holds, N at most. A 2D
/// array whose feed falls short pads with zeros to w x h, so each row starts
/// where the shader reads it.
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

/// True when GL on this card runs on the GPU.
/// A display-only card (a panel under a kernel display driver) still gets
/// hardware GL: Mesa pairs it with the render node and imports each frame
/// back for the flip. False therefore means no usable render node exists,
/// and the kms path refuses to run rather than render through llvmpipe.
pub fn display_renders(dri_device: &str) -> bool {
    let Ok(cdev) = CString::new(dri_device) else {
        return true;
    };
    unsafe { ffi::ghee_display_renders(cdev.as_ptr()) != 0 }
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
        let lines = c.lines().count() as i64;
        if cat >= 0 && cat < line && line <= cat + lines && line <= at {
            return (Some((line - cat).max(1)), true);
        }
    }
    (Some((line - at).max(1)), false)
}

/// A layer's compile error, and where it falls: `line` counts in the layer's
/// `part`, or in the scene's common block when `common` is set. An error on
/// another line carries none.
struct LayerFault {
    message: String,
    part: LayerPart,
    line: Option<i64>,
    common: bool,
}

/// The compile error of the layer named `layer`, with the line where it falls in
/// the layer's own GLSL or in the common block.
fn layer_error(
    layer: &str,
    d: &Layer,
    err: &str,
    vs: &str,
    fs: &str,
    common: Option<&str>,
) -> LayerFault {
    // A log may open with a warning, so the line comes from the first error.
    let first = |log: &'_ str| {
        log.lines()
            .find(|l| l.contains("error"))
            .unwrap_or(log)
            .to_string()
    };
    let vertex = d.vertex_glsl();
    let (log, glsl, own, part, call, needs) = match err.strip_prefix("vertex shader: ") {
        Some(e) => (
            first(e),
            vs,
            vertex.as_str(),
            LayerPart::Vertex,
            "glyph_place(",
            "vec2 glyph_place(vec2 em)",
        ),
        None => (
            first(err),
            fs,
            d.fragment_glsl(),
            LayerPart::Fragment,
            "ghee_main();",
            "a void main()",
        ),
    };
    // A slug layer has no distances: its glyph_dist and glyph_round_dist
    // #define to undeclared names, such as glyph_dist_needs_fill_mtsdf, so the
    // GL's compile error names one of those. This error names the fill in its
    // place.
    let slug = text::SLUG_UNDECLARED
        .iter()
        .find(|(_, s)| part == LayerPart::Fragment && log.contains(s));
    let err = match slug {
        Some((read, _)) => format!("{read} needs \"fill\": \"mtsdf\" on text '{}'", d.text),
        None => err.to_string(),
    };
    let fault = |message: String, line: Option<i64>, in_common: bool| LayerFault {
        message,
        part,
        line,
        common: in_common,
    };
    let Some(line) = error_line_number(&log) else {
        return fault(format!("{layer}: {err}"), None, false);
    };
    // The layer's own GLSL is the last match of it in the assembled source.
    let at = glsl
        .rfind(own)
        .map_or(0, |p| glsl[..p].matches('\n').count() as i64);
    let lines = own.lines().count() as i64;
    let key = part.key();
    if line > at && line <= at + lines {
        let n = line - at;
        let message = format!("{layer}, line {n} of its {key}: {err}");
        return fault(message, Some(n), false);
    }
    if let (Some(n), true) = user_line(&log, glsl, own, common) {
        let message = format!("{layer}, line {n} of the common block: {err}");
        return fault(message, Some(n), true);
    }
    // After the layer's own GLSL, ghee's main calls the function it defines.
    let src = usize::try_from(line - 1)
        .ok()
        .and_then(|i| glsl.lines().nth(i));
    if line > at + lines && src.is_some_and(|l| l.contains(call)) {
        return fault(format!("{layer}: its {key} needs {needs}"), None, false);
    }
    fault(format!("{layer}: {err}"), None, false)
}

/// The refusal of pass `name` with mipmaps at `w` x `h`, on a GPU that
/// mipmaps a texture of any size when `npot`, and a power-of-two size alone
/// otherwise.
fn mip_refusal(name: &str, w: i64, h: i64, npot: bool) -> Option<String> {
    let pot = |n: i64| n > 0 && n & (n - 1) == 0;
    (!(npot || pot(w) && pot(h))).then(|| {
        format!("pass '{name}' takes mipmaps at {w}x{h}, and this GPU mipmaps a power-of-two size alone")
    })
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
    /// The layer holding the error, in the pass of error_pass.
    pub error_layer: Option<ErrorLayer>,
    #[allow(dead_code)]
    pub display: bool,
    /// The readbacks keep the output pass's alpha. set_alpha() sets it.
    alpha: bool,
    /// pass name -> engine pass index, in the order setup() adds the passes.
    mp: Vec<(String, c_int)>,
    /// array name -> size, in declaration order.
    feeds: Vec<(String, ArraySize)>,
    /// The engine assets uploaded for the scene: its image assets, and the
    /// data texture `<name>_tex` of each text. A setup frees those that its
    /// new scene does not declare, so their asset slots come back.
    scene_assets: std::cell::RefCell<Vec<String>>,
    /// The engine index of each batch passed to setup_batches(), in order.
    batches: Vec<c_int>,
    /// The scene's layers, in the order of their passes.
    layers: Vec<LayerBatches>,
    /// The text problem causes logged since the setup, so each logs once.
    reported: std::cell::RefCell<HashSet<String>>,
    /// The problems of each scene text, by its name.
    text_errors: std::cell::RefCell<HashMap<String, TextErrors>>,
    /// Counts the changes to text_errors.
    text_errors_seq: std::cell::Cell<u64>,
    /// One atlas per font, px/em and range among the texts that a layer fills
    /// with MTSDF.
    atlases: std::cell::RefCell<Vec<AtlasSlot>>,
    /// The new words of each MTSDF text still waiting for their glyphs, by
    /// the text's name. The text draws its last words meanwhile.
    held: std::cell::RefCell<HashMap<String, Held>>,
    /// The words that set_text last took for each text, shown or held, by
    /// the text's name. The same words again lay out nothing.
    words: std::cell::RefCell<HashMap<String, String>>,
    /// The texts whose texture of curves a program of the scene reads: a
    /// pass or a layer that calls `<name>_cover`, or a layer of the slug fill.
    /// Any other text lays out with no such texture.
    curves: HashSet<String>,
    /// The colour atlas of each drawn text whose words held colour glyphs,
    /// by the text's name.
    colours: std::cell::RefCell<HashMap<String, ColourAtlas>>,
    /// set_time() pinned the time, so a render waits for held words.
    pinned: std::cell::Cell<bool>,
    /// The font of an atlas generator.
    atlas_font: AtlasFont,
    /// The atlas cache directory. None takes the generator's own.
    atlas_cache: Option<PathBuf>,
    /// A test sets it to compile the array readers that interpolate in the
    /// shader, on a GL that filters a float texture itself.
    #[cfg(test)]
    shader_lerp: bool,
    /// The layouts run by set_text, for a test to count.
    #[cfg(test)]
    layouts: std::cell::Cell<usize>,
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
            error_layer: None,
            display,
            alpha: false,
            mp: Vec::new(),
            feeds: Vec::new(),
            scene_assets: std::cell::RefCell::new(Vec::new()),
            batches: Vec::new(),
            layers: Vec::new(),
            reported: std::cell::RefCell::new(HashSet::new()),
            text_errors: std::cell::RefCell::new(HashMap::new()),
            text_errors_seq: std::cell::Cell::new(0),
            atlases: std::cell::RefCell::new(Vec::new()),
            held: std::cell::RefCell::new(HashMap::new()),
            words: std::cell::RefCell::new(HashMap::new()),
            curves: HashSet::new(),
            colours: std::cell::RefCell::new(HashMap::new()),
            pinned: std::cell::Cell::new(false),
            atlas_font: Box::new(|fonts| Arc::new(text::AtlasFont(fonts.to_vec()))),
            atlas_cache: None,
            #[cfg(test)]
            shader_lerp: false,
            #[cfg(test)]
            layouts: std::cell::Cell::new(0),
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
                bail!("{}", err_text(err));
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
    /// error_line then counts from the start of that block. An error in a
    /// layer's own GLSL sets error_layer too, and error_line counts in that
    /// GLSL.
    pub fn setup(&mut self, scene: &Scene) -> Result<String> {
        self.setup_batches(scene, &[])
    }

    /// setup() with batches. Each batch draws into its named pass, after that
    /// pass's quad and the pass's layers, in the order given. A batch's compile
    /// error reports its pass.
    pub(crate) fn setup_batches(&mut self, scene: &Scene, batches: &[Batch]) -> Result<String> {
        self.error = String::new();
        self.error_line = None;
        self.error_pass = None;
        self.error_layer = None;
        if scene.passes.is_empty() {
            self.error = "scene has no passes".to_string();
            return Ok(self.error.clone());
        }
        if scene.passes.len() > MAX_PASSES {
            let filters = scene.passes.iter().filter(|p| !p.filter.is_empty()).count();
            let of_them = match filters {
                0 => String::new(),
                1 => ", 1 of them a filter".to_string(),
                n => format!(", {n} of them filters"),
            };
            self.error = format!(
                "{} passes{of_them}, over the cap of {MAX_PASSES}",
                scene.passes.len()
            );
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
        let npot = self.mipmaps_npot();
        for p in scene.passes.iter().filter(|p| p.mipmaps) {
            let (w, h) = p
                .size
                .unwrap_or((i64::from(self.width), i64::from(self.height)));
            if let Some(e) = mip_refusal(&p.name, w, h, npot) {
                self.error = e;
                return Ok(self.error.clone());
            }
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
        if let Some((k, l)) = batches
            .iter()
            .enumerate()
            .find(|(_, l)| !names.contains(&l.pass.as_str()))
        {
            self.error = format!(
                "batch {k} draws into pass '{}', which is not a pass in this scene",
                l.pass
            );
            return Ok(self.error.clone());
        }
        // The engine counts the assets and the texts that each program
        // samples once it compiles. These counts come first, and a scene
        // that they refuse leaves the running scene in place.
        for p in &scene.passes {
            let mut used: Vec<&str> = p.arrays.iter().map(|a| a.name.as_str()).collect();
            used.sort_unstable();
            used.dedup();
            let want = p.channels.len() + used.len();
            let mut over = self.units_error(&format!("pass '{}'", p.name), want);
            // A layer samples its text's texture on top of its pass's inputs.
            if over.is_none() && !p.layers.is_empty() {
                over = self.units_error(&p.layer_what(0), want + 1);
            }
            if let Some(e) = over {
                self.error = e;
                return Ok(self.error.clone());
            }
        }
        if let Some(e) = scene.undeclared_layer() {
            self.error = e;
            return Ok(self.error.clone());
        }
        if let Some(e) = scene.passes.iter().find_map(Pass::empty) {
            self.error = e;
            return Ok(self.error.clone());
        }
        let mut mp: Vec<(String, c_int)> = Vec::new();
        let mut batch_ix: Vec<c_int> = vec![-1; batches.len()];
        let mut layers: Vec<LayerBatches> = Vec::new();
        let hw_linear = self.data_texture_linear();
        #[cfg(test)]
        let hw_linear = hw_linear && !self.shader_lerp;
        let total = scene.passes.len();
        // Every pass buffer stores alpha but the output pass's, which stores
        // it only when the readback keeps it or a channel reads it. An RGB
        // buffer reads back alpha 255 at no cost.
        let out = last.name.as_str();
        let out_read = scene
            .passes
            .iter()
            .any(|p| p.channels.iter().any(|c| c.buffer.as_deref() == Some(out)));
        for (i, p) in scene.passes.iter().enumerate() {
            // A VC4 takes a while over a big shader, and a scene of several
            // compiles them one after another with nothing to show for it.
            crate::report::progress("compile", format!("pass {} of {total}, {}", i + 1, p.name));
            let glsl = match p.has_code() {
                true => p.glsl(
                    Some(&scene.params),
                    scene.common.as_deref(),
                    &scene.assets,
                    &scene.texts,
                    hw_linear,
                ),
                false => String::new(),
            };
            let cname = self.staged_cstring(&format!("pass '{}'", p.name))?;
            let cglsl = self.staged_cstring(&glsl)?;
            let alpha = i + 1 < total || self.alpha || out_read;
            let mut idx = unsafe {
                ffi::ghee_mp_add_pass(
                    self.handle,
                    cname.as_ptr(),
                    match p.has_code() {
                        true => cglsl.as_ptr(),
                        false => ptr::null(),
                    },
                    c_int::from(alpha),
                )
            };
            if idx >= 0 {
                if let Some((w, h)) = p.size {
                    if !unsafe {
                        ffi::ghee_mp_set_pass_size(self.handle, idx, w as c_int, h as c_int)
                    } {
                        idx = -1;
                    }
                }
            }
            if idx >= 0 && p.mipmaps && !unsafe { ffi::ghee_mp_set_pass_mipmaps(self.handle, idx) }
            {
                idx = -1;
            }
            for (k, d) in p.layers.iter().enumerate() {
                if idx < 0 {
                    break;
                }
                let (vs, fs) = p.layer_glsl(k, scene, hw_linear);
                let name = p.layer_what(k);
                let (cname, cvs, cfs) = (
                    self.staged_cstring(&name)?,
                    self.staged_cstring(&vs)?,
                    self.staged_cstring(&fs)?,
                );
                let li = unsafe {
                    ffi::ghee_mp_add_batch(
                        self.handle,
                        idx,
                        cname.as_ptr(),
                        cvs.as_ptr(),
                        cfs.as_ptr(),
                        d.blend.engine(),
                    )
                };
                if li < 0 {
                    let err = or_else(self.last_error(), "compile error");
                    unsafe { ffi::ghee_mp_abort(self.handle) };
                    let f = layer_error(&name, d, &err, &vs, &fs, scene.common.as_deref());
                    self.error = f.message;
                    self.error_line = f.line;
                    if f.common {
                        self.error_pass = Some(ErrorPass::Common);
                    } else {
                        self.error_pass = Some(ErrorPass::Index(i));
                        self.error_layer = Some(ErrorLayer {
                            index: k,
                            part: f.part,
                        });
                    }
                    crate::report::progress_done();
                    return Ok(self.error.clone());
                }
                layers.push(LayerBatches {
                    batch: li,
                    colour: -1,
                    boxes: -1,
                    text: d.text.clone(),
                    name,
                });
            }
            // The colour glyphs of each layer come after every layer of the
            // pass, and the error boxes after them, so none hides an error
            // box.
            let first = layers.len() - if idx < 0 { 0 } else { p.layers.len() };
            for (k, dl) in layers[first..].iter_mut().enumerate() {
                let coloured = p.paints_emoji(k)
                    && scene
                        .texts
                        .iter()
                        .any(|t| t.name == dl.text && t.has_colour());
                if !coloured {
                    continue;
                }
                let (vs, fs) = p.layer_colour_glsl(k, scene, hw_linear);
                let name = format!("{} colour glyphs", dl.name);
                let (cname, cvs, cfs) = (
                    self.staged_cstring(&name)?,
                    self.staged_cstring(&vs)?,
                    self.staged_cstring(&fs)?,
                );
                dl.colour = unsafe {
                    ffi::ghee_mp_add_batch(
                        self.handle,
                        idx,
                        cname.as_ptr(),
                        cvs.as_ptr(),
                        cfs.as_ptr(),
                        p.layers[k].blend.engine(),
                    )
                };
                if dl.colour < 0 {
                    let err = or_else(self.last_error(), "compile error");
                    unsafe { ffi::ghee_mp_abort(self.handle) };
                    self.error = format!("{name}: {err}");
                    self.error_pass = Some(ErrorPass::Index(i));
                    crate::report::progress_done();
                    return Ok(self.error.clone());
                }
            }
            for (k, dl) in layers[first..].iter_mut().enumerate() {
                let (vs, fs) = p.layer_box_glsl(k, scene, hw_linear);
                let name = format!("{} error boxes", dl.name);
                let (cname, cvs, cfs) = (
                    self.staged_cstring(&name)?,
                    self.staged_cstring(&vs)?,
                    self.staged_cstring(&fs)?,
                );
                dl.boxes = unsafe {
                    ffi::ghee_mp_add_batch(
                        self.handle,
                        idx,
                        cname.as_ptr(),
                        cvs.as_ptr(),
                        cfs.as_ptr(),
                        Blend::Over.engine(),
                    )
                };
                if dl.boxes < 0 {
                    let err = or_else(self.last_error(), "compile error");
                    unsafe { ffi::ghee_mp_abort(self.handle) };
                    self.error = format!("{name}: {err}");
                    self.error_pass = Some(ErrorPass::Index(i));
                    crate::report::progress_done();
                    return Ok(self.error.clone());
                }
            }
            for (k, l) in batches.iter().enumerate().filter(|(_, l)| l.pass == p.name) {
                if idx < 0 {
                    break;
                }
                let name = format!("batch {k}");
                let (cname, vs, fs) = (
                    self.staged_cstring(&name)?,
                    self.staged_cstring(&l.vertex)?,
                    self.staged_cstring(&l.fragment)?,
                );
                batch_ix[k] = unsafe {
                    ffi::ghee_mp_add_batch(
                        self.handle,
                        idx,
                        cname.as_ptr(),
                        vs.as_ptr(),
                        fs.as_ptr(),
                        l.blend.engine(),
                    )
                };
                if batch_ix[k] < 0 {
                    let err = or_else(self.last_error(), "compile error");
                    unsafe { ffi::ghee_mp_abort(self.handle) };
                    self.error = format!("{name}: {err}");
                    self.error_pass = Some(ErrorPass::Index(i));
                    crate::report::progress_done();
                    return Ok(self.error.clone());
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
        self.batches = batch_ix;
        self.layers = layers;
        // A new scene logs its text problems afresh.
        self.reported.borrow_mut().clear();
        if !self.text_errors.borrow().is_empty() {
            self.text_errors.borrow_mut().clear();
            self.text_errors_seq.set(self.text_errors_seq.get() + 1);
        }
        self.held.borrow_mut().clear();
        self.words.borrow_mut().clear();
        self.curves.clear();
        for t in &scene.texts {
            let tex = CString::new(text::internal(&t.name, "_tex"))?;
            if unsafe { ffi::ghee_mp_reads(self.handle, tex.as_ptr()) } {
                self.curves.insert(t.name.clone());
            }
        }
        // A colour atlas starts again with the words that need it.
        self.colours.borrow_mut().clear();
        self.set_atlases(scene)?;
        // The engine assets of the old scene that the new one does not declare
        // go, and their slots come back.
        let atlases: Vec<String> = self
            .atlases
            .borrow()
            .iter()
            .flat_map(|s| s.textures.iter().map(|t| t.0.clone()))
            .collect();
        let keep: Vec<String> = scene
            .assets
            .iter()
            .filter(|a| !a.is_font())
            .map(|a| a.name.clone())
            .chain(scene.texts.iter().map(|t| text::internal(&t.name, "_tex")))
            .chain(atlases)
            .collect();
        let stale: Vec<String> = self
            .scene_assets
            .borrow()
            .iter()
            .filter(|n| !keep.contains(n))
            .cloned()
            .collect();
        for name in stale {
            let cname = CString::new(name.as_str())?;
            unsafe { ffi::ghee_mp_drop_asset(self.handle, cname.as_ptr()) };
            self.scene_assets.borrow_mut().retain(|n| n != &name);
        }
        // The output aspect reaches every pass that declares it.
        // A pass with its own size reads that size off `resolution`.
        // Such a pass cannot derive the shape of the frame.
        // A scene param of this name overwrites the value below.
        self.set_scene_param(
            "aspect",
            "float",
            &ParamValue::Float(f64::from(self.width) / f64::from(self.height.max(1))),
        )?;
        // Assets go in before the channels and the arrays, so the engine's
        // texture-unit budget counts them.
        for a in scene.assets.iter().filter(|a| !a.is_font()) {
            let Some(img) = &a.image else {
                self.error = format!(
                    "asset '{}': '{}' was never loaded, the scene was built without a directory",
                    a.name, a.file
                );
                return Ok(self.error.clone());
            };
            let data = match &img.pixels {
                Pixels::U8(d) => AssetData::U8(d),
                Pixels::F32(d) => AssetData::F32(d),
            };
            // A pass past its texture units refuses the scene here, as
            // anywhere else.
            if let Err(e) = self.set_asset(
                &a.name,
                img.width,
                img.height,
                img.components,
                a.linear,
                data,
            ) {
                self.error = e.to_string();
                return Ok(self.error.clone());
            }
            self.scene_asset(&a.name);
        }
        // A text builds from the scene's string param of its name, through
        // set_text below, and it looks itself up in self.scene.
        self.scene = Some(scene.clone());
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
        // A scene param sits in every preamble, so it reaches every pass. A
        // text that the engine refuses, or one whose font was never loaded,
        // refuses the scene, as an asset does.
        for (nm, pm) in scene.params.iter() {
            if let ParamValue::Str(words) = &pm.value {
                if let Err(e) = self.set_text(nm, words) {
                    self.error = e.to_string();
                    return Ok(self.error.clone());
                }
                continue;
            }
            self.set_scene_param(nm, &pm.ty, &pm.value)?;
        }
        // The first frame draws the words: they wait here for their glyphs.
        if let Err(e) = self.tick_texts(true) {
            self.error = e.to_string();
            return Ok(self.error.clone());
        }

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

    /// The quads of batch `batch`, an index into the batches passed to
    /// setup_batches(): 4 vertices a quad of `nattr` vec4s each, so 16 * nattr
    /// floats a quad. They replace the last set.
    #[cfg(test)]
    pub(crate) fn set_batch_quads(&self, batch: usize, data: &[f32], nattr: usize) -> Result<()> {
        let Some(&idx) = self.batches.get(batch) else {
            let n = self.batches.len();
            bail!(
                "no batch {batch}: setup_batches() took {n} {}",
                if n == 1 { "batch" } else { "batches" }
            );
        };
        self.batch_quads(idx, &format!("batch {batch}"), data, nattr)
    }

    /// Hand the engine batch `idx` its quads. `batch` names it in an error.
    fn batch_quads(&self, idx: c_int, batch: &str, data: &[f32], nattr: usize) -> Result<()> {
        if !(1..=4).contains(&nattr) {
            bail!("{batch}: a vertex holds 1 to 4 attributes, got {nattr}");
        }
        let per_quad = 16 * nattr;
        if !data.len().is_multiple_of(per_quad) {
            bail!(
                "{batch}: a quad takes {per_quad} floats, and {} is not a multiple of {per_quad}",
                data.len()
            );
        }
        let quads = c_int::try_from(data.len() / per_quad)?;
        if !unsafe {
            ffi::ghee_mp_set_batch_quads(self.handle, idx, data.as_ptr(), quads, nattr as c_int)
        } {
            bail!(
                "{}",
                or_else(self.last_error(), &format!("{batch} refused its quads"))
            );
        }
        Ok(())
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
    /// out_size=(w, h) reads back a GPU-scaled frame at that size, and a whole
    /// multiple of the render size repeats each pixel as a square block.
    /// None reads back the full frame. present() scales the same way.
    pub fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()> {
        self.feed(provider)?;
        let (pw, ph) = out_size.unwrap_or((0, 0));
        // The engine scales only when both dimensions are positive.
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

    /// render() reading back a GPU-scaled frame into a fresh buffer.
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

    /// Upload each declared array for this frame, once per unique name, and
    /// the glyphs that the atlases gained.
    ///
    /// A render uploads what the provider returns. An Unchanged answer states
    /// that the texture already holds this array, and the upload is skipped.
    /// The engine never caches or compares data.
    fn feed(&self, provider: Option<Provider<'_>>) -> Result<()> {
        self.tick_texts(false)?;
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
        let filter = match linear {
            true => ffi::GHEE_FILTER_LINEAR,
            false => ffi::GHEE_FILTER_NEAREST,
        };
        self.put_asset(name, width, height, components, filter, data)
    }

    /// set_asset() with a filter of the engine's: GHEE_FILTER_NEAREST,
    /// GHEE_FILTER_LINEAR, or GHEE_FILTER_MIPMAP for 8-bit pixels at a
    /// power-of-two size.
    fn put_asset(
        &self,
        name: &str,
        width: u32,
        height: u32,
        components: u32,
        filter: c_int,
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
        let ok = unsafe {
            ffi::ghee_mp_set_asset(
                self.handle,
                cname.as_ptr(),
                ptr,
                width as c_int,
                height as c_int,
                components as c_int,
                c_int::from(is_f32),
                filter,
            )
        };
        if !ok {
            bail!(
                "{}",
                or_else(self.last_error(), &format!("asset '{name}' refused"))
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
    /// A "color" param is [r, g, b] and sets a vec3 uniform. A "vecK" param
    /// is K floats and sets a vecK uniform.
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
        let fits = match (ptype, value) {
            ("color", v) => matches!(v, ParamValue::Color(_)),
            ("vec2", ParamValue::Vec(v)) => v.len() == 2,
            ("vec3", ParamValue::Vec(v)) => v.len() == 3,
            ("vec4", ParamValue::Vec(v)) => v.len() == 4,
            ("vec2" | "vec3" | "vec4", _) => false,
            _ => true,
        };
        if !fits {
            match ptype {
                "color" => bail!("color param '{name}' needs three numbers [r, g, b]"),
                _ => bail!("{ptype} param '{name}' needs {} numbers", &ptype[3..]),
            }
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
        if let ParamValue::Str(words) = value {
            return self.set_text(name, words);
        }
        let oks = self
            .mp
            .iter()
            .map(|(pn, _)| self.set_param(pn, name, ptype, value))
            .collect::<Result<Vec<bool>>>()?;
        Ok(oks.into_iter().any(|o| o))
    }

    /// Lay out new words for the scene text `name` and upload them, with no
    /// recompile. Returns false when the scene holds no text of that name.
    /// The words that the text already holds return at once.
    ///
    /// A text problem fails nothing: the text draws an error box for each
    /// character that it cannot draw, and words that fit no texture leave
    /// the last words that did. Each problem logs once, and text_errors()
    /// holds it while it lasts. An error is a font that was never loaded, or
    /// a texture or quads that the engine refuses.
    ///
    /// New words of an MTSDF text whose glyphs the atlas does not hold yet
    /// wait for them, and the text draws its last words meanwhile. A render
    /// with the time pinned by set_time() waits for them, and so does a setup.
    pub fn set_text(&self, name: &str, words: &str) -> Result<bool> {
        let Some(text) = self
            .scene
            .as_ref()
            .and_then(|s| s.texts.iter().find(|t| t.name == name))
        else {
            return Ok(false);
        };
        if self.words.borrow().get(name).is_some_and(|w| w == words) {
            return Ok(true);
        }
        #[cfg(test)]
        self.layouts.set(self.layouts.get() + 1);
        let laid = match self.curves.contains(name) {
            true => text.build(words)?,
            false => text.place(words)?,
        };
        self.held.borrow_mut().remove(name);
        if laid.built.is_some() {
            if let Some(i) = self.atlas_of(name) {
                if !self.atlases.borrow()[i]
                    .gen
                    .request(words.chars())
                    .is_empty()
                {
                    let held = Held {
                        words: words.to_string(),
                        laid,
                    };
                    self.held.borrow_mut().insert(name.to_string(), held);
                    self.words
                        .borrow_mut()
                        .insert(name.to_string(), words.to_string());
                    return Ok(true);
                }
            }
        }
        self.show(text, laid)?;
        self.words
            .borrow_mut()
            .insert(name.to_string(), words.to_string());
        Ok(true)
    }

    /// The index in `atlases` of the atlas that the layers of text `name`
    /// read, None when no layer of it takes the MTSDF fill.
    fn atlas_of(&self, name: &str) -> Option<usize> {
        let tex = text::internal(name, "_atlas");
        self.atlases
            .borrow()
            .iter()
            .position(|s| s.textures.iter().any(|t| t.0 == tex))
    }

    /// Draw `laid`, new words of `text`, from now on, and log their problems.
    /// Words that did not lay out leave the last words drawn.
    fn show(&self, text: &Text, laid: Layout) -> Result<()> {
        let name = &text.name;
        let Layout { built, mut faults } = laid;
        let Some(built) = built else {
            self.text_faults(text, &faults, false);
            return Ok(());
        };
        // The quads of the text's fill, and under MTSDF an error box for each
        // glyph that the atlas holds no field for. An MTSDF text with no atlas
        // has no font with outlines left, and still draws its colour glyphs.
        let mut boxes = built.boxes.floats.clone();
        let quads = match text.fill {
            Fill::Slug => built.quads.floats.clone(),
            Fill::Mtsdf => {
                let atlases = self.atlases.borrow();
                let slot = self.atlas_of(name).map(|i| &atlases[i]);
                let made = slot.and_then(|s| {
                    s.gen.with(|a| {
                        text::atlas_quads(&built.glyphs, |p| {
                            let cell = a.cells.get(&text::atlas_id(p.font, p.gid));
                            Some((*cell?, f64::from(s.px_em)))
                        })
                    })
                });
                match (slot, made) {
                    (Some(slot), Some((quads, more, missing))) => {
                        boxes.extend(&more.floats);
                        let mut told: Vec<char> = Vec::new();
                        for p in missing {
                            if told.contains(&p.ch) {
                                continue;
                            }
                            told.push(p.ch);
                            let why = match slot.gen.failure(text::atlas_id(p.font, p.gid)) {
                                Some(e) => format!("has no field in the atlas: {e}"),
                                None => "has no field in the atlas".to_string(),
                            };
                            faults.push(Fault::Glyph(p.ch, why));
                        }
                        quads.floats
                    }
                    _ => Vec::new(),
                }
            }
        };
        let colour = self.colour_quads(text, &built.colour, &mut faults)?;
        boxes.extend(&colour.1);
        self.text_faults(text, &faults, true);
        // A layout with no texture of curves leaves the texture alone.
        if built.width > 0 {
            self.set_asset(
                &text::internal(name, "_tex"),
                built.width,
                built.height,
                4,
                false,
                AssetData::U8(&built.texels),
            )?;
            self.scene_asset(&text::internal(name, "_tex"));
        }
        for (suffix, v) in &built.uniforms {
            let uniform = text::internal(name, suffix);
            for (_, idx) in &self.mp {
                self.set_uniform(*idx, &uniform, "float", Value::from(*v))?;
            }
        }
        let size = Value::from(built.size.to_vec());
        for (_, idx) in &self.mp {
            self.set_uniform(*idx, &format!("{name}_size"), "vec2", size.clone())?;
        }
        for d in self.layers.iter().filter(|d| d.text == *name) {
            for (batch, floats, what) in [
                (d.batch, &quads, d.name.clone()),
                (d.colour, &colour.0, format!("{} colour glyphs", d.name)),
                (d.boxes, &boxes, format!("{} error boxes", d.name)),
            ] {
                if batch >= 0 {
                    self.batch_quads(batch, &what, floats, text::QUAD_FLOATS / 4)?;
                }
            }
        }
        Ok(())
    }

    /// The quads of the colour glyphs `glyphs` of `text`, for its layers'
    /// colour batches, and an error box for each glyph whose bitmap failed,
    /// with a fault in `faults`. The first colour glyph of the text makes its
    /// colour atlas, and every new one decodes into it and uploads.
    fn colour_quads(
        &self,
        text: &Text,
        glyphs: &[text::Placement],
        faults: &mut Vec<Fault>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let name = &text.name;
        let drawn = self.layers.iter().any(|d| d.text == *name && d.colour >= 0);
        if glyphs.is_empty() || !drawn {
            return Ok((Vec::new(), Vec::new()));
        }
        let fonts = text.data.as_deref().unwrap_or_default();
        let mut colours = self.colours.borrow_mut();
        let slot = colours.entry(name.clone()).or_default();
        let mut cells: Vec<crate::colour::Rect> = Vec::new();
        for p in glyphs {
            let id = text::atlas_id(p.font, p.gid);
            if slot.atlas.cells.contains_key(&id) || slot.failed.contains_key(&id) {
                continue;
            }
            let face = fonts.get(p.font).and_then(|f| f.as_ref().ok());
            let face = face.and_then(|d| ttf_parser::Face::parse(d, 0).ok());
            let made = match face {
                Some(face) => crate::colour::field(&face, p.gid),
                None => Err(anyhow!("its font is unreadable")),
            };
            match made.and_then(|(f, s)| Ok((slot.atlas.insert(id, &f)?, s))) {
                Ok((cell, s)) => {
                    slot.scale.insert(id, s);
                    cells.push(cell.rect);
                }
                Err(e) => {
                    slot.failed.insert(id, format!("{e:#}"));
                }
            }
        }
        // A new atlas's first update holds all of it, as does a grown one's,
        // and uploads every level whole. Otherwise the new cells upload the
        // rects that they changed in each level.
        if let Some(resized) = slot.atlas.take_change() {
            let tex = text::internal(name, "_emoji");
            let (w, h) = (slot.atlas.width, slot.atlas.height);
            if resized {
                slot.mips = crate::colour::Mips::new(w, h);
                let top = slot.atlas.top();
                slot.mips.refresh(&slot.atlas.texels, w, h, [0, 0, w, top]);
                let data = AssetData::U8(&slot.atlas.texels);
                self.put_asset(&tex, w, h, 4, ffi::GHEE_FILTER_MIPMAP, data)?;
                self.scene_asset(&tex);
                for k in 1..=slot.mips.count() {
                    let (lw, lh, texels) = slot.mips.level(k);
                    self.set_asset_rect(&tex, k, [0, 0, lw, lh], texels)?;
                }
            } else {
                let changed = slot.mips.refresh_cells(&slot.atlas.texels, w, h, &cells);
                for (level, rect) in changed {
                    let (lw, texels) = match level {
                        0 => (w, &slot.atlas.texels[..]),
                        _ => {
                            let (lw, _, texels) = slot.mips.level(level);
                            (lw, texels)
                        }
                    };
                    let texels = crate::colour::crop(texels, lw, rect);
                    self.set_asset_rect(&tex, level, rect, &texels)?;
                }
            }
        }
        let (quads, boxes, missing) = text::atlas_quads(glyphs, |p| {
            let id = text::atlas_id(p.font, p.gid);
            Some((*slot.atlas.cells.get(&id)?, slot.scale[&id]))
        });
        let mut told: Vec<char> = Vec::new();
        for p in missing {
            if !told.contains(&p.ch) {
                told.push(p.ch);
                let why = &slot.failed[&text::atlas_id(p.font, p.gid)];
                faults.push(Fault::Glyph(
                    p.ch,
                    format!("has no bitmap in the colour atlas: {why}"),
                ));
            }
        }
        Ok((quads.floats, boxes.floats))
    }

    /// Set up the atlases of `scene`: one generator per font, px/em and range
    /// among its texts that a layer fills with MTSDF. A generator that the last
    /// scene used carries on, and the others stop.
    fn set_atlases(&self, scene: &Scene) -> Result<()> {
        let mut old = std::mem::take(&mut *self.atlases.borrow_mut());
        let mut slots: Vec<AtlasSlot> = Vec::new();
        for t in &scene.texts {
            let Some(fonts) = &t.data else { continue };
            if t.fill != Fill::Mtsdf || !scene.has_layer_of(&t.name) || !t.has_outlines() {
                continue;
            }
            let range = scene.atlas_range(&t.name);
            let same = |s: &AtlasSlot| {
                s.px_em == t.atlas_font_size && s.range == range && s.fonts == *fonts
            };
            let i = match slots.iter().position(same) {
                Some(i) => i,
                None => {
                    let slot = match old.iter().position(same) {
                        Some(j) => {
                            let mut s = old.swap_remove(j);
                            s.textures.clear();
                            s
                        }
                        None => AtlasSlot {
                            fonts: fonts.clone(),
                            px_em: t.atlas_font_size,
                            range,
                            gen: atlas::Generator::new(
                                (self.atlas_font)(fonts),
                                atlas::Config {
                                    px_em: t.atlas_font_size,
                                    range,
                                    cache: self.atlas_cache.clone(),
                                },
                            )?,
                            textures: Vec::new(),
                        },
                    };
                    slots.push(slot);
                    slots.len() - 1
                }
            };
            slots[i]
                .textures
                .push((text::internal(&t.name, "_atlas"), true));
        }
        drop(old);
        *self.atlases.borrow_mut() = slots;
        Ok(())
    }

    /// Hand the engine what the atlases gained, then draw the held words
    /// whose glyphs are all in. With `wait`, or with the time pinned, every
    /// held text waits for its glyphs first.
    fn tick_texts(&self, wait: bool) -> Result<()> {
        if self.atlases.borrow().is_empty() {
            return Ok(());
        }
        let wait = wait || self.pinned.get();
        let mut ready: Vec<String> = Vec::new();
        for (name, h) in self.held.borrow().iter() {
            let Some(i) = self.atlas_of(name) else {
                ready.push(name.clone());
                continue;
            };
            let atlases = self.atlases.borrow();
            let done = loop {
                if atlases[i].gen.request(h.words.chars()).is_empty() {
                    break true;
                }
                if !wait {
                    break false;
                }
                thread::sleep(Duration::from_millis(1));
            };
            if done {
                ready.push(name.clone());
            }
        }
        self.upload_atlases()?;
        for name in ready {
            let Some(h) = self.held.borrow_mut().remove(&name) else {
                continue;
            };
            let scene = self.scene.as_ref();
            if let Some(text) = scene.and_then(|s| s.texts.iter().find(|t| t.name == name)) {
                self.show(text, h.laid)?;
            }
        }
        Ok(())
    }

    /// Upload the texels that each atlas gained: all of it to a texture new to
    /// the atlas, and to the others the rows that changed, or all of it again
    /// when the atlas grew.
    fn upload_atlases(&self) -> Result<()> {
        let mut atlases = self.atlases.borrow_mut();
        for slot in atlases.iter_mut() {
            let update = slot.gen.update();
            for (tex, fresh) in slot.textures.iter_mut() {
                if *fresh {
                    let up = slot.gen.with(|a| {
                        let data = AssetData::U8(&a.texels);
                        self.put_asset(tex, a.width, a.height, 4, ffi::GHEE_FILTER_LINEAR, data)
                    });
                    let Some(up) = up else { continue };
                    up?;
                    *fresh = false;
                    self.scene_asset(tex);
                    continue;
                }
                let Some(u) = &update else { continue };
                if u.resized {
                    let data = AssetData::U8(&u.rows[0].texels);
                    self.put_asset(tex, u.width, u.height, 4, ffi::GHEE_FILTER_LINEAR, data)?;
                    continue;
                }
                for r in &u.rows {
                    self.set_asset_rows(tex, r.y, u.width, &r.texels)?;
                }
            }
        }
        Ok(())
    }

    /// Replace `rect` of mip level `level` of asset `name` with `texels`,
    /// RGBA8 rows of the rect's width.
    fn set_asset_rect(
        &self,
        name: &str,
        level: usize,
        rect: crate::colour::Rect,
        texels: &[u8],
    ) -> Result<()> {
        let [x, y, w, h] = rect;
        if texels.len() != w as usize * h as usize * 4 {
            bail!(
                "asset '{name}': {} bytes are no {w}x{h} rect of RGBA8",
                texels.len()
            );
        }
        let cname = CString::new(name)?;
        let ok = unsafe {
            ffi::ghee_mp_set_asset_rect(
                self.handle,
                cname.as_ptr(),
                level as c_int,
                x as c_int,
                y as c_int,
                w as c_int,
                h as c_int,
                texels.as_ptr().cast::<c_void>(),
            )
        };
        if !ok {
            bail!(
                "{}",
                or_else(
                    self.last_error(),
                    &format!("asset '{name}' refused level {level}")
                )
            );
        }
        Ok(())
    }

    /// Replace the rows of asset `name` from row `y` up with `texels`, RGBA8
    /// rows `width` texels wide.
    fn set_asset_rows(&self, name: &str, y: u32, width: u32, texels: &[u8]) -> Result<()> {
        let row = width as usize * 4;
        if row == 0 || !texels.len().is_multiple_of(row) {
            bail!(
                "asset '{name}': {} bytes are no whole rows of {width} texels",
                texels.len()
            );
        }
        let cname = CString::new(name)?;
        let ok = unsafe {
            ffi::ghee_mp_set_asset_rows(
                self.handle,
                cname.as_ptr(),
                texels.as_ptr().cast::<c_void>(),
                y as c_int,
                (texels.len() / row) as c_int,
            )
        };
        if !ok {
            bail!(
                "{}",
                or_else(
                    self.last_error(),
                    &format!("asset '{name}' refused its rows")
                )
            );
        }
        Ok(())
    }

    /// Log each new cause among the `faults` of a layout of `text`, and hold
    /// them for text_errors(). Words that did not lay out, `drawn` false,
    /// leave the problems of the words still on screen.
    fn text_faults(&self, text: &Text, faults: &[Fault], drawn: bool) {
        let name = &text.name;
        let mut reported = self.reported.borrow_mut();
        // True the first time since the setup that a cause comes up.
        let mut fresh = |cause: String| reported.insert(cause);
        let no_outline = |f: usize, cs: &[char]| {
            let list: Vec<String> = cs.iter().map(|c| format!("{c:?}")).collect();
            format!(
                "text '{name}': {} has no outline for {}",
                text.font_name(f),
                list.join(", ")
            )
        };
        let mut now = match drawn {
            true => TextErrors::default(),
            false => TextErrors {
                refused: None,
                ..self
                    .text_errors
                    .borrow()
                    .get(name)
                    .cloned()
                    .unwrap_or_default()
            },
        };
        for f in faults {
            let line = match f {
                // The reason names the asset, which every text in the font
                // shares.
                Fault::Font(why) => {
                    if fresh(format!("font\0{why}")) {
                        crate::report::warn(why);
                    }
                    why.clone()
                }
                // Each character logs once.
                Fault::NoOutline(f, chars) => {
                    let new: Vec<char> = chars
                        .iter()
                        .copied()
                        .filter(|c| fresh(format!("ink\0{name}\0{c}")))
                        .collect();
                    if !new.is_empty() {
                        crate::report::warn(no_outline(*f, &new));
                    }
                    no_outline(*f, chars)
                }
                Fault::Glyph(ch, why) => format!("text '{name}': {ch:?} {why}"),
                Fault::Layout(why) => format!("text '{name}': {why}"),
            };
            if !matches!(f, Fault::Font(_) | Fault::NoOutline(..)) && fresh(line.clone()) {
                crate::report::warn(&line);
            }
            match f {
                Fault::Layout(_) => now.refused = Some(line),
                _ if drawn => now.drawn.push(line),
                _ => {}
            }
        }
        drop(reported);
        let mut all = self.text_errors.borrow_mut();
        let empty = TextErrors::default();
        if all.get(name).unwrap_or(&empty) != &now {
            self.text_errors_seq.set(self.text_errors_seq.get() + 1);
            if now == empty {
                all.remove(name);
            } else {
                all.insert(name.clone(), now);
            }
        }
    }

    /// The problems of the scene's texts, a line each: a font that failed to
    /// load, characters drawn as error boxes, words that do not fit.
    pub fn text_errors(&self) -> Vec<String> {
        let all = self.text_errors.borrow();
        let Some(scene) = self.scene.as_ref() else {
            return Vec::new();
        };
        let mut out: Vec<String> = Vec::new();
        for t in &scene.texts {
            if let Some(e) = all.get(&t.name) {
                for line in e.drawn.iter().chain(&e.refused) {
                    // Texts in one font share its failure.
                    if !out.contains(line) {
                        out.push(line.clone());
                    }
                }
            }
        }
        out
    }

    /// A count that changes whenever text_errors() does.
    pub fn text_errors_seq(&self) -> u64 {
        self.text_errors_seq.get()
    }

    /// Record `name` among the engine assets uploaded for the scene.
    fn scene_asset(&self, name: &str) {
        let mut names = self.scene_assets.borrow_mut();
        if !names.iter().any(|n| n == name) {
            names.push(name.to_string());
        }
    }

    /// `s` as a C string for a call that stages a pass or a batch. A NUL in
    /// `s` drops the staged passes and batches first, so no later commit runs
    /// them.
    fn staged_cstring(&self, s: &str) -> Result<CString> {
        CString::new(s).map_err(|e| {
            unsafe { ffi::ghee_mp_abort(self.handle) };
            crate::report::progress_done();
            e.into()
        })
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
        self.pinned.set(t.is_some());
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

    /// The texture units one program of a pass binds at most: the pass's
    /// channels and arrays, then the assets and the texts that the program
    /// samples. It is the GPU's own count, capped at 16.
    pub fn max_texture_units(&self) -> i32 {
        unsafe { ffi::ghee_max_texture_units(self.handle) }
    }

    /// The error for `what`, a pass or a layer, when `want` texture units are
    /// past max_texture_units(). The engine words its own the same way.
    fn units_error(&self, what: &str, want: usize) -> Option<String> {
        let units = self.max_texture_units();
        if want as i64 <= i64::from(units) {
            return None;
        }
        let has = match unsafe { ffi::ghee_gl_texture_units(self.handle) } > units {
            true => "a pass binds at most",
            false => "this GPU has",
        };
        Some(format!("{what} needs {want} texture units, {has} {units}"))
    }

    /// Keep the output pass's alpha in 4-component frames. Off by default,
    /// and every frame then reads alpha 255. A 3-component frame and the
    /// scanout carry no alpha, and the H264 input stays opaque. It takes
    /// effect at the next setup().
    pub fn set_alpha(&mut self, keep: bool) {
        self.alpha = keep;
        unsafe { ffi::ghee_keep_alpha(self.handle, c_int::from(keep)) };
    }

    /// True when this GPU linear-filters the float textures that carry arrays.
    /// V3D returns false. It has no OES_texture_float_linear.
    /// A filtered fetch there returns the nearest texel.
    pub fn data_texture_linear(&self) -> bool {
        unsafe { ffi::ghee_data_texture_linear(self.handle) }
    }

    /// True when the GPU mipmaps a pass buffer of any size, false when it
    /// mipmaps a power-of-two size alone.
    pub fn mipmaps_npot(&self) -> bool {
        unsafe { ffi::ghee_mipmaps_npot(self.handle) }
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
        self.batches = Vec::new();
        self.layers = Vec::new();
        self.feeds = Vec::new();
        self.held.borrow_mut().clear();
        self.words.borrow_mut().clear();
        self.atlases.borrow_mut().clear();
        self.colours.borrow_mut().clear();
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
    fn gpu() -> (MutexGuard<'static, ()>, Renderer) {
        gpu_at(W, H)
    }

    /// gpu() at a size of its own.
    fn gpu_at(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
        let guard = lock();
        let mut errs = Vec::new();
        for dev in probe_dri_devices() {
            match Renderer::new(
                w as i32,
                h as i32,
                COMPONENTS as i32,
                Some(&dev),
                false,
                false,
            ) {
                Ok(r) => return (guard, r),
                Err(e) => errs.push(format!("{dev}: {e}")),
            }
        }
        panic!(
            "no GL device opens ({}). A host without one runs the tests under \
             RUSTFLAGS=\"--cfg ghee_no_gpu\", which reports them as ignored",
            errs.join("; ")
        );
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

    /// The refusal of `what`, which needs `want` texture units, on this GPU.
    fn over_units(r: &Renderer, what: &str, want: usize) -> String {
        let units = r.max_texture_units();
        match unsafe { ffi::ghee_gl_texture_units(r.handle()) } > 16 {
            true => format!("{what} needs {want} texture units, a pass binds at most {units}"),
            false => format!("{what} needs {want} texture units, this GPU has {units}"),
        }
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
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
    fn a_present_frees_the_buffer_that_an_accepted_flip_takes_off_the_glass() {
        let [a, b, c] = [1, 2, 3].map(ptr::without_provenance_mut::<c_void>);
        let (mut prev, mut shown) = (a, b);
        // A flip still waiting on the last one, or refused, leaves the glass
        // as it was, and frees the buffer just drawn.
        let freed = unsafe { ffi::ghee_flip_rotate(&mut prev, &mut shown, c, false) };
        assert_eq!((freed, prev, shown), (c, a, b));
        // An accepted flip leaves the last buffer on the glass until it
        // completes, and frees the one before.
        let freed = unsafe { ffi::ghee_flip_rotate(&mut prev, &mut shown, c, true) };
        assert_eq!((freed, prev, shown), (b, c, a));
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn setup_rejects_a_scene_with_no_passes() {
        let (_g, mut r) = gpu();
        let empty = Scene::of(Vec::new()).unwrap();
        assert!(r.setup(&empty).unwrap().contains("no passes"));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn channel_naming_no_pass_is_rejected() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn channel_bound_to_nothing_is_allowed() {
        // buffer=None is the unbound channel. The scene stays valid.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([
            {"name": "a", "channels": [{"buffer": null}], "code": FLAT},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn setup_rejects_a_scene_in_a_fixed_order() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn unbound_channel_reads_black() {
        // 'leak' leaves a white texture bound on unit 0.
        // Channel 0 of 'main' is unbound, and it reads black.
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn every_pass_reads_the_output_aspect() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn compile_error_leaves_the_running_scene() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn unallocatable_pass_buffer_fails_loudly() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();

        let (name, raw) = (
            CString::new("pass 'raw'").unwrap(),
            CString::new(RAW).unwrap(),
        );
        let idx = unsafe { ffi::ghee_mp_add_pass(r.handle(), name.as_ptr(), raw.as_ptr(), 1) };
        assert_eq!(idx, 0);
        assert!(!unsafe { ffi::ghee_mp_set_pass_size(r.handle(), idx, 1 << 17, 1 << 17) });
        assert!(r.last_error().contains("pass buffer failed"));
        unsafe { ffi::ghee_mp_abort(r.handle()) };
        assert_eq!(r.render(None).unwrap(), before);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn render_on_an_empty_scene_reports_instead_of_indexing() {
        let (_g, mut r) = gpu();
        r.reset();
        r.render(None).unwrap();
        assert!(r.last_error().contains("no passes"));
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_degenerate_out_size_cannot_undersize_the_readback_buffer() {
        // The engine downscales only for positive dimensions.
        // A zero or negative dimension falls through to the full-frame readback.
        // The buffer check sizes for that readback.
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn downscaled_readback_matches_the_full_frame() {
        let (_g, mut r) = gpu();
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

    /// A frame that differs at every pixel, so a blend or a shift shows.
    const SPECKLE: &str = "void main(){ vec2 p = floor(gl_FragCoord.xy); \
         gl_FragColor = vec4(fract(p.x * 0.137 + p.y * 0.311), fract(p.x * 0.071), \
         fract(p.y * 0.093), 1.0); }";

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_whole_multiple_readback_repeats_each_pixel() {
        let (_g, mut r) = gpu();
        assert_eq!(
            r.setup(&scene_of(json!([{"name": "main", "code": SPECKLE}])))
                .unwrap(),
            ""
        );
        r.set_time(Some(0.0));
        let full = r.render(None).unwrap().to_vec();
        for k in [2, 3] {
            let (w, h) = (W * k, H * k);
            let big = r.render_scaled(None, (w as i32, h as i32)).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let (at, from) = ((y * w + x) * COMPONENTS, ((y / k) * W + x / k) * COMPONENTS);
                    assert_eq!(
                        big[at..at + COMPONENTS],
                        full[from..from + COMPONENTS],
                        "x{k} at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_whole_multiple_readback_leaves_feedback_filtered() {
        // Frame 1 draws one-pixel stripes, and each later frame reads the last
        // one half a pixel over. A linear fetch there blends two stripes.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{
            "name": "main",
            "channels": [{"buffer": "main"}],
            "code": "void main(){ float s = mod(floor(gl_FragCoord.x), 2.0); \
                     vec4 last = texture2D(iChannel0, v_texCoord + vec2(0.5 / resolution.x, 0.0)); \
                     gl_FragColor = frame < 1.5 ? vec4(vec3(s), 1.0) : last; }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let stripes = r.render_scaled(None, (W as i32 * 2, H as i32 * 2)).unwrap();
        assert_ne!(px(&stripes, W * 2, 0, 0, 0), px(&stripes, W * 2, 2, 0, 0));
        let blended = r.render(None).unwrap().to_vec();
        for x in 1..W - 1 {
            let v = px(&blended, W, x, H / 2, 0);
            assert!((v - 127.5).abs() <= 2.0, "x {x}: {v}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn render_into_a_caller_buffer() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let full = r.render(None).unwrap().to_vec();
        let mut buf = vec![0u8; H * W * COMPONENTS];
        r.render_into(None, &mut buf, None).unwrap();
        assert_eq!(buf, full);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn draw_renders_with_no_readback() {
        // The scene feeds back: each frame adds to the last.
        // A draw() that renders shows up in the next readback.
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn one_upload_per_array_name() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_text_lays_out_its_curves_only_for_a_program_that_reads_them() {
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let layer = |fill: &str, fragment: &str| json!({"fill": fill, "layers": [{"text": "t", "font_size": 20, "fragment": fragment}]});
        let white = "void main(){ gl_FragColor = vec4(glyph_cover()); }";
        let covered = "void main(){ gl_FragColor = vec4(t_cover(glyph_em, 20.0)); }";
        let pass_reads = "void main(){ gl_FragColor = vec4(t_cover(v_texCoord, 20.0)); }";
        let black = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";
        let cache = crate::testing::TempDir::new("engine-curves");
        let (_g, mut r) = gpu();
        r.atlas_cache = Some(cache.join("atlas"));
        for (code, how, reads) in [
            (black, layer("mtsdf", white), false),
            (pass_reads, json!({}), true),
            (black, layer("mtsdf", covered), true),
            (black, layer("slug", white), true),
        ] {
            let mut pass = json!({"name": "main", "code": code});
            let mut text = json!({"value": "AV", "font": "DejaVuSans-ASCII.ttf"});
            if let Some(d) = how.get("layers") {
                pass["layers"] = d.clone();
                text["fill"] = how["fill"].clone();
            }
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "texts": {"t": text}, "passes": [pass]}),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(fonts).unwrap();
            assert_eq!(r.setup(&sc).unwrap(), "");
            let held = r.scene_assets.borrow().iter().any(|n| n == "ghee_t_tex");
            assert_eq!(held, reads, "{code} {how}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_setup_frees_the_slots_of_the_last_scenes_image_assets() {
        // The engine binds 16 assets at a time, so each setup hands back the
        // slots of the image assets that its scene no longer declares. An
        // asset set by the host stays.
        let (_g, mut r) = gpu();
        r.set_asset("host", 1, 1, 1, false, AssetData::U8(&[200u8]))
            .unwrap();
        for i in 0..20u8 {
            let name = format!("img{i}");
            let code = format!(
                "uniform sampler2D host; void main(){{ \
                 gl_FragColor = vec4(texture2D({name}, vec2(0.5)).r, \
                 texture2D(host, vec2(0.5)).r, 0.0, 1.0); }}"
            );
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "assets": {name.clone(): {"file": format!("{name}.png")}},
                        "passes": [{"name": "main", "code": code}]}),
                None,
                None,
            )
            .unwrap();
            sc.assets[0].image = Some(crate::assets::Image {
                width: 1,
                height: 1,
                components: 1,
                pixels: Pixels::U8(vec![i * 12]),
            });
            assert_eq!(r.setup(&sc).unwrap(), "", "{name}");
            r.set_time(Some(0.0));
            let f = r.render(None).unwrap();
            assert_eq!((f[0], f[1]), (i * 12, 200), "{name}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_nul_in_a_shader_drops_the_passes_staged_before_it() {
        // The engine stages 16 passes at most, so a stale stage of 15 would
        // refuse the second pass of the next scene.
        let (_g, mut r) = gpu();
        let mut passes: Vec<Value> = (0..15)
            .map(|i| json!({"name": format!("p{i}"), "code": FLAT}))
            .collect();
        passes.push(json!({"name": "main", "code": "void main(){}\u{0}"}));
        assert!(r.setup(&scene_of(Value::from(passes))).is_err());
        let two = scene_of(json!([{"name": "a", "code": FLAT},
            {"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}]));
        assert_eq!(r.setup(&two).unwrap(), "");
        r.set_time(Some(0.0));
        assert_eq!(&r.render(None).unwrap()[..4], [255; 4]);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn texture_unit_budget_rejects_a_greedy_pass() {
        let (_g, mut r) = gpu();
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
            over_units(&r, "pass 'main'", units + 1)
        );
        assert_eq!(r.render(None).unwrap(), before);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_reads_a_channel_or_an_array_in_every_texture_unit() {
        // main reads pass a through all but five of its units and five arrays
        // through the rest, and pass a declares twelve arrays of its own, so
        // the scene holds seventeen.
        let (_g, mut r) = gpu();
        let units = r.max_texture_units() as usize;
        let own: Vec<Value> = (0..12)
            .map(|i| json!({"name": format!("a{i}"), "size": 4}))
            .collect();
        let arrays: Vec<Value> = (0..5)
            .map(|i| json!({"name": format!("m{i}"), "size": 4}))
            .collect();
        let channels: Vec<Value> = (0..units - 5).map(|_| json!({"buffer": "a"})).collect();
        let sc = scene_of(json!([
            {"name": "a", "arrays": own,
             "code": "void main(){ gl_FragColor = vec4(a11_at(0), 0.0, 0.0, 1.0); }"},
            {"name": "main", "channels": channels, "arrays": arrays, "code": format!(
                "void main(){{ gl_FragColor = vec4(texture2D(iChannel{}, v_texCoord).r, \
                 m4_at(0), 0.0, 1.0); }}", units - 6)},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut feed = |name: &str, _size: ArraySize| match name {
            "a11" => Feed::Data(vec![0.25; 4]),
            "m4" => Feed::Data(vec![0.5; 4]),
            _ => Feed::Data(vec![0.0; 4]),
        };
        let f = r.render(Some(&mut feed)).unwrap();
        assert!(
            near3([f[0], f[1], f[2]].map(f64::from), [64.0, 128.0, 0.0]),
            "{:?}",
            &f[..4]
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_name_of_300_characters_reaches_its_uniform() {
        // A param, an array and an asset, each named past any fixed buffer.
        let (_g, mut r) = gpu();
        let (p, a, i) = ("p".repeat(300), "a".repeat(300), "i".repeat(300));
        let mut sc = Scene::from_dict(
            &json!({"version": 1,
                "params": {p.clone(): {"type": "float", "value": 0.25}},
                "assets": {i.clone(): {"file": "i.png"}},
                "passes": [{"name": "main", "arrays": [{"name": a.clone(), "size": 4}],
                    "code": format!("void main(){{ gl_FragColor = vec4({p}, {a}_at(0), \
                        texture2D({i}, vec2(0.5)).r, 1.0); }}")}]}),
            None,
            None,
        )
        .unwrap();
        sc.assets[0].image = Some(crate::assets::Image {
            width: 1,
            height: 1,
            components: 1,
            pixels: Pixels::U8(vec![191]),
        });
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let mut feed = |_: &str, _: ArraySize| Feed::Data(vec![0.5; 4]);
        let f = r.render(Some(&mut feed)).unwrap();
        assert!(
            near3([f[0], f[1], f[2]].map(f64::from), [64.0, 128.0, 191.0]),
            "{:?}",
            &f[..4]
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scene_past_16_passes_is_refused_counting_its_filters() {
        // The filters come from the registry, which $GHEE_FILTERS extends.
        let _f = crate::filters::env_guard();
        let (_g, mut r) = gpu();
        let passes = |n: usize| -> Vec<Value> {
            (0..n)
                .map(|i| json!({"name": format!("p{i}"), "code": FLAT}))
                .collect()
        };
        assert_eq!(r.setup(&scene_of(json!(passes(16)))).unwrap(), "");
        assert_eq!(
            r.setup(&scene_of(json!(passes(17)))).unwrap(),
            "17 passes, over the cap of 16"
        );
        let mut filtered = scene_of(json!(passes(15)));
        crate::filters::apply_filters(&mut filtered, &["dither".into(), "frc".into()], None)
            .unwrap();
        assert_eq!(
            r.setup(&filtered).unwrap(),
            "17 passes, 2 of them filters, over the cap of 16"
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_binds_16_texture_units_at_most_on_any_gpu() {
        let (_g, r) = gpu();
        let gl = unsafe { ffi::ghee_gl_texture_units(r.handle()) };
        assert_eq!(r.max_texture_units(), gl.min(16));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_assets_a_pass_samples_count_against_its_texture_units() {
        // The scene holds four assets past the units of one pass. main
        // samples an asset in every unit, pass a samples the other four, and
        // a channel on main takes one unit more.
        let (_g, mut r) = gpu();
        let units = r.max_texture_units() as usize;
        let names: Vec<String> = (0..units + 4).map(|i| format!("img{i}")).collect();
        let sum = |names: &[String]| -> String {
            names
                .iter()
                .map(|n| format!("texture2D({n}, vec2(0.5)).r"))
                .collect::<Vec<_>>()
                .join(" + ")
        };
        let scene = |channels: Value| {
            let assets: serde_json::Map<String, Value> = names
                .iter()
                .map(|n| (n.clone(), json!({"file": format!("{n}.png")})))
                .collect();
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "assets": assets, "passes": [
                    {"name": "a", "code": format!(
                        "void main(){{ gl_FragColor = vec4(0.0, {}, 0.0, 1.0); }}",
                        sum(&names[units..]))},
                    {"name": "main", "channels": channels, "code": format!(
                        "void main(){{ gl_FragColor = vec4({}, 0.0, 0.0, 1.0); }}",
                        sum(&names[..units]))},
                ]}),
                None,
                None,
            )
            .unwrap();
            for (i, a) in sc.assets.iter_mut().enumerate() {
                a.image = Some(crate::assets::Image {
                    width: 1,
                    height: 1,
                    components: 1,
                    pixels: Pixels::U8(vec![i as u8 + 1]),
                });
            }
            sc
        };
        // main's red sums the values 1 to units, and pass a's green the next
        // four, which a scene of pass a alone shows.
        let two = scene(json!([]));
        let mut only_a = two.clone();
        only_a.passes.truncate(1);
        for (sc, want) in [
            (&two, [(1..=units).sum::<usize>(), 0]),
            (&only_a, [0, (units + 1..=units + 4).sum::<usize>()]),
        ] {
            assert_eq!(r.setup(sc).unwrap(), "");
            r.set_time(Some(0.0));
            let f = r.render(None).unwrap();
            assert_eq!([f[0] as usize, f[1] as usize], want);
        }
        assert_eq!(
            r.setup(&scene(json!([{"buffer": "a"}]))).unwrap(),
            over_units(&r, "pass 'main'", units + 1)
        );
    }

    const HALF: &str = "void main(){ gl_FragColor = vec4(0.25, 0.5, 0.0, 0.5); }";

    fn first_frames(r: &mut Renderer, sc: &Scene, n: usize) -> Vec<u8> {
        assert_eq!(r.setup(sc).unwrap(), "");
        let mut f = Vec::new();
        for i in 0..n {
            r.set_time(Some(i as f64 / 60.0));
            f = r.render(None).unwrap().to_vec();
        }
        f
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_readback_reads_alpha_255_unless_the_alpha_is_kept() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "code": HALF}]));
        let opaque = first_frames(&mut r, &sc, 1);
        assert!(channel(&opaque, 3).iter().all(|a| *a == 255.0));
        let small = (W as i32 / 2, H as i32 / 2);
        let scaled = r.render_scaled(None, small).unwrap();
        assert!(channel(&scaled, 3).iter().all(|a| *a == 255.0));
        r.set_alpha(true);
        let kept = first_frames(&mut r, &sc, 1);
        assert!(channel(&kept, 3).iter().all(|a| (a - 127.5).abs() <= 1.0));
        let scaled = r.render_scaled(None, small).unwrap();
        assert!(channel(&scaled, 3).iter().all(|a| (a - 127.5).abs() <= 1.0));
        for c in 0..3 {
            assert_eq!(channel(&opaque, c), channel(&kept, c), "channel {c}");
        }
        // A 3-component frame reads the same colour with the alpha kept.
        drop(r);
        let dev = probe_dri_devices()
            .into_iter()
            .find(|d| Renderer::new(8, 8, 3, Some(d), false, false).is_ok())
            .unwrap();
        let mut rgb = Vec::new();
        for keep in [false, true] {
            let mut r3 = Renderer::new(8, 8, 3, Some(&dev), false, false).unwrap();
            r3.set_alpha(keep);
            assert_eq!(r3.setup(&sc).unwrap(), "");
            r3.set_time(Some(0.0));
            rgb.push(r3.render(None).unwrap().to_vec());
        }
        assert_eq!(rgb[0][..3], [64, 128, 0]);
        assert_eq!(rgb[0], rgb[1]);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_channel_reads_the_alpha_its_pass_wrote() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([
            {"name": "a", "code": "void main(){ gl_FragColor = vec4(1.0, 1.0, 1.0, 0.25); }"},
            {"name": "main", "channels": [{"buffer": "a"}], "code":
                "void main(){ gl_FragColor = vec4(texture2D(iChannel0, v_texCoord).a, 0.0, 0.0, 1.0); }"},
        ]));
        let f = first_frames(&mut r, &sc, 1);
        assert!(channel(&f, 0).iter().all(|v| (v - 63.75).abs() <= 1.0));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_output_pass_that_a_channel_reads_keeps_its_alpha_and_reads_back_opaque() {
        // main writes the alpha it held a frame ago into red, and 0.5 into
        // alpha.
        let (_g, mut r) = gpu();
        let sc = scene_of(
            json!([{"name": "main", "channels": [{"buffer": "main"}], "code":
            "void main(){ gl_FragColor = vec4(texture2D(iChannel0, v_texCoord).a, 0.0, 0.0, 0.5); }"}]),
        );
        let f = first_frames(&mut r, &sc, 2);
        assert!(channel(&f, 0).iter().all(|v| (v - 127.5).abs() <= 1.0));
        assert!(channel(&f, 3).iter().all(|a| *a == 255.0));
        r.set_alpha(true);
        let kept = first_frames(&mut r, &sc, 2);
        assert_eq!(channel(&kept, 0), channel(&f, 0));
        assert!(channel(&kept, 3).iter().all(|a| (a - 127.5).abs() <= 1.0));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_with_no_code_and_no_layers_is_refused_naming_it() {
        let (_g, mut r) = gpu();
        let mut sc = flat_scene();
        sc.passes[0].code = String::new();
        assert_eq!(
            r.setup(&sc).unwrap(),
            "pass 'main' has no 'code' (inline GLSL), no 'shader' (a file next to the scene) \
             and no 'layers'"
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn channel_source_pass_is_bound_checked() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        assert!(!unsafe { ffi::ghee_mp_set_channel(r.handle(), 0, 0, 7) });
        assert!(unsafe { ffi::ghee_mp_set_channel(r.handle(), 0, 0, -1) });
        assert!(unsafe { ffi::ghee_mp_set_channel(r.handle(), 0, 0, 0) });
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn set_time_none_returns_to_the_wall_clock() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn set_dt_reaches_every_pass() {
        // DT rides the draw like time, so one store lands in every pass.
        // Red carries the first pass's DT through a channel.
        // Green carries the second pass's DT.
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn closing_a_second_renderer_leaves_the_first_alive() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_short_2d_feed_is_zero_padded_and_a_long_1d_one_is_cut() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_unchanged_feed_leaves_the_texture_alone() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_error_in_the_common_reports_the_common_line() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_error_in_a_pass_after_a_common_block_names_that_pass_and_its_line() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn array_samples_correctly() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn sample_interpolates() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_u8_feed_renders_what_the_same_floats_render() {
        // The two wire forms name the same values: byte / 255. The frames must
        // match to the pixel, on a float data texture and on an 8-bit one.
        use sha2::{Digest, Sha256};
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_asset_binds_by_name_and_carries_its_size() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn sample_matches_texel_lerp() {
        // The test compares every column of a rendered _sample(t) ramp against
        // the lerp above. It holds on a GL that linear-filters float textures.
        // It also holds on V3D, which has no float linear filtering.
        // The renderer emits a reader that makes the two agree.
        let (_g, mut r) = gpu();
        let vals: Vec<f32> = vec![0.10, 0.90, 0.30, 0.70, 0.50];
        let sc = array_scene(
            "void main(){ gl_FragColor=vec4(vec3(vals_sample(v_texCoord.x)),1.0); }",
            "vals",
            json!(vals.len()),
        );
        for shader_lerp in [false, true] {
            r.shader_lerp = shader_lerp;
            assert_eq!(r.setup(&sc).unwrap(), "");
            let fed = vals.clone();
            let mut provider = |_: &str, _: ArraySize| Feed::Data(fed.clone());
            let f = r.render(Some(&mut provider)).unwrap().to_vec();
            for x in 0..W {
                let got = px(&f, W, x, H / 2, 0) / 255.0;
                let want = lerp1(&vals, (x as f64 + 0.5) / W as f64);
                assert!(
                    (got - want).abs() < 4.0 / 255.0,
                    "shader_lerp {shader_lerp}, column {x}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn sample2_matches_bilinear() {
        // _sample2 reads points mid-texel in one axis, in both, and in neither.
        let (_g, mut r) = gpu();
        let m: Vec<f32> = vec![0.0, 1.0, 0.5, 0.25]; // A 2x2 mask, indexed [y][x].
        let reads = [
            ((0.5, 0.5), 0.4375),             // The centre reads all four texels.
            ((0.25, 0.5), (0.0 + 0.5) / 2.0), // Column 0, between the rows.
            ((0.5, 0.25), (0.0 + 1.0) / 2.0), // Row 0, between the columns.
        ];
        for ((uv, want), shader_lerp) in reads.iter().flat_map(|c| [(*c, false), (*c, true)]) {
            r.shader_lerp = shader_lerp;
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
            assert!(
                (mean(&scaled) - want).abs() < 4.0 / 255.0,
                "{uv:?} shader_lerp {shader_lerp}"
            );
            let (lo, hi) = (
                got.iter().cloned().fold(f64::MAX, f64::min),
                got.iter().cloned().fold(f64::MIN, f64::max),
            );
            assert_eq!(lo, hi); // One value covers the whole frame.
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn every_shipped_scene_compiles_and_draws_under_dither_with_either_array_reader() {
        let _f = crate::filters::env_guard();
        let (_g, mut r) = gpu();
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/ghee/examples");
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".scene.json"))
            .collect();
        names.sort();
        assert!(!names.is_empty());
        for n in names {
            let mut sc = Scene::load(&format!("{dir}/{n}")).unwrap();
            crate::filters::apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
            for shader_lerp in [false, true] {
                r.shader_lerp = shader_lerp;
                assert_eq!(r.setup(&sc).unwrap(), "", "{n} shader_lerp {shader_lerp}");
                r.render(None).unwrap();
            }
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
        // So does a line past the common block's end.
        assert_eq!(user_line("0:4(1): x", glsl, code, common), (Some(1), false));
        assert_eq!(user_line("0:3(1): x", glsl, code, common), (Some(1), true));
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn setup_error_strings_match_the_reference() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_texture_budget_counts_an_array_name_once() {
        // A name declared twice on one pass takes one texture unit, so the
        // budget passes it. A scene load refuses the redeclaration, and so
        // does the compiler.
        let (_g, mut r) = gpu();
        let units = r.max_texture_units() as usize;
        let channels: Vec<Value> = (0..units - 1).map(|_| json!({"buffer": "a"})).collect();
        let mut sc = scene_of(json!([
            {"name": "a", "code": FLAT},
            {"name": "main", "channels": channels,
             "arrays": [{"name": "v", "size": 4}],
             "code": "void main(){ gl_FragColor = vec4(v_at(0)); }"},
        ]));
        let v = sc.passes[1].arrays[0].clone();
        sc.passes[1].arrays.push(v);
        assert_eq!(
            sc.name_clash().as_deref(),
            Some("array 'v' in pass 'main' is declared twice")
        );
        let err = r.setup(&sc).unwrap();
        assert!(!err.contains("texture units"), "{err}");
        if !err.is_empty() {
            assert_eq!(r.error_pass, Some(ErrorPass::Index(1)));
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_vec_or_color_param_sets_one_vector_uniform() {
        // This shader reads tint.r, off.x and off.y, so the whole of each
        // vector uniform is live.
        let (_g, mut r) = gpu();
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
        assert!(r.set_param("main", "tint", "color", &color).unwrap());
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
        // A value of another length fails, naming the param.
        assert_eq!(
            r.set_param("main", "off", "vec4", &off)
                .unwrap_err()
                .to_string(),
            "vec4 param 'off' needs 4 numbers"
        );
        let f = r.render(None).unwrap();
        for (c, want) in [(0, 128.0), (1, 64.0), (2, 191.0), (3, 255.0)] {
            assert!(
                (f64::from(f[c]) - want).abs() <= 2.0,
                "channel {c}: {}",
                f[c]
            );
        }
    }

    /// The colour in each quarter of a frame that splits at x = W/4, W/2 and
    /// 3W/4, read at the quarter's centre, 0..1.
    fn quarters(f: &[u8]) -> Vec<[f64; 4]> {
        (0..4)
            .map(|q| {
                let x = W * (2 * q + 1) / 8;
                [0, 1, 2, 3].map(|c| px(f, W, x, H / 2, c) / 255.0)
            })
            .collect()
    }

    fn near(got: &[[f64; 4]], want: &[[f64; 4]]) -> bool {
        got.iter()
            .flatten()
            .zip(want.iter().flatten())
            .all(|(g, w)| (g - w).abs() <= 1.5 / 255.0)
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn vec_and_color_values_reach_glsl_from_the_scene_and_a_set() {
        let (_g, mut r) = gpu();
        let code = "void main(){ float x = v_texCoord.x; \
                    gl_FragColor = x < 0.25 ? vec4(a, 0.0, 1.0) : x < 0.5 ? vec4(b, 1.0) \
                    : x < 0.75 ? vec4(c.xyz * c.w, 1.0) : vec4(tint, 1.0); }";
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "params": {"a": {"type": "vec2", "value": [0.25, 0.5]},
                           "tint": {"type": "color", "value": [0.5, 0.25, 1.0]}},
                "passes": [{"name": "main", "code": code, "params": {
                    "b": {"type": "vec3", "value": [0.75, 0.5, 0.25]},
                    "c": {"type": "vec4", "value": [1.0, 0.5, 0.25, 0.5]}}}],
            }),
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        let want = [
            [0.25, 0.5, 0.0, 1.0],
            [0.75, 0.5, 0.25, 1.0],
            [0.5, 0.25, 0.125, 1.0],
            [0.5, 0.25, 1.0, 1.0],
        ];
        assert!(near(&quarters(&f), &want), "{:?}", quarters(&f));
        let v = |x: &[f64]| ParamValue::Vec(x.to_vec());
        assert!(r.set_scene_param("a", "vec2", &v(&[1.0, 0.0])).unwrap());
        assert!(r
            .set_param("main", "b", "vec3", &v(&[0.0, 0.25, 1.0]))
            .unwrap());
        assert!(r
            .set_param("main", "c", "vec4", &v(&[0.5, 1.0, 0.0, 1.0]))
            .unwrap());
        let tint = ParamValue::Color([0.0, 1.0, 0.5]);
        assert!(r.set_scene_param("tint", "color", &tint).unwrap());
        let f = r.render(None).unwrap().to_vec();
        let want = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.25, 1.0, 1.0],
            [0.5, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.5, 1.0],
        ];
        assert!(near(&quarters(&f), &want), "{:?}", quarters(&f));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_local_shadows_a_vec_param_as_it_shadows_any_uniform() {
        let (_g, mut r) = gpu();
        let code = "vec3 lift(vec3 tint){ return tint + 0.25; }\n\
                    void main(){ vec3 tint = vec3(0.0, 0.25, 0.0); \
                    gl_FragColor = vec4(lift(tint), 1.0); }";
        let sc = scene_of(json!([{"name": "main", "code": code,
            "params": {"tint": {"type": "color", "value": [1.0, 1.0, 1.0]}}}]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        assert!(
            near(&quarters(&f)[..1], &[[0.25, 0.5, 0.25, 1.0]]),
            "{:?}",
            quarters(&f)
        );
    }

    #[test]
    fn a_pass_takes_mipmaps_at_a_size_that_the_gpu_mipmaps() {
        assert_eq!(mip_refusal("words", 1024, 512, false), None);
        assert_eq!(mip_refusal("words", 1, 1, false), None);
        assert_eq!(mip_refusal("words", 1280, 800, true), None);
        assert_eq!(
            mip_refusal("words", 1280, 800, false).as_deref(),
            Some(
                "pass 'words' takes mipmaps at 1280x800, and this GPU mipmaps a \
                 power-of-two size alone"
            )
        );
        assert!(mip_refusal("words", 1024, 800, false).is_some());
    }

    /// A 256x256 pass `check` of one-pixel black and white squares, times
    /// the param `level`, and the output pass, which reads it a quarter
    /// size at the texel centres through `main`'s code.
    fn checker_scene(mipmaps: bool, main: &str) -> Scene {
        scene_of(json!([
            {"name": "check", "size": [256, 256], "mipmaps": mipmaps,
             "params": {"level": {"type": "float", "value": 1.0}},
             "code": "void main(){ vec2 t = floor(v_texCoord * 256.0); \
                      gl_FragColor = vec4(vec3(mod(t.x + t.y, 2.0) * level), 1.0); }"},
            {"name": "main", "channels": [{"buffer": "check"}], "code": main},
        ]))
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_with_mipmaps_read_small_reads_its_average() {
        let (_g, mut r) = gpu_at(64, 64);
        let read = "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord + 0.5 / 256.0); }";
        let reds = |f: &[u8]| {
            (0..64 * 64)
                .map(|i| f64::from(f[i * COMPONENTS]))
                .collect::<Vec<_>>()
        };
        // Read at a texel centre with no mip chain, the checker gives black.
        assert_eq!(r.setup(&checker_scene(false, read)).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        assert!(reds(&f).iter().all(|v| *v == 0.0), "{:?}", &reds(&f)[..8]);
        // Four texels a pixel read from level 2, where each texel averages
        // 16 of the checker's.
        assert_eq!(r.setup(&checker_scene(true, read)).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        assert!(
            reds(&f).iter().all(|v| (v - 127.5).abs() <= 1.5),
            "{:?}",
            &reds(&f)[..8]
        );
        // The chain follows each frame.
        assert!(r
            .set_param("check", "level", "float", &ParamValue::Float(0.0))
            .unwrap());
        let f = r.render(None).unwrap().to_vec();
        assert!(reds(&f).iter().all(|v| *v == 0.0), "{:?}", &reds(&f)[..8]);
        // A batch that reads the pass through its channel samples the chain
        // too.
        let red = "void main(){ gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }";
        let fs = "uniform sampler2D iChannel0; uniform vec2 resolution;\n\
                  void main(){ gl_FragColor = texture2D(iChannel0, v_px / resolution + 0.5 / 256.0); }";
        let batches = [batch("main", fs, Blend::Replace)];
        assert_eq!(
            r.setup_batches(&checker_scene(true, red), &batches)
                .unwrap(),
            ""
        );
        r.set_batch_quads(0, &quad(0.0, 0.0, 64.0, 64.0, [0.0; 4]), 2)
            .unwrap();
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        assert!(
            reds(&f).iter().all(|v| (v - 127.5).abs() <= 1.5),
            "{:?}",
            &reds(&f)[..8]
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scene_param_reaches_every_pass_not_just_the_first() {
        // 'a' writes P to red. 'main' adds P to green through the channel.
        // A set that stops at the first pass leaves green at its default.
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn feed_lengths_that_miss_the_declared_size() {
        let (_g, mut r) = gpu();
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_readback_size_is_w_times_h_times_components() {
        // out_size reads back a GPU-scaled frame at that exact size.
        // The size scales up or down, and on either axis alone.
        let (_g, mut r) = gpu();
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

    // Batches

    /// The preamble's precision block. A uniform both stages declare must
    /// carry one precision in both.
    const PRECISION: &str = "#version 100\n#ifdef GL_FRAGMENT_PRECISION_HIGH\n\
        precision highp float;\n#else\nprecision mediump float;\n#endif\n";

    /// ghee_a0.xy is the vertex in pass pixels, y down like v_texCoord, and
    /// ghee_a1 the premultiplied colour.
    fn batch_vs(extra: &str) -> String {
        format!(
            "{PRECISION}attribute vec4 ghee_a0; attribute vec4 ghee_a1;\n\
             uniform vec2 resolution; uniform float ghee_vflip;\n\
             varying vec4 v_color; varying vec2 v_px;\n{extra}\n\
             void main() {{\n\
               vec2 ndc = ghee_a0.xy / resolution * 2.0 - 1.0;\n\
               gl_Position = vec4(ndc.x, ndc.y * ghee_vflip, 0.0, 1.0);\n\
               v_color = ghee_a1; v_px = ghee_a0.xy;\n\
             }}"
        )
    }

    fn batch_fs(body: &str) -> String {
        format!("{PRECISION}varying vec4 v_color; varying vec2 v_px;\n{body}")
    }

    const PLAIN_FS: &str = "void main(){ gl_FragColor = v_color; }";

    fn batch(pass: &str, fs_body: &str, blend: Blend) -> Batch {
        Batch {
            pass: pass.to_string(),
            vertex: batch_vs(""),
            fragment: batch_fs(fs_body),
            blend,
        }
    }

    /// One quad over the pixels [x0, x1) x [y0, y1), in triangle-strip order.
    fn quad(x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) -> Vec<f32> {
        [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            .iter()
            .flat_map(|&(x, y)| [x, y, 0.0, 0.0, color[0], color[1], color[2], color[3]])
            .collect()
    }

    fn rgb_at(f: &[u8], x: usize, y: usize) -> [f64; 3] {
        [0, 1, 2].map(|c| px(f, W, x, y, c))
    }

    fn near3(got: [f64; 3], want: [f64; 3]) -> bool {
        got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 2.0)
    }

    const GRAY: &str = "void main(){ gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0); }";

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn each_blend_mode_composes_its_formula_over_the_pass() {
        let (_g, mut r) = gpu();
        // src (0.25, 0, 0, 0.5) premultiplied, over a gray pass of 0.5.
        let src = [0.25, 0.0, 0.0, 0.5];
        for (blend, want) in [
            (Blend::Over, [128.0, 64.0, 64.0]),
            (Blend::Add, [191.0, 128.0, 128.0]),
            (Blend::Replace, [64.0, 0.0, 0.0]),
        ] {
            let sc = scene_of(json!([{"name": "main", "code": GRAY}]));
            let l = batch("main", PLAIN_FS, blend);
            assert_eq!(r.setup_batches(&sc, &[l]).unwrap(), "", "{blend:?}");
            r.set_batch_quads(0, &quad(0.0, 0.0, (W / 2) as f32, H as f32, src), 2)
                .unwrap();
            let f = r.render(None).unwrap().to_vec();
            for (x, y) in [(0, 0), (W / 2 - 1, H - 1), (W / 4, H / 2)] {
                let got = rgb_at(&f, x, y);
                assert!(near3(got, want), "{blend:?} at ({x}, {y}): {got:?}");
            }
            for (x, y) in [(W / 2, 0), (W - 1, H - 1)] {
                let got = rgb_at(&f, x, y);
                assert!(
                    near3(got, [128.0; 3]),
                    "{blend:?} outside at ({x}, {y}): {got:?}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_batch_lands_with_rows_counted_like_v_tex_coord() {
        // The pass paints the top rows, v_texCoord.y < 0.25, red. The batch
        // covers the same rows by pixel and adds green.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "code":
            "void main(){ gl_FragColor = vec4(v_texCoord.y < 0.25 ? 1.0 : 0.0, 0.0, 0.0, 1.0); }"}]));
        assert_eq!(
            r.setup_batches(&sc, &[batch("main", PLAIN_FS, Blend::Add)])
                .unwrap(),
            ""
        );
        let rows = (H / 4) as f32;
        r.set_batch_quads(0, &quad(0.0, 0.0, W as f32, rows, [0.0, 1.0, 0.0, 1.0]), 2)
            .unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(near3(rgb_at(&f, 5, 0), [255.0, 255.0, 0.0]));
        assert!(near3(rgb_at(&f, 5, H / 4 - 1), [255.0, 255.0, 0.0]));
        assert!(near3(rgb_at(&f, 5, H / 4), [0.0, 0.0, 0.0]));
        assert!(near3(rgb_at(&f, 5, H - 1), [0.0, 0.0, 0.0]));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn quads_past_one_16_bit_draw_land_where_their_vertices_say() {
        // The first draw holds 16383 quads. Quads 0..W*H paint each pixel
        // green, the next fill the first draw with nothing, and as many past
        // it paint each pixel red. Every pixel ends yellow only if the second
        // draw reads its own vertices.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "code":
            "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"}]));
        assert_eq!(
            r.setup_batches(&sc, &[batch("main", PLAIN_FS, Blend::Add)])
                .unwrap(),
            ""
        );
        let mut data = Vec::new();
        let pixels = |color: [f32; 4], data: &mut Vec<f32>| {
            for y in 0..H {
                for x in 0..W {
                    let (x, y) = (x as f32, y as f32);
                    data.extend(quad(x, y, x + 1.0, y + 1.0, color));
                }
            }
        };
        pixels([0.0, 1.0, 0.0, 1.0], &mut data);
        let empty = quad(0.0, 0.0, 0.0, 0.0, [1.0; 4]);
        for _ in W * H..16383 {
            data.extend(&empty);
        }
        pixels([1.0, 0.0, 0.0, 1.0], &mut data);
        assert_eq!(data.len() / 32, 16383 + W * H);
        r.set_batch_quads(0, &data, 2).unwrap();
        let f = r.render(None).unwrap().to_vec();
        for y in 0..H {
            for x in 0..W {
                let got = rgb_at(&f, x, y);
                assert!(near3(got, [255.0, 255.0, 0.0]), "({x}, {y}): {got:?}");
            }
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_draws_a_hundred_batches() {
        // Each batch adds 1/255 of red over black.
        let (_g, mut r) = gpu();
        let black = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";
        let sc = scene_of(json!([{"name": "main", "code": black}]));
        let batches: Vec<Batch> = (0..100)
            .map(|_| batch("main", PLAIN_FS, Blend::Add))
            .collect();
        assert_eq!(r.setup_batches(&sc, &batches).unwrap(), "");
        let red = quad(0.0, 0.0, W as f32, H as f32, [1.0 / 255.0, 0.0, 0.0, 0.0]);
        for k in 0..batches.len() {
            r.set_batch_quads(k, &red, 2).unwrap();
        }
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        assert!(
            near3(rgb_at(&f, W / 2, H / 2), [100.0, 0.0, 0.0]),
            "{:?}",
            rgb_at(&f, W / 2, H / 2)
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn batches_of_a_pass_draw_in_order_and_new_quads_replace_the_last() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "code": GRAY}]));
        let batches = [
            batch("main", PLAIN_FS, Blend::Replace),
            batch("main", PLAIN_FS, Blend::Replace),
        ];
        assert_eq!(r.setup_batches(&sc, &batches).unwrap(), "");
        let all = |c| quad(0.0, 0.0, W as f32, H as f32, c);
        r.set_batch_quads(0, &all([1.0, 0.0, 0.0, 1.0]), 2).unwrap();
        r.set_batch_quads(1, &all([0.0, 0.0, 1.0, 1.0]), 2).unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(near3(rgb_at(&f, 3, 3), [0.0, 0.0, 255.0]));
        // No quads draw nothing, and the first batch shows again.
        r.set_batch_quads(1, &[], 2).unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(near3(rgb_at(&f, 3, 3), [255.0, 0.0, 0.0]));
        r.set_batch_quads(0, &quad(0.0, 0.0, 1.0, 1.0, [0.0, 1.0, 0.0, 1.0]), 2)
            .unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(near3(rgb_at(&f, 0, 0), [0.0, 255.0, 0.0]));
        assert!(near3(rgb_at(&f, 3, 3), [128.0; 3]));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_batch_reads_scene_params_in_both_stages_and_its_pass_inputs() {
        // The vertex stage shifts the quad by `shift` pixels once `level` is
        // set, and the fragment stage writes `level`, an array texel, an asset
        // texel and the channel. `level` and `resolution` sit in both stages.
        let (_g, mut r) = gpu();
        r.set_asset("pad", 1, 1, 1, false, AssetData::U8(&[64u8]))
            .unwrap();
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "params": {
                    "level": {"type": "float", "value": 0.0},
                    "shift": {"type": "float", "value": 0.0},
                },
                "passes": [
                    {"name": "a", "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.75, 1.0); }"},
                    {"name": "main", "channels": [{"buffer": "a"}],
                     "arrays": [{"name": "vals", "size": 4}],
                     "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"},
                ],
            }),
            None,
            None,
        )
        .unwrap();
        let l = Batch {
            pass: "main".to_string(),
            vertex: batch_vs("uniform float shift; uniform float level;").replace(
                "v_px = ghee_a0.xy;",
                "v_px = ghee_a0.xy; gl_Position.x += step(0.25, level) * shift * 2.0 / resolution.x;",
            ),
            fragment: batch_fs(
                "uniform float level; uniform sampler2D vals_tex; uniform sampler2D pad;\n\
                 uniform sampler2D iChannel0; uniform vec2 pad_size; uniform vec2 resolution;\n\
                 void main(){ gl_FragColor = vec4(level, \
                   texture2D(vals_tex, vec2(0.125, 0.5)).r * 0.5 + texture2D(pad, vec2(0.5)).r, \
                   texture2D(iChannel0, v_px / resolution).b * pad_size.x, 1.0); }",
            ),
            blend: Blend::Replace,
        };
        assert_eq!(r.setup_batches(&sc, &[l]).unwrap(), "");
        r.set_time(Some(0.0));
        assert!(r
            .set_scene_param("level", "float", &ParamValue::Float(0.5))
            .unwrap());
        assert!(r
            .set_scene_param("shift", "float", &ParamValue::Float(8.0))
            .unwrap());
        r.set_batch_quads(0, &quad(0.0, 0.0, 8.0, H as f32, [0.0; 4]), 2)
            .unwrap();
        let f = {
            let mut provider = |_: &str, _: ArraySize| Feed::Data(vec![0.5; 4]);
            r.render(Some(&mut provider)).unwrap().to_vec()
        };
        // 0.5 * 0.5 + 64 / 255 is 0.501.
        let want = [128.0, 128.0, 191.0];
        assert!(near3(rgb_at(&f, 8, 4), want), "{:?}", rgb_at(&f, 8, 4));
        assert!(near3(rgb_at(&f, 15, 4), want), "{:?}", rgb_at(&f, 15, 4));
        assert!(near3(rgb_at(&f, 7, 4), [0.0; 3]), "{:?}", rgb_at(&f, 7, 4));
        assert!(
            near3(rgb_at(&f, 16, 4), [0.0; 3]),
            "{:?}",
            rgb_at(&f, 16, 4)
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_batch_reads_an_asset_its_pass_never_samples_and_one_set_later() {
        let (_g, mut r) = gpu();
        r.set_asset("early", 1, 1, 1, false, AssetData::U8(&[64u8]))
            .unwrap();
        let sc = scene_of(json!([{"name": "main", "code": GRAY}]));
        let l = batch(
            "main",
            "uniform sampler2D early; uniform sampler2D late;\n\
             void main(){ gl_FragColor = vec4(texture2D(early, vec2(0.5)).r, \
               texture2D(late, vec2(0.5)).r, 0.0, 1.0); }",
            Blend::Replace,
        );
        assert_eq!(r.setup_batches(&sc, &[l]).unwrap(), "");
        r.set_asset("late", 1, 1, 1, false, AssetData::U8(&[191u8]))
            .unwrap();
        r.set_batch_quads(0, &quad(0.0, 0.0, W as f32, H as f32, [0.0; 4]), 2)
            .unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(
            near3(rgb_at(&f, 1, 1), [64.0, 191.0, 0.0]),
            "{:?}",
            rgb_at(&f, 1, 1)
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_later_pass_samples_the_batch_drawn_into_an_earlier_one() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([
            {"name": "a", "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"},
            {"name": "main", "channels": [{"buffer": "a"}],
             "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }"},
        ]));
        assert_eq!(
            r.setup_batches(&sc, &[batch("a", PLAIN_FS, Blend::Over)])
                .unwrap(),
            ""
        );
        r.set_batch_quads(0, &quad(0.0, 0.0, (W / 2) as f32, H as f32, [1.0; 4]), 2)
            .unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(near3(rgb_at(&f, 2, 2), [255.0; 3]));
        assert!(near3(rgb_at(&f, W - 3, 2), [0.0; 3]));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_broken_batch_leaves_the_running_scene_and_names_its_stage() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "code": GRAY}]));
        assert_eq!(
            r.setup_batches(&sc, &[batch("main", PLAIN_FS, Blend::Replace)])
                .unwrap(),
            ""
        );
        r.set_batch_quads(
            0,
            &quad(0.0, 0.0, W as f32, H as f32, [1.0, 0.0, 0.0, 1.0]),
            2,
        )
        .unwrap();
        let before = r.render(None).unwrap().to_vec();
        let mut bad = batch("main", PLAIN_FS, Blend::Over);
        bad.vertex = bad
            .vertex
            .replace("v_color = ghee_a1;", "v_color = no_such_symbol;");
        let err = r.setup_batches(&flat_scene(), &[bad]).unwrap();
        assert!(err.starts_with("batch 0: vertex shader:"), "{err}");
        assert_eq!(r.error_pass, Some(ErrorPass::Index(0)));
        let bad = batch(
            "main",
            "void main(){ gl_FragColor = no_such_symbol; }",
            Blend::Over,
        );
        let err = r.setup_batches(&flat_scene(), &[bad]).unwrap();
        assert!(
            err.starts_with("batch 0: ") && !err.contains("vertex shader"),
            "{err}"
        );
        assert_eq!(r.render(None).unwrap(), &before[..]);
        // The old batch still takes quads.
        r.set_batch_quads(0, &[], 2).unwrap();
        assert!(near3(rgb_at(r.render(None).unwrap(), 1, 1), [128.0; 3]));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_batch_and_its_quads_are_checked_before_the_engine_sees_them() {
        let (_g, mut r) = gpu();
        let err = r
            .setup_batches(&flat_scene(), &[batch("typo", PLAIN_FS, Blend::Over)])
            .unwrap();
        assert_eq!(
            err,
            "batch 0 draws into pass 'typo', which is not a pass in this scene"
        );
        assert_eq!(
            r.setup_batches(&flat_scene(), &[batch("main", PLAIN_FS, Blend::Over)])
                .unwrap(),
            ""
        );
        let one = quad(0.0, 0.0, 1.0, 1.0, [1.0; 4]);
        assert_eq!(
            r.set_batch_quads(1, &one, 2).unwrap_err().to_string(),
            "no batch 1: setup_batches() took 1 batch"
        );
        assert_eq!(
            r.set_batch_quads(0, &one, 5).unwrap_err().to_string(),
            "batch 0: a vertex holds 1 to 4 attributes, got 5"
        );
        assert_eq!(
            r.set_batch_quads(0, &one[1..], 2).unwrap_err().to_string(),
            "batch 0: a quad takes 32 floats, and 31 is not a multiple of 32"
        );
        // setup() drops the batches with the scene.
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        assert!(r.set_batch_quads(0, &one, 2).is_err());
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_batch_with_four_attributes_reads_each_one() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "code": GRAY}]));
        let vs = format!(
            "{PRECISION}attribute vec4 ghee_a0; attribute vec4 ghee_a1; attribute vec4 ghee_a2; attribute vec4 ghee_a3;\n\
             uniform vec2 resolution; uniform float ghee_vflip; varying vec4 v_color;\n\
             void main() {{ vec2 ndc = ghee_a0.xy / resolution * 2.0 - 1.0;\n\
               gl_Position = vec4(ndc.x, ndc.y * ghee_vflip, 0.0, 1.0);\n\
               v_color = vec4(ghee_a1.x, ghee_a2.y, ghee_a3.z, 1.0); }}"
        );
        let l = Batch {
            pass: "main".to_string(),
            vertex: vs,
            fragment: format!("{PRECISION}varying vec4 v_color;\n{PLAIN_FS}"),
            blend: Blend::Replace,
        };
        assert_eq!(r.setup_batches(&sc, &[l]).unwrap(), "");
        let data: Vec<f32> = [
            (0.0, 0.0),
            (W as f32, 0.0),
            (0.0, H as f32),
            (W as f32, H as f32),
        ]
        .iter()
        .flat_map(|&(x, y)| {
            [
                x, y, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0,
            ]
        })
        .collect();
        r.set_batch_quads(0, &data, 4).unwrap();
        let f = r.render(None).unwrap().to_vec();
        assert!(
            near3(rgb_at(&f, 9, 9), [64.0, 128.0, 191.0]),
            "{:?}",
            rgb_at(&f, 9, 9)
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_layer_with_no_error_boxes_renders_as_the_layer_alone() {
        // The scene `with` holds a layer, and so an error-box batch. `alone`
        // holds the same layer as a plain batch, with none behind it.
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let place = "vec2 glyph_place(vec2 em) { return vec2(3.0, 2.0) + em * 30.0; }";
        let code = "void main(){ float c = glyph_cover(); \
                    gl_FragColor = vec4(c, c * 0.5, 0.0, c); }";
        let load = |layers: Value| {
            let mut sc = Scene::from_dict(
                &json!({
                    "version": 1,
                    "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
                    "texts": {"label": {"value": "AV", "font": "sans", "fill": "slug"}},
                    "passes": [{"name": "main", "layers": layers,
                        "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 1.0); }"}],
                }),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(fonts).unwrap();
            sc
        };
        let with = load(json!([{"text": "label", "fragment": code, "vertex": place}]));
        let alone = load(json!([]));
        let (_g, mut r) = gpu();
        let frame = |r: &mut Renderer| {
            r.set_time(Some(0.0));
            r.render(None).unwrap().to_vec()
        };
        assert_eq!(r.setup(&with).unwrap(), "");
        let built = with.texts[0].build("AV").unwrap().built.unwrap();
        assert_eq!((built.quads.count, built.boxes.count), (2, 0));
        let drawn = frame(&mut r);
        assert!(r.text_errors().is_empty());
        // The error-box batch is live: given the glyphs' quads, it covers
        // them.
        let boxes = r.layers[0].boxes;
        r.batch_quads(boxes, "boxes", &built.quads.floats, 4)
            .unwrap();
        assert!(frame(&mut r) != drawn, "the error-box batch drew nothing");
        let (vs, fs) = with.passes[0].layer_glsl(0, &with, r.data_texture_linear());
        let l = Batch {
            pass: "main".to_string(),
            vertex: vs,
            fragment: fs,
            blend: Blend::Over,
        };
        assert_eq!(r.setup_batches(&alone, &[l]).unwrap(), "");
        r.set_batch_quads(0, &built.quads.floats, 4).unwrap();
        assert!(
            frame(&mut r) == drawn,
            "the error-box batch changed the frame"
        );
    }

    /// A text read by `label_cover` across the frame, set `align`.
    fn cover_scene(words: &str, align: &str) -> Scene {
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let mut sc = Scene::from_dict(
            &json!({
                "version": 1,
                "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
                "texts": {"label": {"value": words, "font": "sans", "align": align}},
                "passes": [{"name": "main", "code": "void main(){ \
                    float c = label_cover(v_texCoord * label_size, resolution.y / label_size.y); \
                    gl_FragColor = vec4(c, c, c, 1.0); }"}],
            }),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(fonts).unwrap();
        sc
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_words_a_text_holds_lay_out_once() {
        let (_g, mut r) = gpu();
        let words = "the words a text holds\n".repeat(90);
        assert_eq!(r.setup(&cover_scene(&words, "left")).unwrap(), "");
        r.set_time(Some(0.0));
        let left = still(&mut r);
        assert!(r.set_text("label", "other words").unwrap());
        assert!(r.set_text("label", &words).unwrap());
        assert!(still(&mut r) == left);
        let laid = r.layouts.get();
        for _ in 0..100 {
            assert!(r.set_text("label", &words).unwrap());
        }
        assert_eq!(r.layouts.get(), laid);
        assert!(still(&mut r) == left);
        // A new setup lays the same words out for its own text.
        assert_eq!(r.setup(&cover_scene("I\nIIIIII", "left")).unwrap(), "");
        let left = still(&mut r);
        assert_eq!(r.setup(&cover_scene("I\nIIIIII", "right")).unwrap(), "");
        assert!(still(&mut r) != left);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_colour_atlas_exists_once_the_words_hold_a_colour_glyph() {
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let load = |font: Value| {
            let mut sc = Scene::from_dict(
                &json!({
                    "version": 1,
                    "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"},
                               "emoji": {"file": "NotoColorEmoji-Smiles.ttf"}},
                    "texts": {"label": {"value": "AV", "font": font}},
                    "passes": [{"name": "main",
                        "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                        "layers": [{"text": "label",
                            "fragment": "void main(){ gl_FragColor = vec4(glyph_cover()); }",
                            "vertex": "vec2 glyph_place(vec2 em) { return em * 20.0; }"}]}],
                }),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(fonts).unwrap();
            sc
        };
        let cache = crate::testing::TempDir::new("engine-colour");
        let (_g, mut r) = gpu();
        r.atlas_cache = Some(cache.join("atlas"));
        let emoji = |r: &Renderer| {
            let held = r
                .scene_assets
                .borrow()
                .iter()
                .any(|n| n == "ghee_label_emoji");
            (r.colours.borrow().len(), held)
        };
        // A text whose fonts hold no colour bitmaps takes no colour batch.
        assert_eq!(r.setup(&load(json!("sans"))).unwrap(), "");
        assert_eq!((r.layers[0].colour, r.batches.len()), (-1, 0));
        assert_eq!(emoji(&r), (0, false));
        // A colour font gives the layer its colour batch, and the atlas waits
        // for a colour glyph in the words.
        assert_eq!(r.setup(&load(json!(["sans", "emoji"]))).unwrap(), "");
        assert!(r.layers[0].colour >= 0);
        assert_eq!(emoji(&r), (0, false));
        assert!(r.set_text("label", "A\u{1f600}").unwrap());
        // A render with the time pinned waits for the held words.
        r.set_time(Some(0.0));
        r.render(None).unwrap();
        assert_eq!(emoji(&r), (1, true));
        assert!(r.text_errors().is_empty(), "{:?}", r.text_errors());
        // A new setup frees the atlas until its words need it again.
        assert_eq!(r.setup(&load(json!(["sans", "emoji"]))).unwrap(), "");
        assert_eq!(emoji(&r), (0, false));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_new_colour_glyph_uploads_its_cell_at_every_level_as_a_whole_atlas_does() {
        // Five layers of the words, at 1, 1/2, 1/4, 1/8 and 1/16 of the
        // bitmaps' 109 pixels per em, each read from its own mip level.
        let (w, h) = (400, 224);
        let layers: Vec<Value> = (0..5)
            .map(|k| {
                let s = 109.0 / f64::from(1 << k);
                let at = [[4.0, 4.0], [4.0, 140.0], [150.0, 140.0], [230.0, 140.0], [280.0, 140.0]][k];
                json!({"text": "label", "emoji": true,
                    "fragment": "void main(){ gl_FragColor = vec4(glyph_cover()); }",
                    "vertex": format!("vec2 glyph_place(vec2 em) {{ return vec2({:?}, {:?}) + em * {s:?}; }}",
                        at[0], at[1])})
            })
            .collect();
        let mut sc = Scene::from_dict(
            &json!({
                "version": 1,
                "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"},
                           "emoji": {"file": "NotoColorEmoji-Smiles.ttf"}},
                "texts": {"label": {"value": "A", "font": ["sans", "emoji"], "fill": "slug"}},
                "passes": [{"name": "main",
                    "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                    "layers": layers}],
            }),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts"))
            .unwrap();
        let (_g, mut r) = gpu_at(w as usize, h as usize);
        // A blank cell grows the atlas to 512 texels wide, room for both
        // emoji on one shelf, so the second emoji leaves the size as it was.
        let start = |r: &mut Renderer, words: &str| {
            assert_eq!(r.setup(&sc).unwrap(), "");
            let blank = atlas::Field {
                width: 512,
                height: 16,
                texels: vec![0; 512 * 16 * 4],
                ink: [0.0; 4],
                pad: 0.0,
            };
            let mut colours = r.colours.borrow_mut();
            let slot = colours.entry("label".to_string()).or_default();
            slot.atlas.insert(u32::MAX, &blank).unwrap();
            drop(colours);
            assert!(r.set_text("label", words).unwrap());
        };
        let size = |r: &Renderer| {
            let c = r.colours.borrow();
            (c["label"].atlas.width, c["label"].atlas.height)
        };
        start(&mut r, "\u{1f600}");
        let first = size(&r);
        assert!(r.set_text("label", "\u{1f600}\u{1f642}").unwrap());
        assert_eq!(size(&r), first);
        r.set_time(Some(0.0));
        let cell = r.render(None).unwrap().to_vec();
        start(&mut r, "\u{1f600}\u{1f642}");
        assert_eq!(size(&r), first);
        let whole = r.render(None).unwrap().to_vec();
        assert!(r.text_errors().is_empty(), "{:?}", r.text_errors());
        assert!(
            whole.chunks_exact(4).any(|p| p[0] != p[2]),
            "no emoji drawn"
        );
        assert!(cell == whole);
    }

    // The MTSDF fill's atlas

    /// The fixture font for an atlas, with two switches. Until `visible`, the
    /// glyphs of LATE draw nothing to the atlas, so its preload leaves them
    /// out, and until `open`, their outlines wait. A character from U+E000
    /// takes a glyph of its own, 'O' three times as large, drawn by no layout.
    struct Switched {
        inner: text::AtlasFont,
        visible: std::sync::atomic::AtomicBool,
        open: std::sync::atomic::AtomicBool,
    }

    const LATE: &str = "QZ";
    const BIG: u32 = 0xf000;

    impl Switched {
        fn new() -> Arc<Switched> {
            let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
            let bytes = fs::read(format!("{fonts}/DejaVuSans-ASCII.ttf")).unwrap();
            Arc::new(Switched {
                inner: text::AtlasFont(vec![Ok(Arc::new(bytes))]),
                visible: false.into(),
                open: true.into(),
            })
        }

        fn set(&self, flag: &std::sync::atomic::AtomicBool, on: bool) {
            flag.store(on, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl atlas::Font for Switched {
        fn key(&self) -> [u8; 32] {
            self.inner.key()
        }

        fn glyph(&self, ch: char) -> Option<u32> {
            let on =
                |f: &std::sync::atomic::AtomicBool| f.load(std::sync::atomic::Ordering::SeqCst);
            match u32::from(ch) {
                c @ 0xe000..=0xe0ff => Some(BIG + (c - 0xe000)),
                _ if LATE.contains(ch) && !on(&self.visible) => None,
                _ => self.inner.glyph(ch),
            }
        }

        fn outline(&self, gid: u32) -> Result<Vec<Vec<atlas::Curve>>> {
            if gid >= BIG {
                let o = self.inner.outline(self.inner.glyph('O').unwrap())?;
                return Ok(o
                    .into_iter()
                    .map(|c| {
                        c.into_iter()
                            .map(|q| q.map(|p| p.map(|v| v * 3.0)))
                            .collect()
                    })
                    .collect());
            }
            let late = LATE.chars().any(|c| self.inner.glyph(c) == Some(gid));
            while late && !self.open.load(std::sync::atomic::Ordering::SeqCst) {
                sleep(Duration::from_millis(1));
            }
            self.inner.outline(gid)
        }
    }

    /// A text drawn white over black by an MTSDF layer, at 40 px/em.
    fn mtsdf_scene(words: &str) -> Scene {
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let mut sc = Scene::from_dict(
            &json!({
                "version": 1,
                "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
                "texts": {"label": {"value": words, "font": "sans"}},
                "passes": [{"name": "main",
                    "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                    "layers": [{"text": "label",
                        "vertex": "vec2 glyph_place(vec2 em) { return vec2(2.0, 1.0) + em * 40.0; }",
                        "fragment": "void main(){ gl_FragColor = vec4(glyph_cover()); }"}]}],
            }),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(fonts).unwrap();
        sc
    }

    /// A renderer whose atlases read `font` and cache in a fresh directory,
    /// which the caller removes.
    fn switched(font: &Arc<Switched>, tag: &str) -> (MutexGuard<'static, ()>, Renderer, PathBuf) {
        let (g, mut r) = gpu();
        let f = font.clone();
        r.atlas_font = Box::new(move |_| f.clone());
        let dir = std::env::temp_dir().join(format!("ghee-engine-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        r.atlas_cache = Some(dir.clone());
        (g, r, dir)
    }

    fn still(r: &mut Renderer) -> Vec<u8> {
        r.render(None).unwrap().to_vec()
    }

    /// The largest difference between two frames.
    fn worst(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
    }

    /// The frame of `words` in a scene of their own, with every glyph in the
    /// atlas from the start.
    fn reference(words: &str) -> Vec<u8> {
        let (_g, mut r) = gpu();
        let dir = std::env::temp_dir().join(format!("ghee-engine-{}-ref", std::process::id()));
        r.atlas_cache = Some(dir.clone());
        assert_eq!(r.setup(&mtsdf_scene(words)).unwrap(), "");
        r.set_time(Some(0.0));
        let f = still(&mut r);
        drop(r);
        let _ = fs::remove_dir_all(&dir);
        f
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn new_words_hold_the_last_until_their_glyphs_are_in() {
        let (want_av, want_avz) = (reference("AV"), reference("AVZ"));
        let font = Switched::new();
        let (_g, mut r, dir) = switched(&font, "hold");
        assert_eq!(r.setup(&mtsdf_scene("AV")).unwrap(), "");
        let size = |r: &Renderer| {
            r.atlases.borrow()[0]
                .gen
                .with(|a| (a.width, a.height))
                .unwrap()
        };
        let before = size(&r);
        let av = still(&mut r);
        assert!(worst(&av, &want_av) <= 1);
        font.set(&font.visible, true);
        font.set(&font.open, false);
        assert!(r.set_text("label", "AVZ").unwrap());
        for _ in 0..5 {
            sleep(Duration::from_millis(20));
            assert!(still(&mut r) == av, "the words switched before Z was in");
        }
        font.set(&font.open, true);
        let t0 = std::time::Instant::now();
        let shown = loop {
            let f = still(&mut r);
            if f != av {
                break f;
            }
            assert!(t0.elapsed() < Duration::from_secs(20), "Z never came in");
            sleep(Duration::from_millis(1));
        };
        assert!(
            worst(&shown, &want_avz) <= 1,
            "{}",
            worst(&shown, &want_avz)
        );
        // Z took a cell in the atlas as it stood: rows, no new texture.
        assert_eq!(size(&r), before);
        drop(r);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn with_the_time_pinned_a_render_waits_for_the_glyphs() {
        let want = reference("AVQ");
        let font = Switched::new();
        let (_g, mut r, dir) = switched(&font, "pinned");
        assert_eq!(r.setup(&mtsdf_scene("AV")).unwrap(), "");
        r.set_time(Some(0.0));
        let av = still(&mut r);
        font.set(&font.visible, true);
        font.set(&font.open, false);
        assert!(r.set_text("label", "AVQ").unwrap());
        let opener = {
            let font = font.clone();
            thread::spawn(move || {
                sleep(Duration::from_millis(300));
                font.set(&font.open, true);
            })
        };
        let t0 = std::time::Instant::now();
        let f = still(&mut r);
        assert!(
            t0.elapsed() >= Duration::from_millis(250),
            "{:?}",
            t0.elapsed()
        );
        opener.join().unwrap();
        assert!(f != av && worst(&f, &want) <= 1, "{}", worst(&f, &want));
        drop(r);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_atlas_that_grows_uploads_again_and_draws_the_same() {
        let want = reference("AVQ");
        let font = Switched::new();
        let (_g, mut r, dir) = switched(&font, "grow");
        assert_eq!(r.setup(&mtsdf_scene("AV")).unwrap(), "");
        r.set_time(Some(0.0));
        let av = still(&mut r);
        let size = |r: &Renderer| {
            r.atlases.borrow()[0]
                .gen
                .with(|a| (a.width, a.height))
                .unwrap()
        };
        let before = size(&r);
        let big: String = (0..40)
            .map(|i| char::from_u32(0xe000 + i).unwrap())
            .collect();
        let t0 = std::time::Instant::now();
        while !r.atlases.borrow()[0].gen.request(big.chars()).is_empty() {
            assert!(
                t0.elapsed() < Duration::from_secs(60),
                "the big glyphs never came in"
            );
            sleep(Duration::from_millis(1));
        }
        let after = size(&r);
        assert!(
            after.0 * after.1 > before.0 * before.1,
            "{before:?} to {after:?}"
        );
        assert!(still(&mut r) == av, "the grown atlas moved the glyphs");
        // New words after the growth take their glyphs from the new texture.
        font.set(&font.visible, true);
        assert!(r.set_text("label", "AVQ").unwrap());
        let f = still(&mut r);
        assert!(worst(&f, &want) <= 1, "{}", worst(&f, &want));
        drop(r);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scene_of_slug_layers_starts_no_atlas() {
        let (_g, mut r) = gpu();
        let mut sc = mtsdf_scene("AV");
        sc.texts[0].fill = Fill::Slug;
        assert_eq!(r.setup(&sc).unwrap(), "");
        assert!(r.atlases.borrow().is_empty());
        // A text that no layer names takes no atlas either.
        let mut sc = mtsdf_scene("AV");
        sc.passes[0].layers.clear();
        assert_eq!(r.setup(&sc).unwrap(), "");
        assert!(r.atlases.borrow().is_empty());
    }
}
