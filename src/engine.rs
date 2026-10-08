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
//! The Renderer draws the scene's layers too. An image layer draws one quad
//! that samples the image bound to it. [`Renderer::set_text`]
//! lays a layer's words out through `crate::text`, uploads the texture of
//! curves that `_cover` and the slug render read, or the glyphs new to the
//! one that the slug layers share, and hands the layer one quad a glyph. An
//! MTSDF layer reads its glyphs from an atlas that a `crate::atlas`
//! generator fills on its own threads, and every render uploads the glyphs
//! that the atlas gained since the last. Consecutive shapes and slug texts
//! with no effect draw as runs, one batch each, as `crate::run` groups
//! them, and a change to a run's layers groups their pass again.

use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::assets::Pixels;
use crate::atlas;
use crate::cparam::create_c_parameter;
use crate::draw::{self, Call};
use crate::ffi;
use crate::layer::{
    is_image_file, Address, Detail, ImageLayer, Kind, Layer, Render, TextLayer, GLSL_NAME,
};
use crate::run::{Candidate, Class, Mode};
use crate::scene::{ArraySize, Look, ParamValue, Pass, Scene};
use crate::shape::ShapeLayer;
use crate::text::{self, Fault, Layout};

/// The passes that the engine holds, filter passes included. engine.c holds
/// the same number as MP_MAX_PASSES.
const MAX_PASSES: usize = 16;

/// The vec4 attributes of a vertex of a batch at most. engine.c holds the
/// same number as MP_BATCH_ATTRIBS.
const BATCH_ATTRIBUTES: usize = 8;

/// The engine's time step until set_dt() sets one.
const DT: f64 = 1.0 / 60.0;

/// The engine asset of the texture of curves that the slug layers share.
const GLYPHS: &str = "ghee_glyphs";

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
/// premultiplied by its alpha. Multiply and Screen hold their formulas over
/// an opaque buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// src + dst * (1 - src.a)
    Normal,
    /// src + dst
    Add,
    /// src * dst + dst * (1 - src.a)
    Multiply,
    /// src + dst * (1 - src)
    Screen,
}

impl Blend {
    /// Each blend with its name in a scene.
    pub const NAMES: [(Blend, &'static str); 4] = [
        (Blend::Normal, "normal"),
        (Blend::Add, "add"),
        (Blend::Multiply, "multiply"),
        (Blend::Screen, "screen"),
    ];

    /// The blend's name in a scene.
    pub fn name(self) -> &'static str {
        Blend::NAMES
            .iter()
            .find(|(b, _)| *b == self)
            .map_or("normal", |(_, n)| n)
    }

    fn engine(self) -> c_int {
        match self {
            Blend::Normal => ffi::GHEE_BLEND_NORMAL,
            Blend::Add => ffi::GHEE_BLEND_ADD,
            Blend::Multiply => ffi::GHEE_BLEND_MULTIPLY,
            Blend::Screen => ffi::GHEE_BLEND_SCREEN,
        }
    }
}

/// Quads that a test draws into a pass after the pass's layers, with a
/// vertex and a fragment shader of their own and a blend, which
/// setup_batches() hands the engine through ghee_mp_add_batch.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Batch {
    /// The name of the pass that the batch draws into.
    pub pass: String,
    pub vertex: String,
    pub fragment: String,
    pub blend: Blend,
}

/// A layer that the engine draws: the batch of a text layer's glyphs or of an
/// image layer's quad, and the batches of a text layer's colour glyphs and
/// of its error boxes.
struct LayerBatches {
    /// The engine's index of the batch.
    batch: c_int,
    /// The engine's index of the colour batch, -1 for an image layer. The
    /// batch has no program until the layer's words first hold a colour
    /// glyph.
    colour: c_int,
    /// True once the colour batch has its program.
    coloured: std::cell::Cell<bool>,
    /// The engine's index of the error-box batch, -1 for an image layer.
    boxes: c_int,
    /// The layer's name.
    name: String,
    /// The phase of each of the layer's effects that moves by one, as the
    /// frames advanced it.
    phases: std::cell::RefCell<Vec<(Kind, f64)>>,
    /// True while the batch reads the curves of its glyphs from GLYPHS.
    shared: std::cell::Cell<bool>,
    /// The uniforms and the size of the words that show() last set on the
    /// batches, None before the first.
    shown: std::cell::RefCell<Option<Shown>>,
}

/// A layout's float uniforms and its block's size, as show() sets them.
type Shown = (Vec<(&'static str, f64)>, [f64; 2]);

/// Where each layer sits, by its name: its pass and its index among the
/// pass's layers, its index among the drawn layers, and for a shape drawn in
/// a run, the run and its index there.
#[derive(Debug, Default, PartialEq)]
struct Places {
    scene: HashMap<String, (usize, usize)>,
    drawn: HashMap<String, usize>,
    runs: HashMap<String, (usize, usize)>,
}

/// What update_calls() did with each call, by its index.
#[derive(Debug, Default, PartialEq)]
pub struct Taken {
    /// The calls left for update_layer().
    pub left: Vec<usize>,
    /// The calls that gave their layer a new look.
    pub changed: Vec<usize>,
}

/// Consecutive layers of a pass that one batch draws, as `crate::run`
/// groups them.
struct Run {
    /// The engine's index of the batch.
    batch: c_int,
    /// The name of the pass and its size in pixels.
    pass: String,
    size: [f64; 2],
    /// The layers, in their order.
    members: Vec<Member>,
    /// The mode whose program the batch holds, None for none yet.
    installed: std::cell::Cell<Option<Mode>>,
    /// The texture of curves that the batch reads, as bound last.
    curves: std::cell::Cell<Bound>,
    /// The quads of each member that the batch holds, empty when it must
    /// take every quad again.
    sent: std::cell::RefCell<Vec<usize>>,
}

impl Run {
    /// The place in `members` of layer `name`.
    fn find(&self, name: &str) -> Option<usize> {
        self.members.iter().position(|m| m.name == name)
    }

    /// The program that the members ask for.
    fn mode(&self) -> Mode {
        let texts = self.members.iter().filter(|m| m.text.is_some()).count();
        Mode::of(self.members.len(), texts)
    }
}

/// The texture of curves bound to a run's batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    None,
    /// GLYPHS at this many rows.
    Glyphs(u32),
    /// The texture of the words of the run's one text.
    Own,
}

/// A layer of a run, and its quads: a shape's, shape::ATTRIBUTES vec4s a
/// vertex, or the glyph quads of a text's last words, text::QUAD_FLOATS
/// floats a vertex.
struct Member {
    name: String,
    quads: std::cell::RefCell<Vec<f32>>,
    /// A text's own, None for a shape.
    text: Option<TextMember>,
    /// The class that the member's last quads and settings give it.
    class: std::cell::Cell<Class>,
    /// The class that the last grouping of its pass took.
    placed: Class,
    /// True when the batch must take the member's vertices again.
    dirty: std::cell::Cell<bool>,
    /// True when the member's class must be worked out again.
    reclass: std::cell::Cell<bool>,
}

/// What a text of a run holds past its quads.
struct TextMember {
    /// Its block's size in em.
    size: std::cell::Cell<[f64; 2]>,
    /// True while its glyphs read GLYPHS.
    shared: std::cell::Cell<bool>,
    /// True once its words set a texture of their own.
    texture: std::cell::Cell<bool>,
    /// True when its words set that texture since a run of the text alone
    /// bound it.
    fresh: std::cell::Cell<bool>,
    /// The text's own program, which the renderer holds while the text
    /// draws, so a run of the text alone compiles nothing.
    own: Hold,
}

impl Member {
    /// Shape layer `l` of a pass of `size` pixels.
    fn shape(l: &ShapeLayer, size: [f64; 2]) -> Member {
        let mut quads = Vec::new();
        l.quads(size, &mut quads);
        let class = shape_class(l, size);
        Member {
            name: l.name.clone(),
            quads: std::cell::RefCell::new(quads),
            text: None,
            class: std::cell::Cell::new(class),
            placed: class,
            dirty: std::cell::Cell::new(true),
            reclass: std::cell::Cell::new(false),
        }
    }

    /// Text layer `l`, with no words yet, of class `class`, whose own
    /// program the renderer holds as `own`.
    fn text(l: &TextLayer, class: Class, own: Hold) -> Member {
        Member {
            name: l.name.clone(),
            quads: std::cell::RefCell::new(Vec::new()),
            text: Some(TextMember {
                size: std::cell::Cell::new([0.0; 2]),
                shared: std::cell::Cell::new(true),
                texture: std::cell::Cell::new(false),
                fresh: std::cell::Cell::new(false),
                own,
            }),
            class: std::cell::Cell::new(class),
            placed: class,
            dirty: std::cell::Cell::new(true),
            reclass: std::cell::Cell::new(false),
        }
    }

    /// The quads that the member holds.
    fn count(&self) -> usize {
        self.quads.borrow().len() / QUAD
    }
}

/// The floats of a quad of a shape or of a glyph, 4 vertices of 4 vec4s.
const QUAD: usize = 4 * text::QUAD_FLOATS;
const _: () = assert!(4 * crate::shape::ATTRIBUTES == text::QUAD_FLOATS);

/// The class of shape layer `l` in a pass of `size` pixels.
fn shape_class(l: &ShapeLayer, size: [f64; 2]) -> Class {
    match l.area(size) > crate::run::SHAPE_PIXELS {
        true => Class::Large,
        false => Class::Joins,
    }
}

/// The class of text layer `l` in a pass of `pass` pixels, with the glyph
/// quads `quads` of a block of `size` em, which read GLYPHS when `shared`.
fn text_class(l: &TextLayer, quads: &[f32], size: [f64; 2], shared: bool, pass: [f64; 2]) -> Class {
    let fs = l.font_px(pass, size);
    match !shared || crate::run::glyph_pixels(quads, fs) > crate::run::GLYPH_PIXELS {
        true => Class::Alone,
        false => Class::Joins,
    }
}

/// True for a layer that a run takes: a shape, or a slug text with no
/// effect.
fn joins_runs(l: &Layer) -> bool {
    match l {
        Layer::Shape(_) => true,
        Layer::Text(t) => t.render == Render::Slug && t.effects.is_empty(),
        Layer::Image(_) => false,
    }
}

/// A batch program that the renderer holds through ghee_mp_keep_program,
/// and lets go of when dropped.
struct Hold {
    handle: *mut ffi::GheeScene,
    vertex: CString,
    fragment: CString,
}

impl Hold {
    /// Hold `program` in the engine of `handle`, which compiles it unless a
    /// batch holds it. `what` names it in an error.
    fn keep(
        handle: *mut ffi::GheeScene,
        what: &str,
        program: &crate::layer::Program,
    ) -> Result<Hold> {
        let vertex = CString::new(program.0.as_str())?;
        let fragment = CString::new(program.1.as_str())?;
        if !unsafe { ffi::ghee_mp_keep_program(handle, vertex.as_ptr(), fragment.as_ptr()) } {
            let err = unsafe { ffi::ghee_last_error(handle) };
            let err = match err.is_null() {
                true => String::new(),
                false => unsafe { CStr::from_ptr(err) }
                    .to_string_lossy()
                    .into_owned(),
            };
            bail!("{what}: {}", or_else(err, "compile error"));
        }
        Ok(Hold {
            handle,
            vertex,
            fragment,
        })
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        unsafe {
            ffi::ghee_mp_let_go_program(self.handle, self.vertex.as_ptr(), self.fragment.as_ptr())
        };
    }
}

/// One atlas generator, and the engine textures that hold its atlas.
struct AtlasSlot {
    fonts: Vec<text::Font>,
    px_em: u32,
    range: u32,
    gen: atlas::Generator,
    /// The engine asset `ghee_<name>_atlas` of each layer drawn from this
    /// atlas, and whether it still waits for its first upload, which takes all
    /// of it.
    textures: Vec<(String, bool)>,
}

/// New words of a layer that wait for the atlas to hold their glyphs.
struct Held {
    words: String,
    laid: Layout,
}

/// The font that an atlas generator reads, from a layer's fonts.
type AtlasFont = Box<dyn Fn(&[text::Font]) -> Arc<dyn atlas::Font>>;

/// The colour atlas of one layer, the engine asset `ghee_<name>_emoji`.
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

/// The problems of one text layer, a line each.
#[derive(Clone, Default, PartialEq)]
struct TextErrors {
    /// The problems of the words on screen.
    drawn: Vec<String>,
    /// Why the last words failed to lay out, when they did.
    refused: Option<String>,
}

/// The quad of an image layer, one vec4 a vertex: its corner of the picture,
/// x right and y down from the top left.
const IMAGE_QUAD: [f32; 16] = [
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
];

/// A value for a live field of a layer, as a caller holds it.
#[derive(Clone, Copy, Debug)]
pub enum Given<'a> {
    /// Written as the scene format writes the key.
    Json(&'a Value),
    /// The numbers of a wire message.
    Numbers(&'a [f32]),
    /// The text of a wire message.
    Text(&'a str),
}

/// Why the scene took no live value: an unknown layer, effect or field, a
/// field that changes with a rebuild, or a value that the field does not
/// take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// True when `new`, a look of image layer `old`, draws another image.
fn other_image(old: &Layer, new: &Layer) -> bool {
    matches!((old, new), (Layer::Image(o), Layer::Image(n)) if o.image != n.image)
}

/// A fault of the caller's in a layer or a value, as a Refused error.
fn refused(e: anyhow::Error) -> anyhow::Error {
    anyhow::Error::new(Refused(format!("{e:#}")))
}

/// The image layers of `drawn` that draw a file of their own.
fn drawn_files(drawn: &[Pass]) -> impl Iterator<Item = &ImageLayer> {
    drawn
        .iter()
        .flat_map(|p| &p.layers)
        .filter_map(Layer::as_image)
        .filter(|l| l.is_file())
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

/// The turns that a renderer takes, in degrees clockwise.
pub const ROTATIONS: [i64; 4] = [0, 90, 180, 270];

/// The quarter turns clockwise of a rotation of `degrees`.
pub fn quarter_turns(degrees: i64) -> Result<c_int> {
    match ROTATIONS.iter().position(|d| *d == degrees) {
        Some(q) => Ok(q as c_int),
        None => bail!("rotate takes 0, 90, 180 or 270, got {degrees}"),
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

/// The line a GL compile error names. Mesa writes `0:18(5): error`, and the
/// line is the first `:<digits>(` in the error text. NVIDIA writes
/// `0(18) : error C1503`, and the line is the first `<digits>(<digits>)`.
fn error_line_number(err: &str) -> Option<i64> {
    let b = err.as_bytes();
    let digits_end = |i: usize| i + b[i..].iter().take_while(|c| c.is_ascii_digit()).count();
    let line = |from: usize, to: usize| Some(err[from..to].parse::<i64>().unwrap_or(i64::MAX));
    for (i, _) in err.match_indices(':') {
        let j = digits_end(i + 1);
        if j > i + 1 && b.get(j) == Some(&b'(') {
            return line(i + 1, j);
        }
    }
    for (i, _) in err.match_indices('(') {
        let j = digits_end(i + 1);
        if i > 0 && b[i - 1].is_ascii_digit() && j > i + 1 && b.get(j) == Some(&b')') {
            return line(i + 1, j);
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
    /// The scene of the last setup, with the live values set since.
    scene: std::cell::RefCell<Option<Scene>>,
    /// The last setup() error text. "" means the setup succeeded.
    pub error: String,
    /// The last setup() error is the GL's log of a GLSL compile or link.
    pub compile_error: bool,
    /// The user-code line of the last setup() compile error.
    pub error_line: Option<i64>,
    /// The pass holding the error. Common is the scene's common block.
    pub error_pass: Option<ErrorPass>,
    #[allow(dead_code)]
    pub display: bool,
    /// The card node that the display mode scans out on.
    card: Option<String>,
    /// The display's backlight before the first set_brightness(), which
    /// close() puts back.
    backlight: std::cell::RefCell<Option<crate::panel::Earlier>>,
    /// The readbacks keep the output pass's alpha. set_alpha() sets it.
    alpha: bool,
    /// The quarter turns clockwise that set_rotate() set.
    turns: c_int,
    /// pass name -> engine pass index, in the order setup() adds the passes.
    mp: Vec<(String, c_int)>,
    /// array name -> size, in declaration order.
    feeds: Vec<(String, ArraySize)>,
    /// The engine assets uploaded for the scene: its image assets, the data
    /// texture `ghee_<name>_curves` of each text layer, and the picture
    /// `ghee_<name>_file` of each image layer that draws a file. A setup
    /// frees those that its new scene does not declare, so their asset slots
    /// come back.
    scene_assets: std::cell::RefCell<Vec<String>>,
    /// The engine index of each batch passed to setup_batches(), in order.
    #[cfg(test)]
    batches: Vec<c_int>,
    /// The batches that setup_batches() hands to setup().
    #[cfg(test)]
    test_batches: Vec<Batch>,
    /// The layers that the engine draws, in the order of their passes.
    layers: Vec<LayerBatches>,
    /// The runs of the layers that the engine draws.
    runs: Vec<Run>,
    /// True when a run must take a program, a binding or quads before the
    /// next frame draws.
    unsettled: std::cell::Cell<bool>,
    /// True when a member of a run may have changed its class.
    reclassed: std::cell::Cell<bool>,
    /// The text problem causes logged since the setup, so each logs once.
    reported: std::cell::RefCell<HashSet<String>>,
    /// The problems of each text layer, by its name.
    text_errors: std::cell::RefCell<HashMap<String, TextErrors>>,
    /// Counts the changes to text_errors.
    text_errors_seq: std::cell::Cell<u64>,
    /// One atlas per font, px/em and range among the MTSDF layers drawn.
    atlases: std::cell::RefCell<Vec<AtlasSlot>>,
    /// The new words of each MTSDF layer still waiting for their glyphs, by
    /// the layer's name. The layer draws its last words meanwhile.
    held: std::cell::RefCell<HashMap<String, Held>>,
    /// The words that set_text last took for each layer, shown or held, by
    /// the layer's name. The same words again lay out nothing.
    words: std::cell::RefCell<HashMap<String, String>>,
    /// The engine asset of each image file that an image layer of the
    /// running scene draws or drew, by the file as the scene names it. A
    /// file loads once, and stays until the next setup.
    files: std::cell::RefCell<Vec<(String, String)>>,
    /// The files that the renderer loaded since it was created.
    file_count: std::cell::Cell<usize>,
    /// True once the engine holds the shape program for the renderer's life.
    shapes_kept: std::cell::Cell<bool>,
    /// True once the engine holds crate::run::program for the renderer's
    /// life.
    mixed_kept: std::cell::Cell<bool>,
    /// The layers whose texture of curves a program of the scene reads: a
    /// pass that calls `<name>_cover`, or a slug layer. Any other layer lays
    /// out with no such texture.
    curves: HashSet<String>,
    /// The layers whose `<name>_cover` a pass reads, which lay out into a
    /// texture of their own, with its grid. A slug layer that no pass reads
    /// lays out into GLYPHS.
    covered: HashSet<String>,
    /// The outlines that the layouts read, and the texture of GLYPHS.
    glyphs: std::cell::RefCell<text::Glyphs>,
    /// The uniforms and the size of each text layer's words that show() last
    /// set on the passes, by the layer's name.
    pass_shown: std::cell::RefCell<HashMap<String, Shown>>,
    /// Where each layer sits, None after a change of places until the next
    /// lookup makes it again.
    places: std::cell::RefCell<Option<Places>>,
    /// The colour atlas of each drawn layer whose words held colour glyphs,
    /// by the layer's name.
    colours: std::cell::RefCell<HashMap<String, ColourAtlas>>,
    /// The time that set_time() pinned. A render then waits for held words.
    pinned: std::cell::Cell<Option<f64>>,
    /// The pinned time of the last render since the setup, 0 before the
    /// first.
    drawn_at: std::cell::Cell<f64>,
    /// The pin that iTime reads as 0, and whether the next set_time() takes
    /// its place, after restart_clock().
    time_base: std::cell::Cell<(f64, bool)>,
    /// The time step that set_dt() set, which advances the phases while the
    /// time runs free.
    dt: std::cell::Cell<f64>,
    /// True from restart_clock() until a loop reads it.
    pub(crate) restarted: std::cell::Cell<bool>,
    /// The texels per em of each drawn layer's atlas, by the layer's name.
    px_em: HashMap<String, u32>,
    /// The font of an atlas generator.
    atlas_font: AtlasFont,
    /// The atlas cache directory. None takes the generator's own.
    atlas_cache: Option<PathBuf>,
    /// A test sets it to compile the array readers that interpolate in the
    /// shader, on a GL that filters a float texture itself.
    #[cfg(test)]
    shader_lerp: bool,
    /// A test sets it to give the glyphs of every text layer this fragment
    /// shader.
    #[cfg(test)]
    glyphs_fragment: Option<&'static str>,
    /// The layouts run by set_text, for a test to count.
    #[cfg(test)]
    layouts: std::cell::Cell<usize>,
    /// The floats of the vertices of runs handed to the engine, for a test
    /// to count.
    #[cfg(test)]
    run_floats: std::cell::Cell<usize>,
    /// The bytes of GLYPHS handed to the engine, for a test to count.
    #[cfg(test)]
    glyph_bytes: std::cell::Cell<usize>,
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
            scene: std::cell::RefCell::new(None),
            error: String::new(),
            compile_error: false,
            error_line: None,
            error_pass: None,
            display,
            card: None,
            backlight: std::cell::RefCell::new(None),
            alpha: false,
            turns: 0,
            mp: Vec::new(),
            feeds: Vec::new(),
            scene_assets: std::cell::RefCell::new(Vec::new()),
            #[cfg(test)]
            batches: Vec::new(),
            #[cfg(test)]
            test_batches: Vec::new(),
            layers: Vec::new(),
            runs: Vec::new(),
            unsettled: std::cell::Cell::new(false),
            reclassed: std::cell::Cell::new(false),
            reported: std::cell::RefCell::new(HashSet::new()),
            text_errors: std::cell::RefCell::new(HashMap::new()),
            text_errors_seq: std::cell::Cell::new(0),
            atlases: std::cell::RefCell::new(Vec::new()),
            held: std::cell::RefCell::new(HashMap::new()),
            words: std::cell::RefCell::new(HashMap::new()),
            files: std::cell::RefCell::new(Vec::new()),
            file_count: std::cell::Cell::new(0),
            shapes_kept: std::cell::Cell::new(false),
            mixed_kept: std::cell::Cell::new(false),
            curves: HashSet::new(),
            covered: HashSet::new(),
            glyphs: std::cell::RefCell::new(text::Glyphs::default()),
            pass_shown: std::cell::RefCell::new(HashMap::new()),
            places: std::cell::RefCell::new(None),
            colours: std::cell::RefCell::new(HashMap::new()),
            pinned: std::cell::Cell::new(None),
            drawn_at: std::cell::Cell::new(0.0),
            time_base: std::cell::Cell::new((0.0, false)),
            dt: std::cell::Cell::new(DT),
            restarted: std::cell::Cell::new(false),
            px_em: HashMap::new(),
            atlas_font: Box::new(|fonts| Arc::new(text::AtlasFont(fonts.to_vec()))),
            atlas_cache: None,
            #[cfg(test)]
            shader_lerp: false,
            #[cfg(test)]
            glyphs_fragment: None,
            #[cfg(test)]
            layouts: std::cell::Cell::new(0),
            #[cfg(test)]
            run_floats: std::cell::Cell::new(0),
            #[cfg(test)]
            glyph_bytes: std::cell::Cell::new(0),
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
            r.card = Some(dev);
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
    /// A compile error sets compile_error, and error_line and error_pass to
    /// the user-code line.
    /// An error in the scene's common block sets error_pass to Common.
    /// error_line then counts from the start of that block.
    pub fn setup(&mut self, scene: &Scene) -> Result<String> {
        #[cfg(test)]
        let batches = std::mem::take(&mut self.test_batches);
        self.error = String::new();
        self.compile_error = false;
        self.error_line = None;
        self.error_pass = None;
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
        for (name, size) in &seen {
            let (w, h) = match *size {
                ArraySize::Linear(n) => (n, 1),
                ArraySize::Grid(w, h) => (w, h),
            };
            if let Some(e) = self.past_max(&format!("array '{name}'"), w, h) {
                self.error = e;
                return Ok(self.error.clone());
            }
        }
        // The engine draws the hidden passes as copies, and the passes after
        // the last visible one not at all.
        let mut drawn = scene.drawn_passes();
        // The output pass renders at the renderer size. Its buffer feeds the
        // readback, the display and the encoder. The check reads the drawn
        // passes, so a filter pass appended after a sized pass keeps its
        // size, and a sized pass that hidden passes leave last loses it.
        let n = drawn.len();
        let (sw, sh) = self.scene_size();
        let frame = (i64::from(sw), i64::from(sh));
        let out = &drawn[n - 1];
        let set = match (out.size, out.scale) {
            (Some((w, h)), _) => Some(format!("a size ({w}x{h})")),
            (_, Some((x, y))) if x == y => Some(format!("a scale ({x})")),
            (_, Some((x, y))) => Some(format!("a scale ([{x}, {y}])")),
            (None, None) => None,
        };
        if let Some(set) = set {
            crate::report::warn(format!(
                "output pass '{}' sets {set} and renders at the screen size \
                 ({sw}x{sh}); to keep its size, add a filter or a pass after it that reads it \
                 through a channel",
                out.name,
            ));
            let out = &mut drawn.to_mut()[n - 1];
            (out.size, out.scale) = (None, None);
        }
        if let Some(e) = drawn.iter().find_map(|p| p.scale_refusal(frame)) {
            self.error = e;
            return Ok(self.error.clone());
        }
        let assets = scene
            .assets
            .iter()
            .filter_map(|a| Some((format!("asset '{}'", a.name), a.image.as_ref()?)));
        let files = drawn_files(&drawn).filter_map(|l| {
            Some((
                format!("layer '{}': '{}'", l.name, l.image),
                l.data.as_ref()?,
            ))
        });
        let mut images = assets.chain(files);
        if let Some(e) = images.find_map(|(what, img)| {
            self.past_max(&what, i64::from(img.width), i64::from(img.height))
        }) {
            self.error = e;
            return Ok(self.error.clone());
        }
        let last = &drawn[n - 1];
        let npot = self.mipmaps_npot();
        for p in drawn.iter().filter(|p| p.mipmaps) {
            let (w, h) = p.size_in(frame).unwrap_or(frame);
            if let Some(e) = mip_refusal(&p.name, w, h, npot) {
                self.error = e;
                return Ok(self.error.clone());
            }
        }
        // A channel binds to a pass by name. A name that names no pass is an
        // error. buffer=None is the unbound channel, and it reads black.
        let names: Vec<&str> = drawn.iter().map(|p| p.name.as_str()).collect();
        for p in drawn.iter() {
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
        #[cfg(test)]
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
        // The engine counts the assets and the layers' textures that each
        // program samples once it compiles. These counts come first, and a
        // scene that they refuse leaves the running scene in place.
        for p in drawn.iter() {
            let mut used: Vec<&str> = p.arrays.iter().map(|a| a.name.as_str()).collect();
            used.sort_unstable();
            used.dedup();
            let want = p.channels.len() + used.len();
            let mut over = self.units_error(&format!("pass '{}'", p.name), want);
            // A layer samples its own texture, its words' or its image, on
            // top of its pass's channels and arrays, which stay bound for it.
            // A shape samples none.
            let sampling = p.layers.iter().find(|l| l.as_shape().is_none());
            if let (None, Some(l)) = (&over, sampling) {
                over = self.units_error(&format!("layer '{}'", l.name()), want + 1);
            }
            if let Some(e) = over {
                self.error = e;
                return Ok(self.error.clone());
            }
        }
        if let Some(e) = drawn.iter().find_map(Pass::empty) {
            self.error = e;
            return Ok(self.error.clone());
        }
        let mut mp: Vec<(String, c_int)> = Vec::new();
        #[cfg(test)]
        let mut batch_ix: Vec<c_int> = vec![-1; batches.len()];
        let mut layers: Vec<LayerBatches> = Vec::new();
        let mut runs: Vec<Run> = Vec::new();
        let mut glyphs = text::Glyphs::default();
        let hw_linear = self.data_texture_linear();
        let sizes = self.atlas_sizes(&drawn);
        #[cfg(test)]
        let hw_linear = hw_linear && !self.shader_lerp;
        let total = drawn.len();
        let texts = scene.text_names();
        // Every pass buffer stores alpha but the output pass's, which stores
        // it only when the readback keeps it or a channel reads it. An RGB
        // buffer reads back alpha 255 at no cost.
        let out = last.name.as_str();
        let out_read = drawn
            .iter()
            .any(|p| p.channels.iter().any(|c| c.buffer.as_deref() == Some(out)));
        for (i, p) in drawn.iter().enumerate() {
            // A VC4 takes a while over a big shader, and a scene of several
            // compiles them one after another with nothing to show for it.
            crate::report::progress("compile", format!("pass {} of {total}, {}", i + 1, p.name));
            let glsl = match p.has_fragment() {
                true => p.glsl(
                    Some(&scene.params),
                    scene.common.as_deref(),
                    &scene.assets,
                    &texts,
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
                    match p.has_fragment() {
                        true => cglsl.as_ptr(),
                        false => ptr::null(),
                    },
                    c_int::from(alpha),
                )
            };
            if idx >= 0 {
                if let Some((w, h)) = p.size_in(frame) {
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
            let shaders: Vec<Option<crate::layer::Shaders>> = p
                .layers
                .iter()
                .map(|l| {
                    let l = l.as_text()?;
                    let px_em = sizes.get(&l.name).copied().unwrap_or(atlas::PX_EM);
                    let sh = l.shaders(l.detail(px_em));
                    #[cfg(test)]
                    let sh = match self.glyphs_fragment {
                        Some(fs) => crate::layer::Shaders {
                            glyphs: (sh.glyphs.0, fs.to_string()),
                            ..sh
                        },
                        None => sh,
                    };
                    Some(sh)
                })
                .collect();
            let size = p
                .size_in(frame)
                .map_or([f64::from(sw), f64::from(sh)], |(w, h)| {
                    [w as f64, h as f64]
                });
            // A text's class comes from its words laid out into the new
            // texture of curves, whose layouts set_text takes again.
            let candidates: Vec<Option<Candidate>> = p
                .layers
                .iter()
                .map(|l| {
                    if !joins_runs(l) {
                        return None;
                    }
                    let class = match l {
                        Layer::Shape(s) => shape_class(s, size),
                        Layer::Text(t) => {
                            match t.lay_out(&t.text, text::Curves::Shared, &mut glyphs) {
                                Ok(Layout { built: Some(b), .. }) => {
                                    text_class(t, &b.quads.floats, b.size, b.shared, size)
                                }
                                _ => Class::Joins,
                            }
                        }
                        Layer::Image(_) => unreachable!("a run takes no image"),
                    };
                    Some(Candidate {
                        text: l.as_text().is_some(),
                        class,
                        blend: l.blend(),
                    })
                })
                .collect();
            let spans = crate::run::partition(&candidates);
            for (k, (l, sh)) in p.layers.iter().zip(&shaders).enumerate() {
                if idx < 0 {
                    break;
                }
                let name = format!("layer '{}'", l.name());
                let span = spans.iter().find(|r| r.contains(&k));
                let li = match (span, l, sh) {
                    (Some(r), _, _) if r.start < k => {
                        runs.last().expect("the run of its first").batch
                    }
                    (Some(r), _, _) => {
                        let mut members = Vec::new();
                        for (m, sh) in p.layers[r.clone()].iter().zip(&shaders[r.clone()]) {
                            let class =
                                candidates[members.len() + r.start].expect("a member").class;
                            members.push(match (m, sh) {
                                (Layer::Shape(s), _) => Member::shape(s, size),
                                (Layer::Text(t), Some(sh)) => {
                                    let what = format!("layer '{}'", t.name);
                                    match Hold::keep(self.handle, &what, &sh.glyphs) {
                                        Ok(own) => Member::text(t, class, own),
                                        Err(e) => return self.staged_refused(e.to_string(), i),
                                    }
                                }
                                _ => unreachable!("a run holds shapes and texts"),
                            });
                        }
                        let texts = members.iter().filter(|m| m.text.is_some()).count();
                        let mode = Mode::of(members.len(), texts);
                        let program = match (mode, sh) {
                            (Mode::Shapes, _) => self.shape_program()?,
                            (Mode::Mixed, _) => self.run_program()?,
                            (Mode::Alone, Some(sh)) => sh.glyphs.clone(),
                            (Mode::Alone, None) => unreachable!("a text layer has shaders"),
                        };
                        let b = self.add_batch(idx, &name, &program, l.blend())?;
                        runs.push(Run {
                            batch: b,
                            pass: p.name.clone(),
                            size,
                            members,
                            installed: std::cell::Cell::new(Some(mode)),
                            curves: std::cell::Cell::new(Bound::None),
                            sent: std::cell::RefCell::new(Vec::new()),
                        });
                        b
                    }
                    (None, Layer::Image(l), _) => {
                        self.add_batch(idx, &name, &l.shaders(), l.blend)?
                    }
                    (None, _, Some(sh)) => self.add_batch(idx, &name, &sh.glyphs, l.blend())?,
                    (None, _, None) => unreachable!("a text layer has shaders"),
                };
                if li < 0 {
                    return self.batch_refused(&name, i);
                }
                layers.push(LayerBatches {
                    batch: li,
                    colour: -1,
                    coloured: std::cell::Cell::new(false),
                    boxes: -1,
                    name: l.name().to_string(),
                    phases: std::cell::RefCell::new(Vec::new()),
                    shared: std::cell::Cell::new(false),
                    shown: std::cell::RefCell::new(None),
                });
            }
            // A test's batches draw after the layers of their pass.
            #[cfg(test)]
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
                    self.compile_error = unsafe { ffi::ghee_last_error_is_compile(self.handle) };
                    unsafe { ffi::ghee_mp_abort(self.handle) };
                    self.error = format!("{name}: {err}");
                    self.error_pass = Some(ErrorPass::Index(i));
                    crate::report::progress_done();
                    return Ok(self.error.clone());
                }
            }
            // The colour glyphs of each layer come after every layer of the
            // pass, and the error boxes after them, so none hides an error
            // box. A colour batch compiles at the layer's first colour glyph.
            let first = layers.len() - if idx < 0 { 0 } else { p.layers.len() };
            for (dl, l) in layers[first..].iter_mut().zip(&p.layers) {
                let Some(l) = l.as_text() else {
                    continue;
                };
                let name = format!("layer '{}' colour glyphs", l.name);
                dl.colour = self.add_batch_unbuilt(idx, &name, l.blend)?;
                if dl.colour < 0 {
                    return self.batch_refused(&name, i);
                }
            }
            for ((dl, l), sh) in layers[first..].iter_mut().zip(&p.layers).zip(&shaders) {
                let (Some(l), Some(sh)) = (l.as_text(), sh) else {
                    continue;
                };
                let name = format!("layer '{}' error boxes", l.name);
                dl.boxes = self.add_batch(idx, &name, &sh.boxes, Blend::Normal)?;
                if dl.boxes < 0 {
                    return self.batch_refused(&name, i);
                }
            }
            if idx < 0 {
                let err = or_else(self.last_error(), "compile error");
                self.compile_error = unsafe { ffi::ghee_last_error_is_compile(self.handle) };
                unsafe { ffi::ghee_mp_abort(self.handle) };
                let (line, in_common) =
                    user_line(&err, &glsl, &p.fragment, scene.common.as_deref());
                let pass = format!("pass '{}'", p.name);
                self.error = match (line, in_common) {
                    (Some(n), true) => format!("{pass}, line {n} of the common block: {err}"),
                    (Some(n), false) => format!("{pass}, line {n} of its code: {err}"),
                    // The engine's own errors about a pass open with its name.
                    (None, _) if err.starts_with(&pass) => err,
                    (None, _) => format!("{pass}: {err}"),
                };
                self.error_line = line;
                self.error_pass = Some(if in_common {
                    ErrorPass::Common
                } else {
                    ErrorPass::Index(i)
                });
                crate::report::progress_done();
                return Ok(self.error.clone());
            }
            mp.push((p.name.clone(), idx));
        }
        crate::report::progress_done();
        if unsafe { ffi::ghee_mp_commit(self.handle) } < 0 {
            self.error = or_else(self.last_error(), "commit failed");
            return Ok(self.error.clone());
        }
        self.mp = mp;
        #[cfg(test)]
        {
            self.batches = batch_ix;
        }
        self.layers = layers;
        // The last scene's runs let go of their texts' programs once the new
        // scene's runs hold theirs.
        self.runs = runs;
        self.unsettled.set(true);
        self.moved();
        self.drawn_at.set(0.0);
        self.px_em = sizes;
        // A new scene logs its text problems afresh.
        self.reported.borrow_mut().clear();
        if !self.text_errors.borrow().is_empty() {
            self.text_errors.borrow_mut().clear();
            self.text_errors_seq.set(self.text_errors_seq.get() + 1);
        }
        self.held.borrow_mut().clear();
        self.words.borrow_mut().clear();
        self.curves.clear();
        self.covered.clear();
        *self.glyphs.borrow_mut() = glyphs;
        self.pass_shown.borrow_mut().clear();
        for l in scene.text_layers() {
            let tex = CString::new(text::internal(&l.name, "_curves_tex"))?;
            let slug = l.render == Render::Slug && self.layers.iter().any(|d| d.name == l.name);
            if unsafe { ffi::ghee_mp_reads(self.handle, tex.as_ptr()) } {
                self.covered.insert(l.name.clone());
            }
            if slug || self.covered.contains(&l.name) {
                self.curves.insert(l.name.clone());
            }
        }
        // A colour atlas starts again with the words that need it.
        self.colours.borrow_mut().clear();
        self.set_atlases(&drawn)?;
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
            .chain(
                scene
                    .text_layers()
                    .map(|l| text::internal(&l.name, "_curves")),
            )
            .chain(atlases)
            .collect();
        self.files.borrow_mut().clear();
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
        // An image layer's file goes in as an engine asset, one for each
        // file.
        for l in drawn_files(&drawn) {
            let Some(img) = &l.data else {
                self.error = format!(
                    "layer '{}': '{}' was never loaded, the scene was built without a directory",
                    l.name, l.image
                );
                return Ok(self.error.clone());
            };
            if let Err(e) = self.file_asset(&l.name, &l.image, img) {
                self.error = e.to_string();
                return Ok(self.error.clone());
            }
        }
        // A layer lays its words out below, and it looks itself up in
        // self.scene.
        *self.scene.borrow_mut() = Some(scene.clone());
        self.moved();
        for (dl, l) in self.layers.iter().zip(drawn.iter().flat_map(|p| &p.layers)) {
            for (name, ty, value) in l.uniforms() {
                self.set_layer_uniform(dl, &name, ty, value)?;
            }
            let Layer::Image(l) = l else { continue };
            let image = self.image_asset(l)?;
            let shown = self.bind_image(dl.batch, &image).and_then(|()| {
                self.batch_quads(dl.batch, &format!("layer '{}'", l.name), &IMAGE_QUAD, 1)
            });
            if let Err(e) = shown {
                self.error = e.to_string();
                return Ok(self.error.clone());
            }
        }
        for (i, p) in drawn.iter().enumerate() {
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
        // A layer's words that the engine refuses, or a layer whose font was
        // never loaded, refuse the scene, as an asset does.
        for l in scene.text_layers() {
            if let Err(e) = self.lay_text(&l.name, &l.text) {
                self.error = e.to_string();
                return Ok(self.error.clone());
            }
        }
        // The first frame draws the words: they wait here for their glyphs.
        if let Err(e) = self
            .tick_texts(true)
            .and_then(|()| self.regroup())
            .and_then(|()| self.settle())
        {
            self.error = e.to_string();
            return Ok(self.error.clone());
        }

        let mut feeds: Vec<(String, ArraySize)> = Vec::new();
        for (i, p) in drawn.iter().enumerate() {
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

    /// setup() with batches. Each batch draws into its named pass, after that
    /// pass's quad and the pass's layers, and ahead of their colour glyphs
    /// and error boxes, in the order given. A batch's compile error reports
    /// its pass.
    #[cfg(test)]
    pub(crate) fn setup_batches(&mut self, scene: &Scene, batches: &[Batch]) -> Result<String> {
        self.test_batches = batches.to_vec();
        self.setup(scene)
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
        if !(1..=BATCH_ATTRIBUTES).contains(&nattr) {
            bail!("{batch}: a vertex holds 1 to {BATCH_ATTRIBUTES} attributes, got {nattr}");
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
        self.regroup()?;
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
        self.regroup()?;
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
        self.regroup()?;
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
        self.advance()?;
        self.settle()?;
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
    /// Every pass that reads it gets it as a sampler uniform `<name>_tex`,
    /// plus a vec2 `<name>_size` of (width, height). An asset outlives the
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
        let (w, h) = (i64::from(width), i64::from(height));
        if let Some(e) = self.past_max(&format!("asset '{name}'"), w, h) {
            bail!("{e}");
        }
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
        self.regroup()?;
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
    /// this call sets no uniform.
    /// The renderer's scene holds the value, as with_scene() gives it.
    pub fn set_param(
        &self,
        pass_name: &str,
        name: &str,
        ptype: &str,
        value: &ParamValue,
    ) -> Result<bool> {
        if ptype == "string" {
            self.hold_param(Some(pass_name), name, value);
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
                "color" => bail!("color param '{name}' needs {}", crate::layer::COLOUR_FORMS),
                _ => bail!("{ptype} param '{name}' needs {} numbers", &ptype[3..]),
            }
        }
        let json = match value {
            ParamValue::Color(c) => Value::from(c.rgb.to_vec()),
            v => v.to_json(),
        };
        let set = self.set_uniform(idx, name, ptype, json)?;
        self.hold_param(Some(pass_name), name, value);
        Ok(set)
    }

    /// Write `value` into the renderer's scene as the value of param `name`
    /// of pass `pass`, or of the scene when None, when it declares one.
    fn hold_param(&self, pass: Option<&str>, name: &str, value: &ParamValue) {
        let mut scene = self.scene.borrow_mut();
        let Some(scene) = scene.as_mut() else {
            return;
        };
        let params = match pass {
            Some(pass) => match scene.passes.iter_mut().find(|p| p.name == pass) {
                Some(p) => &mut p.params,
                None => return,
            },
            None => &mut scene.params,
        };
        if let Some(p) = params.get_mut(name) {
            p.value = value.clone();
        }
    }

    fn set_uniform(&self, idx: c_int, name: &str, ptype: &str, value: Value) -> Result<bool> {
        let mut cp = create_c_parameter(ptype, &value)?;
        cp.set_c_name(name)?;
        Ok(unsafe { ffi::ghee_mp_set_uniform(self.handle, idx, cp.as_ptr()) })
    }

    /// set_uniform() on the program of pass `idx` alone, and on none of its
    /// batches.
    fn set_pass_uniform(&self, idx: c_int, name: &str, ptype: &str, value: Value) -> Result<bool> {
        let mut cp = create_c_parameter(ptype, &value)?;
        cp.set_c_name(name)?;
        Ok(unsafe { ffi::ghee_mp_set_pass_uniform(self.handle, idx, cp.as_ptr()) })
    }

    /// Upload the rows of GLYPHS that changed since the last upload. A
    /// texture that grew goes up whole, and every layer that reads it takes
    /// its new height.
    fn send_glyphs(&self) -> Result<()> {
        let mut glyphs = self.glyphs.borrow_mut();
        let Some(c) = glyphs.changed() else {
            return Ok(());
        };
        #[cfg(test)]
        self.glyph_bytes
            .set(self.glyph_bytes.get() + c.texels.len());
        if c.height > glyphs.rows() {
            self.set_asset(
                GLYPHS,
                c.width,
                c.height,
                4,
                false,
                AssetData::U8(&c.texels),
            )?;
            self.scene_asset(GLYPHS);
            // A run's batch takes the new height as it settles.
            self.unsettled.set(true);
            for d in self.layers.iter().filter(|d| d.shared.get()) {
                for (suffix, v) in text::Glyphs::uniforms(c.height) {
                    let own = text::internal(GLSL_NAME, suffix);
                    self.set_layer_uniform(d, &own, "float", Value::from(v))?;
                }
            }
        } else {
            self.set_asset_rows(GLYPHS, c.first, c.width, &c.texels)?;
        }
        glyphs.took();
        Ok(())
    }

    /// Set one scene-scoped uniform on every pass live, with no recompile.
    ///
    /// A scene param sits in every pass's preamble, so the set reaches every
    /// pass. A pass that never reads the param holds an inactive uniform, and
    /// the engine skips it.
    /// Returns true when at least one pass takes the value. The renderer's
    /// scene holds the value, as with_scene() gives it.
    pub fn set_scene_param(&self, name: &str, ptype: &str, value: &ParamValue) -> Result<bool> {
        let set = match value {
            ParamValue::Str(_) => false,
            _ => {
                let oks = self
                    .mp
                    .iter()
                    .map(|(pn, _)| self.set_param(pn, name, ptype, value))
                    .collect::<Result<Vec<bool>>>()?;
                oks.into_iter().any(|o| o)
            }
        };
        self.hold_param(None, name, value);
        Ok(set)
    }

    /// Lay out `words`, the new words of the text layer `name`, and upload
    /// them, with no recompile. The words that the layer already shows
    /// return at once.
    ///
    /// A text problem fails nothing: the layer draws an error box for each
    /// character that it cannot draw, and words that fit no texture leave
    /// the last words that did. Each problem logs once, and text_errors()
    /// holds it while it lasts. An error is a font that was never loaded, or
    /// a texture or quads that the engine refuses.
    ///
    /// New words of an MTSDF layer whose glyphs the atlas does not hold yet
    /// wait for them, and the layer draws its last words meanwhile. A render
    /// with the time pinned by set_time() waits for them, and so does a setup.
    fn set_text(&self, name: &str, words: &str) -> Result<()> {
        let laid = self.lay_text(name, words);
        let settled = self.settle();
        laid?;
        settled
    }

    /// set_text() with the runs left to the next settle().
    fn lay_text(&self, name: &str, words: &str) -> Result<bool> {
        let scene = self.scene.borrow();
        let Some(layer) = scene.as_ref().and_then(|s| self.text_layer(s, name)) else {
            return Ok(false);
        };
        if self.words.borrow().get(name).is_some_and(|w| w == words) {
            return Ok(true);
        }
        #[cfg(test)]
        self.layouts.set(self.layouts.get() + 1);
        let curves = match self.curves.contains(name) {
            false => text::Curves::None,
            true if layer.render == Render::Slug && !self.covered.contains(name) => {
                text::Curves::Shared
            }
            true => text::Curves::Own,
        };
        let laid = layer.lay_out(words, curves, &mut self.glyphs.borrow_mut())?;
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
        self.show(layer, laid)?;
        self.words
            .borrow_mut()
            .insert(name.to_string(), words.to_string());
        Ok(true)
    }

    /// Set the live field at `at` to `given`, with no rebuild. The layer's
    /// uniforms that change take their new values, a new image binds in
    /// place of the last, new words in a text layer's `text` lay out as
    /// set_text() lays them out, and a text layer that shows for the first
    /// time since the setup builds its atlas and lays its words out on it. A
    /// layer of a hidden pass takes the value and draws nothing.
    ///
    /// The scene's checks of the field and the value fail with a Refused
    /// error, and change nothing.
    pub fn set_layer(&self, at: &Address, given: Given) -> Result<()> {
        let refused = |e: anyhow::Error| anyhow::Error::new(Refused(e.to_string()));
        let (before, shown, image, value) = {
            let scene = self.scene.borrow();
            let Some(layer) = scene.as_ref().and_then(|s| s.layer(at.layer)) else {
                return Err(Refused(format!("the scene has no layer '{}'", at.layer)).into());
            };
            let live = layer.live(at.effect, at.field).map_err(refused)?;
            let value = match given {
                Given::Json(v) => v.clone(),
                Given::Numbers(v) => live.of_numbers(v).map_err(refused)?,
                Given::Text(t) => live.of_text(t).map_err(refused)?,
            };
            let shown = layer.as_text().is_some_and(|l| l.visible);
            let image = layer.as_image().map(|l| l.image.clone());
            (layer.uniforms(), shown, image, value)
        };
        if let (Some(_), Some(file)) = (&image, value.as_str().filter(|v| is_image_file(v))) {
            if self.layers.iter().any(|d| d.name == at.layer) {
                self.load_file(at.layer, file).map_err(refused)?;
            }
        }
        if let Some(scene) = self.scene.borrow_mut().as_mut() {
            scene.set_layer(at, &value).map_err(refused)?;
        }
        if let (None, "text", Some(words)) = (at.effect, at.field, value.as_str()) {
            return self.set_text(at.layer, words);
        }
        let Some(dl) = self.drawn(at.layer) else {
            return Ok(());
        };
        let now_shown = {
            let scene = self.scene.borrow();
            let Some(layer) = scene.as_ref().and_then(|s| s.layer(at.layer)) else {
                return Ok(());
            };
            for (name, ty, value) in layer.uniforms() {
                if !before.iter().any(|(n, _, v)| *n == name && *v == value) {
                    self.set_layer_uniform(dl, &name, ty, value)?;
                }
            }
            if let Layer::Image(l) = layer {
                if image.as_ref() != Some(&l.image) {
                    self.bind_image(dl.batch, &self.image_asset(l)?)?;
                }
            }
            if let Layer::Shape(l) = layer {
                self.reshape(l);
            }
            layer.as_text().is_some_and(|l| l.visible)
        };
        if now_shown && !shown {
            self.first_shown(at.layer)?;
        }
        self.settle()
    }

    /// The shaders of every shape. The first shape of the renderer compiles
    /// them, and the engine keeps the program while the renderer lives, so
    /// no shape after it compiles anything.
    fn shape_program(&self) -> Result<crate::layer::Program> {
        let (vs, fs) = crate::shape::program();
        if !self.shapes_kept.get() {
            let (cvs, cfs) = (CString::new(vs.as_str())?, CString::new(fs.as_str())?);
            if !unsafe { ffi::ghee_mp_keep_program(self.handle, cvs.as_ptr(), cfs.as_ptr()) } {
                bail!(
                    "the shapes: {}",
                    or_else(self.last_error(), "compile error")
                );
            }
            self.shapes_kept.set(true);
        }
        Ok((vs, fs))
    }

    /// The shaders of every run of a text and another member, which the
    /// engine keeps from the first such run on while the renderer lives.
    fn run_program(&self) -> Result<crate::layer::Program> {
        let (vs, fs) = crate::run::program();
        if !self.mixed_kept.get() {
            let (cvs, cfs) = (CString::new(vs.as_str())?, CString::new(fs.as_str())?);
            if !unsafe { ffi::ghee_mp_keep_program(self.handle, cvs.as_ptr(), cfs.as_ptr()) } {
                bail!(
                    "the runs of shapes and texts: {}",
                    or_else(self.last_error(), "compile error")
                );
            }
            self.mixed_kept.set(true);
        }
        Ok((vs, fs))
    }

    /// The engine asset that image layer `l` draws: its image asset, or the
    /// engine asset of its file, which a setup or a live set loaded.
    fn image_asset(&self, l: &ImageLayer) -> Result<String> {
        if !l.is_file() {
            return Ok(l.image.clone());
        }
        let files = self.files.borrow();
        match files.iter().find(|(f, _)| *f == l.image) {
            Some((_, asset)) => Ok(asset.clone()),
            None => bail!("layer '{}': '{}' was never loaded", l.name, l.image),
        }
    }

    /// The engine asset of the image file `file` of layer `layer`: the one
    /// that the renderer holds, else one loaded from the scene's files.
    fn load_file(&self, layer: &str, file: &str) -> Result<String> {
        if let Some((_, asset)) = self.files.borrow().iter().find(|(f, _)| f == file) {
            return Ok(asset.clone());
        }
        let img = {
            let scene = self.scene.borrow();
            match scene.as_ref().and_then(|s| s.root.as_ref()) {
                Some(root) => root.image(layer, file)?,
                None => bail!(
                    "layer '{layer}': '{file}' is a file, and the scene was built without a \
                     directory"
                ),
            }
        };
        self.file_asset(layer, file, &img)
    }

    /// Hand the engine the pixels `img` of the image file `file` of layer
    /// `layer` as an asset, once, and return the asset's name.
    fn file_asset(&self, layer: &str, file: &str, img: &crate::assets::Image) -> Result<String> {
        if let Some((_, asset)) = self.files.borrow().iter().find(|(f, _)| f == file) {
            return Ok(asset.clone());
        }
        let (w, h) = (i64::from(img.width), i64::from(img.height));
        if let Some(e) = self.past_max(&format!("layer '{layer}': '{file}'"), w, h) {
            bail!("{e}");
        }
        let n = self.file_count.get();
        self.file_count.set(n + 1);
        // A layer's name starts with a letter, so no program reads
        // `<name>_tex` of this asset, and it binds to the batches given it
        // alone.
        let name = text::internal(&n.to_string(), "_file");
        let data = match &img.pixels {
            Pixels::U8(d) => AssetData::U8(d),
            Pixels::F32(d) => AssetData::F32(d),
        };
        self.set_asset(&name, img.width, img.height, img.components, false, data)?;
        self.scene_asset(&name);
        self.files
            .borrow_mut()
            .push((file.to_string(), name.clone()));
        Ok(name)
    }

    /// The size in pixels of drawn pass `pass`: the scene's for the pass
    /// that holds the frame, else its own size or the scene's.
    fn pass_size(&self, pass: &str) -> [f64; 2] {
        let (sw, sh) = self.scene_size();
        let full = [f64::from(sw), f64::from(sh)];
        if self.mp.last().is_some_and(|(n, _)| n == pass) {
            return full;
        }
        let scene = self.scene.borrow();
        let size = scene
            .as_ref()
            .and_then(|s| s.passes.iter().find(|p| p.name == pass))
            .and_then(|p| p.size_in((i64::from(sw), i64::from(sh))));
        size.map_or(full, |(w, h)| [w as f64, h as f64])
    }

    /// Add `layer`, a layer as a scene declares it, to the running scene at
    /// `index` among the layers of pass `pass`, with no rebuild. Every other
    /// layer keeps its program, its atlas, its phases and its quads, and
    /// every pass keeps its program, its arrays and its buffers. A shape
    /// layer compiles nothing, an image layer loads its picture, and a text
    /// layer compiles its shaders and builds its atlas. A layer whose
    /// shaders come out the same as another layer's draws through the same
    /// program, and compiles nothing. The layer's files resolve as the
    /// scene's did at its load. A layer of a hidden pass joins the scene and
    /// draws nothing.
    ///
    /// A layer that the scene refuses, as a scene file would refuse it in
    /// that place, fails with a Refused error and changes nothing.
    pub fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Result<()> {
        let at = format!("the layer added at {index} to pass '{pass}'");
        let mut new = Layer::from_dict(&at, layer).map_err(refused)?;
        let drawn = self.admit(pass, &mut new)?;
        let name = new.name().to_string();
        self.scene
            .borrow_mut()
            .as_mut()
            .expect("admit() found the scene")
            .insert_layer(pass, index, new.clone())
            .map_err(refused)?;
        self.moved();
        if !drawn {
            return Ok(());
        }
        if let Err(e) = self.draw_new(pass, index, &new) {
            self.unwind(&name);
            return Err(e);
        }
        self.refresh()
    }

    /// Take layer `name`, which failed to draw, out of the engine and out of
    /// the scene.
    fn unwind(&mut self, name: &str) {
        // The layer leaves the engine as best it can, and the error that
        // the caller holds says why.
        let _ = self.drop_drawn(name);
        if let Some(scene) = self.scene.borrow_mut().as_mut() {
            scene.remove_layer(name);
        }
        self.moved();
        let _ = self.refresh();
    }

    /// Ready `new`, a layer that joins pass `pass`, as the scene's load
    /// readied its own: its fonts or its image file, and a texture unit for
    /// a layer that samples one. Returns true when the pass draws.
    fn admit(&mut self, pass: &str, new: &mut Layer) -> Result<bool> {
        let (drawn, want) = {
            let scene = self.scene.borrow();
            let Some(scene) = scene.as_ref() else {
                return Err(Refused("the renderer holds no scene".to_string()).into());
            };
            scene.load_layer(new).map_err(refused)?;
            let p = scene.passes.iter().find(|p| p.name == pass);
            let drawn = p.is_some_and(|p| p.visible) && self.pass_index(pass).is_some();
            let want = p.map_or(0, |p| {
                let mut arrays: Vec<&str> = p.arrays.iter().map(|a| a.name.as_str()).collect();
                arrays.sort_unstable();
                arrays.dedup();
                p.channels.len() + arrays.len()
            });
            (drawn, want)
        };
        // A text or an image layer samples one texture on top of its pass's
        // channels and arrays.
        if drawn && new.as_shape().is_none() {
            if let Some(e) = self.units_error(&format!("layer '{}'", new.name()), want + 1) {
                return Err(Refused(e).into());
            }
        }
        if let Layer::Image(l) = &*new {
            if drawn && l.is_file() {
                let img = l.data.as_ref().expect("load_layer read the file");
                self.file_asset(&l.name, &l.image, img).map_err(refused)?;
            }
        }
        Ok(drawn)
    }

    /// Draw `new`, which the model holds at `index` of drawn pass `pass`.
    fn draw_new(&mut self, pass: &str, index: usize, new: &Layer) -> Result<()> {
        let idx = self.pass_index(pass).expect("the pass is drawn");
        match new {
            Layer::Shape(l) => self.add_shape(pass, index, l),
            Layer::Image(l) => self.add_image(pass, idx, index, l),
            Layer::Text(l) => self.add_text(pass, idx, index, l),
        }
    }

    /// Build the layer of the name of `layer`, a layer as a scene declares
    /// it, again from `layer`, in its place among the layers of its pass.
    /// Every other layer keeps its program, its atlas, its phases and its
    /// quads, and every pass keeps its program, its arrays and its buffers.
    /// The layer's moving effects carry their phases on, and its kind may
    /// change.
    ///
    /// A layer that the scene refuses in that place, or a name that names no
    /// layer, fails with a Refused error and changes nothing.
    pub fn replace_layer(&mut self, layer: &Value) -> Result<()> {
        let mut new = Layer::from_dict("the new layer", layer).map_err(refused)?;
        let name = new.name().to_string();
        let (pass, index) = {
            let scene = self.scene.borrow();
            let place = scene.as_ref().and_then(|s| {
                let (p, i) = s.place_of(&name)?;
                Some((s.passes[p].name.clone(), i))
            });
            match place {
                Some(place) => place,
                None => return Err(Refused(format!("the scene has no layer '{name}'")).into()),
            }
        };
        let drawn = self.admit(&pass, &mut new)?;
        self.scene
            .borrow_mut()
            .as_mut()
            .expect("admit() found the scene")
            .replace_layer(new.clone())
            .map_err(refused)?;
        self.moved();
        if !drawn {
            return Ok(());
        }
        let phases = self.drop_drawn(&name)?;
        if let Err(e) = self.draw_new(&pass, index, &new) {
            self.unwind(&name);
            return Err(e);
        }
        if let (Some(l), Some(dl)) = (new.as_text(), self.drawn(&name)) {
            let mut carried = Vec::new();
            for (kind, uniform, _) in l.paces() {
                if let Some((_, phase)) = phases.iter().find(|(k, _)| *k == kind) {
                    self.set_layer_uniform(dl, &uniform, "float", Value::from(*phase))?;
                    carried.push((kind, *phase));
                }
            }
            *dl.phases.borrow_mut() = carried;
        }
        self.refresh()
    }

    /// Give the layer of the name of `layer` the look of `layer`, a layer as
    /// a scene declares it: every key that `layer` leaves out takes its
    /// default. An effect of the layer that `layer` leaves out stays,
    /// bypassed, with its values, and so does a rect's border. A look that
    /// differs in live fields alone takes them as set_layer() would, with no
    /// rebuild, and any other difference builds the layer again in its
    /// place, as replace_layer() does. Returns false when the layer has that
    /// look already, and then changes nothing.
    ///
    /// A layer that the scene refuses, or a name that names no layer, fails
    /// with a Refused error and changes nothing.
    pub fn update_layer(&mut self, layer: &Value) -> Result<bool> {
        let found = match self.scene.borrow().as_ref() {
            Some(scene) => scene.look(layer).map_err(refused)?,
            None => return Err(Refused("the renderer holds no scene".to_string()).into()),
        };
        let Some(Look {
            old,
            new,
            look,
            rebuild,
        }) = found
        else {
            return Ok(false);
        };
        if rebuild {
            self.replace_layer(&look)?;
            return Ok(true);
        }
        let name = new.name().to_string();
        let drawn = self.drawn(&name).is_some();
        if let (Layer::Image(n), Layer::Image(o)) = (&new, &old) {
            if drawn && n.image != o.image && is_image_file(&n.image) {
                self.load_file(&name, &n.image).map_err(refused)?;
            }
        }
        self.scene
            .borrow_mut()
            .as_mut()
            .expect("the layer was just found")
            .swap_live(new.clone())
            .map_err(refused)?;
        let words = match (&new, &old) {
            (Layer::Text(n), Layer::Text(o)) if n.text != o.text => Some(n.text.as_str()),
            _ => None,
        };
        let shown = self.show_live(&old, &new, words);
        self.refresh()?;
        shown?;
        Ok(true)
    }

    /// Draw the live fields of `new`, which the scene holds now in place of
    /// `old`, its look before: the layer's uniforms that changed, its image,
    /// its shape's quads and its new words `words`, which the runs take at
    /// the next settle(). A text layer that shows for the first time since
    /// the setup builds its atlas.
    fn show_live(&self, old: &Layer, new: &Layer, words: Option<&str>) -> Result<()> {
        let name = new.name();
        let Some(dl) = self.drawn(name) else {
            return Ok(());
        };
        for (u, ty, value) in old.uniforms_changed(new) {
            self.set_layer_uniform(dl, &u, ty, value)?;
        }
        if let (Layer::Image(n), Layer::Image(o)) = (new, old) {
            if n.image != o.image {
                self.bind_image(dl.batch, &self.image_asset(n)?)?;
            }
        }
        if let Layer::Shape(l) = new {
            self.reshape(l);
        }
        if let Some(words) = words {
            self.lay_text(name, words)?;
        }
        let shown = |l: &Layer| l.as_text().is_some_and(|l| l.visible);
        if shown(new) && !shown(old) {
            self.first_shown(name)?;
        }
        Ok(())
    }

    /// Give each layer named by a call of `calls` the look of that call, as
    /// update_layer() gives it the same look written as a scene declares
    /// it, when the look differs from the layer's in live fields alone. The
    /// shapes of one run that changed take one upload. Leaves for
    /// update_layer() a call that draw::layer_of() makes no layer of, a
    /// look that builds its layer again or gives an image layer another
    /// image, and a name that names no layer of the scene.
    pub fn update_calls(&mut self, calls: &[Call]) -> Result<Taken> {
        let mut taken = Taken::default();
        // The runs take the quads that their members hold whatever fails.
        let looped = self.take_calls(calls, &mut taken);
        let settled = self.refresh();
        looped.and(settled)?;
        Ok(taken)
    }

    /// The calls of update_calls(), into `taken`.
    fn take_calls(&self, calls: &[Call], taken: &mut Taken) -> Result<()> {
        for (k, call) in calls.iter().enumerate() {
            let found = {
                let scene = self.scene.borrow();
                let Some(scene) = scene.as_ref() else {
                    taken.left.push(k);
                    continue;
                };
                let places = self.places();
                let Some(&(p, i)) = places.scene.get(&call.layer) else {
                    taken.left.push(k);
                    continue;
                };
                let old = &scene.passes[p].layers[i];
                match draw::layer_of(call, old) {
                    Some(new) if new == *old => continue,
                    Some(new) if old.builds_as(&new) && !other_image(old, &new) => {
                        Some((p, i, new))
                    }
                    _ => None,
                }
            };
            let Some((p, i, new)) = found else {
                taken.left.push(k);
                continue;
            };
            let (old, words) = self
                .scene
                .borrow_mut()
                .as_mut()
                .expect("the layer was just found")
                .swap_live_at(p, i, new);
            taken.changed.push(k);
            let scene = self.scene.borrow();
            let new = &scene.as_ref().expect("the layer was just found").passes[p].layers[i];
            self.show_live(&old, new, words.as_deref())?;
        }
        Ok(())
    }

    /// The size in pixels of the block that `words` lay out in, as a text
    /// layer of pass `pass` would lay them out in its font `font_family`,
    /// the built-in font when None: at `font_size` pixels per em, or at the
    /// size that fits the block to the pass, centred.
    ///
    /// A font that the scene cannot give the layer fails with a Refused
    /// error.
    pub fn measure(
        &self,
        pass: &str,
        words: &str,
        font_size: Option<f64>,
        font_family: Option<&str>,
    ) -> Result<[f64; 2]> {
        let scene = self.scene.borrow();
        let Some(scene) = scene.as_ref() else {
            return Err(Refused("the renderer holds no scene".to_string()).into());
        };
        scene
            .measure(self.pass_size(pass), words, font_size, font_family)
            .map_err(refused)
    }

    /// Take the drawing of layer `name` out of the engine: its batches, its
    /// place in a run of shapes, its atlas and what its words held. The
    /// model keeps the layer. Returns the phases of its moving effects.
    fn drop_drawn(&mut self, name: &str) -> Result<Vec<(Kind, f64)>> {
        let Some(k) = self.layers.iter().position(|d| d.name == name) else {
            return Ok(Vec::new());
        };
        let pass = self
            .runs
            .iter()
            .find(|r| r.find(name).is_some())
            .map_or_else(
                || {
                    let scene = self.scene.borrow();
                    let (p, _) = *self.places().scene.get(name)?;
                    Some(scene.as_ref()?.passes[p].name.clone())
                },
                |r| Some(r.pass.clone()),
            );
        let member = self.places().runs.contains_key(name);
        let dl = self.layers.remove(k);
        self.moved();
        for batch in [dl.batch, dl.colour, dl.boxes] {
            if batch >= 0 && !(member && batch == dl.batch) {
                self.remove_batch(batch)?;
            }
        }
        // The runs of the pass group again with the layer gone, which takes
        // a member out of its run and joins the runs that the layer split.
        if let Some(pass) = pass {
            self.rerun(&pass, Vec::new())?;
        }
        // An atlas that no layer draws from stays, for a layer of the same
        // font and detail to take with its glyphs.
        let atlas = text::internal(name, "_atlas");
        for slot in self.atlases.borrow_mut().iter_mut() {
            slot.textures.retain(|t| t.0 != atlas);
        }
        self.px_em.remove(name);
        self.words.borrow_mut().remove(name);
        self.held.borrow_mut().remove(name);
        self.colours.borrow_mut().remove(name);
        if self.text_errors.borrow_mut().remove(name).is_some() {
            self.text_errors_seq.set(self.text_errors_seq.get() + 1);
        }
        // A pass that reads the layer's `_cover` keeps reading its curves.
        self.curves.remove(name);
        let tex = CString::new(text::internal(name, "_curves_tex"))?;
        if unsafe { ffi::ghee_mp_reads(self.handle, tex.as_ptr()) } {
            self.curves.insert(name.to_string());
        }
        Ok(dl.phases.into_inner())
    }

    /// Take batch `batch` out of the engine.
    fn remove_batch(&self, batch: c_int) -> Result<()> {
        if !unsafe { ffi::ghee_mp_remove_batch(self.handle, batch) } {
            bail!(
                "{}",
                or_else(self.last_error(), &format!("batch {batch} stayed"))
            );
        }
        Ok(())
    }

    /// The names of the layers of pass `pass`, in their order.
    fn pass_layers(&self, pass: &str) -> Vec<String> {
        let scene = self.scene.borrow();
        scene
            .as_ref()
            .and_then(|s| s.passes.iter().find(|p| p.name == pass))
            .map(|p| p.layers.iter().map(|l| l.name().to_string()).collect())
            .unwrap_or_default()
    }

    /// The drawn layer named `name`.
    fn drawn(&self, name: &str) -> Option<&LayerBatches> {
        let k = *self.places().drawn.get(name)?;
        Some(&self.layers[k])
    }

    /// The text layer of the scene named `name`.
    fn text_layer<'s>(&self, scene: &'s Scene, name: &str) -> Option<&'s TextLayer> {
        let (p, i) = *self.places().scene.get(name)?;
        scene.passes.get(p)?.layers.get(i)?.as_text()
    }

    /// Drop the places of the layers, after a change to the scene's layers,
    /// the drawn layers or the runs.
    fn moved(&self) {
        self.places.borrow_mut().take();
    }

    /// Where each layer sits now.
    fn places(&self) -> std::cell::Ref<'_, Places> {
        #[cfg(test)]
        if let Some(held) = self.places.borrow().as_ref() {
            assert!(*held == self.places_now(), "the places outlived a change");
        }
        if self.places.borrow().is_none() {
            let now = self.places_now();
            *self.places.borrow_mut() = Some(now);
        }
        std::cell::Ref::map(self.places.borrow(), |p| {
            p.as_ref().expect("the places were just made")
        })
    }

    fn places_now(&self) -> Places {
        let mut places = Places::default();
        if let Some(scene) = self.scene.borrow().as_ref() {
            for (p, pass) in scene.passes.iter().enumerate() {
                for (i, l) in pass.layers.iter().enumerate() {
                    places.scene.insert(l.name().to_string(), (p, i));
                }
            }
        }
        for (k, d) in self.layers.iter().enumerate() {
            places.drawn.insert(d.name.clone(), k);
        }
        for (r, run) in self.runs.iter().enumerate() {
            for (k, m) in run.members.iter().enumerate() {
                places.runs.insert(m.name.clone(), (r, k));
            }
        }
        places
    }

    /// The batch after which a new layer at `index` of pass `pass` draws:
    /// the batch of the layer before it, or -1 ahead of every batch of the
    /// pass.
    fn after(&self, pass: &str, index: usize) -> c_int {
        let names = self.pass_layers(pass);
        match index.checked_sub(1).and_then(|k| names.get(k)) {
            Some(n) => self.drawn(n).map_or(-1, |d| d.batch),
            None => -1,
        }
    }

    /// Put `dl` among the drawn layers, after the layer before it in its
    /// pass, or last.
    fn put_drawn(&mut self, pass: &str, index: usize, dl: LayerBatches) {
        let names = self.pass_layers(pass);
        let before = index.checked_sub(1).and_then(|k| names.get(k));
        let at = before
            .and_then(|n| self.layers.iter().position(|d| d.name == *n))
            .map_or(self.layers.len(), |k| k + 1);
        self.layers.insert(at, dl);
        self.moved();
    }

    /// The run of layer `name` and its place there.
    fn member_of(&self, name: &str) -> Option<(usize, usize)> {
        self.places().runs.get(name).copied()
    }

    /// Group the layers of drawn pass `pass` into runs again, as
    /// crate::run::partition() groups them, with `fresh` the members of the
    /// layers that join the pass. A run that starts at the layer where a run
    /// started keeps that run's batch, and any other run takes a new batch
    /// right after the batch of the layer before it, so the batches draw in
    /// the order of the layers. A layer's blend changes as it is built
    /// again, which takes it out of its run first, so a kept batch holds the
    /// run's blend. settle() gives the batches their programs and their
    /// quads.
    fn rerun(&mut self, pass: &str, fresh: Vec<Member>) -> Result<()> {
        let Some(pass_index) = self.pass_index(pass) else {
            return Ok(());
        };
        let size = self.pass_size(pass);
        let mut old: Vec<Run> = Vec::new();
        let mut k = 0;
        while k < self.runs.len() {
            match self.runs[k].pass == pass {
                true => old.push(self.runs.remove(k)),
                false => k += 1,
            }
        }
        self.moved();
        let olds: Vec<Vec<String>> = old
            .iter()
            .map(|r| r.members.iter().map(|m| m.name.clone()).collect())
            .collect();
        let mut pool: HashMap<String, Member> = HashMap::new();
        for m in fresh
            .into_iter()
            .chain(old.iter_mut().flat_map(|r| r.members.drain(..)))
        {
            pool.insert(m.name.clone(), m);
        }
        let (names, candidates): (Vec<String>, Vec<Option<Candidate>>) = {
            let scene = self.scene.borrow();
            let layers = scene
                .as_ref()
                .and_then(|s| s.passes.iter().find(|p| p.name == pass))
                .map_or(&[][..], |p| &p.layers[..]);
            layers
                .iter()
                .map(|l| {
                    let m = pool
                        .get(l.name())
                        .filter(|_| joins_runs(l) && self.drawn(l.name()).is_some());
                    let c = m.map(|m| {
                        if m.reclass.replace(false) {
                            m.class.set(self.class_now(m, l, size));
                        }
                        Candidate {
                            text: m.text.is_some(),
                            class: m.class.get(),
                            blend: l.blend(),
                        }
                    });
                    (l.name().to_string(), c)
                })
                .unzip()
        };
        let spans = crate::run::partition(&candidates);
        // Each run's batch, kept or new, before any member moves, so a
        // batch that the engine refuses leaves the runs as they were.
        let mut batches: Vec<(c_int, Option<usize>)> = Vec::new();
        let mut claimed = vec![false; old.len()];
        let mut planned: HashMap<&str, c_int> = HashMap::new();
        for span in &spans {
            let first = &names[span.start];
            let blend = candidates[span.start].expect("a run's layer").blend;
            let kept = (0..old.len()).find(|k| !claimed[*k] && olds[*k][0] == *first);
            let made = match kept {
                Some(k) => {
                    claimed[k] = true;
                    Ok(old[k].batch)
                }
                None => {
                    let before = names[..span.start].iter().rev().find_map(|n| {
                        planned
                            .get(n.as_str())
                            .copied()
                            .or_else(|| self.drawn(n).map(|d| d.batch))
                    });
                    self.insert_batch(
                        pass_index,
                        before.unwrap_or(-1),
                        &format!("layer '{first}'"),
                        None,
                        blend,
                    )
                }
            };
            let batch = match made {
                Ok(b) => b,
                Err(e) => {
                    for (b, k) in &batches {
                        if k.is_none() {
                            let _ = self.remove_batch(*b);
                        }
                    }
                    for (run, names) in old.iter_mut().zip(&olds) {
                        run.members = names.iter().filter_map(|n| pool.remove(n)).collect();
                    }
                    self.runs.extend(old);
                    self.moved();
                    return Err(e);
                }
            };
            for n in &names[span.clone()] {
                planned.insert(n, batch);
            }
            batches.push((batch, kept));
        }
        for (span, (batch, kept)) in spans.iter().zip(batches) {
            let members: Vec<Member> = names[span.clone()]
                .iter()
                .map(|n| {
                    let mut m = pool.remove(n).expect("each member of a run is in the pool");
                    m.placed = m.class.get();
                    m
                })
                .collect();
            for d in self
                .layers
                .iter_mut()
                .filter(|d| names[span.clone()].contains(&d.name))
            {
                d.batch = batch;
            }
            let was = kept.map(|k| &old[k]);
            let same = kept.is_some_and(|k| olds[k] == names[span.clone()]);
            self.runs.push(Run {
                batch,
                pass: pass.to_string(),
                size,
                members,
                installed: std::cell::Cell::new(was.and_then(|r| r.installed.get())),
                curves: std::cell::Cell::new(was.map_or(Bound::None, |r| r.curves.get())),
                sent: std::cell::RefCell::new(match same {
                    true => was.map(|r| r.sent.take()).unwrap_or_default(),
                    false => Vec::new(),
                }),
            });
        }
        self.moved();
        self.unsettled.set(true);
        // The batches that no run kept go, and the members of layers that
        // left the pass go with the pool.
        let mut gone = Ok(());
        for (k, run) in old.iter().enumerate() {
            if !claimed[k] {
                gone = gone.and(self.remove_batch(run.batch));
            }
        }
        gone
    }

    /// The class of member `m` of a pass of `size` pixels, layer `l`, from
    /// its quads and `l`'s settings now.
    fn class_now(&self, m: &Member, l: &Layer, size: [f64; 2]) -> Class {
        match (l, &m.text) {
            (Layer::Shape(s), _) => shape_class(s, size),
            (Layer::Text(t), Some(own)) => {
                text_class(t, &m.quads.borrow(), own.size.get(), own.shared.get(), size)
            }
            _ => m.class.get(),
        }
    }

    /// Group the runs of each pass again where a member's class changed.
    fn regroup(&mut self) -> Result<()> {
        if !self.reclassed.replace(false) {
            return Ok(());
        }
        let mut passes: Vec<String> = Vec::new();
        {
            let scene = self.scene.borrow();
            for run in &self.runs {
                for m in run.members.iter().filter(|m| m.reclass.get()) {
                    m.reclass.set(false);
                    let layer = scene.as_ref().and_then(|s| {
                        let (p, i) = *self.places().scene.get(&m.name)?;
                        s.passes[p].layers.get(i)
                    });
                    if let Some(l) = layer {
                        m.class.set(self.class_now(m, l, run.size));
                    }
                }
                if run.members.iter().any(|m| m.class.get() != m.placed)
                    && !passes.contains(&run.pass)
                {
                    passes.push(run.pass.clone());
                }
            }
        }
        for pass in passes {
            self.rerun(&pass, Vec::new())?;
        }
        Ok(())
    }

    /// regroup(), then settle().
    fn refresh(&mut self) -> Result<()> {
        self.regroup()?;
        self.settle()
    }

    /// Give each run's batch the program of its mode, the texture of curves
    /// that it reads, a run of one text the text's uniforms, and the quads
    /// of its members that changed: every quad when a member's count of
    /// quads changed, else one upload from the first changed member to the
    /// last.
    fn settle(&self) -> Result<()> {
        if !self.unsettled.replace(false) {
            return Ok(());
        }
        let settled = self.runs.iter().try_for_each(|run| self.settle_run(run));
        if settled.is_err() {
            self.unsettled.set(true);
        }
        settled
    }

    fn settle_run(&self, run: &Run) -> Result<()> {
        let mode = run.mode();
        let first = &run.members[0].name;
        let what = format!("layer '{first}'");
        let alone = run.members[0].text.as_ref().filter(|_| mode == Mode::Alone);
        if run.installed.get() != Some(mode) {
            let program = match (mode, alone) {
                (Mode::Shapes, _) => self.shape_program()?,
                (Mode::Mixed, _) => self.run_program()?,
                (Mode::Alone, Some(t)) => (
                    t.own.vertex.to_string_lossy().into_owned(),
                    t.own.fragment.to_string_lossy().into_owned(),
                ),
                (Mode::Alone, None) => unreachable!("a run of one text"),
            };
            let (vs, fs) = (CString::new(program.0)?, CString::new(program.1)?);
            if !unsafe {
                ffi::ghee_mp_set_batch_program(self.handle, run.batch, vs.as_ptr(), fs.as_ptr())
            } {
                bail!("{what}: {}", or_else(self.last_error(), "compile error"));
            }
            run.installed.set(Some(mode));
            run.curves.set(Bound::None);
            run.sent.borrow_mut().clear();
            if alone.is_some() {
                self.install(run)?;
            }
        }
        let rows = self.glyphs.borrow().rows();
        let glyphs = match rows {
            0 => Bound::None,
            r => Bound::Glyphs(r),
        };
        let want = match (mode, alone) {
            (Mode::Shapes, _) => Bound::None,
            (Mode::Alone, Some(t)) if !t.shared.get() => Bound::Own,
            _ => glyphs,
        };
        let fresh = alone.is_some_and(|t| t.fresh.replace(false));
        if want != Bound::None && (want != run.curves.get() || fresh) {
            match want {
                Bound::Own if !alone.is_some_and(|t| t.texture.get()) => {}
                Bound::Own => {
                    let curves = text::internal(first, "_curves");
                    self.bind_asset(run.batch, &text::internal(GLSL_NAME, "_curves"), &curves)?;
                    run.curves.set(want);
                }
                _ => {
                    self.bind_glyphs(run.batch, rows)?;
                    run.curves.set(want);
                }
            }
        }
        self.run_quads(run, mode, &what)
    }

    /// Set on the batch of `run`, a run of one text, the text's uniforms
    /// and those of its words.
    fn install(&self, run: &Run) -> Result<()> {
        let scene = self.scene.borrow();
        let first = &run.members[0].name;
        let Some(l) = scene.as_ref().and_then(|s| self.text_layer(s, first)) else {
            return Ok(());
        };
        for (u, ty, value) in l.uniforms() {
            self.set_batch_uniform(run.batch, &u, ty, value)?;
        }
        let Some(dl) = self.drawn(first) else {
            return Ok(());
        };
        if let Some((values, size)) = dl.shown.borrow().as_ref() {
            for (suffix, v) in values {
                let own = text::internal(GLSL_NAME, suffix);
                self.set_batch_uniform(run.batch, &own, "float", Value::from(*v))?;
            }
            let own = text::internal(GLSL_NAME, "_size");
            self.set_batch_uniform(run.batch, &own, "vec2", Value::from(size.to_vec()))?;
        }
        Ok(())
    }

    /// Bind GLYPHS at `rows` rows to `batch`, with the uniforms that read it.
    fn bind_glyphs(&self, batch: c_int, rows: u32) -> Result<()> {
        self.bind_asset(batch, &text::internal(GLSL_NAME, "_curves"), GLYPHS)?;
        for (suffix, v) in text::Glyphs::uniforms(rows) {
            let own = text::internal(GLSL_NAME, suffix);
            self.set_batch_uniform(batch, &own, "float", Value::from(v))?;
        }
        Ok(())
    }

    /// Hand the batch of `run` the vertices of its members in `mode` that
    /// changed since it last took them. `what` names the batch in an error.
    fn run_quads(&self, run: &Run, mode: Mode, what: &str) -> Result<()> {
        let counts: Vec<usize> = run.members.iter().map(Member::count).collect();
        let all = *run.sent.borrow() != counts;
        let span = match all {
            true => 0..run.members.len(),
            false => {
                let changed = |m: &&Member| m.dirty.get();
                let Some(first) = run.members.iter().position(|m| changed(&m)) else {
                    return Ok(());
                };
                let last = run
                    .members
                    .iter()
                    .rposition(|m| changed(&m))
                    .expect("a first");
                first..last + 1
            }
        };
        let nattr = match mode {
            Mode::Mixed => crate::run::ATTRIBUTES,
            _ => crate::shape::ATTRIBUTES,
        };
        let per_quad = 16 * nattr;
        let mut data: Vec<f32> =
            Vec::with_capacity(counts[span.clone()].iter().sum::<usize>() * per_quad);
        {
            let scene = self.scene.borrow();
            for m in &run.members[span.clone()] {
                m.dirty.set(false);
                let quads = m.quads.borrow();
                match (mode, &m.text) {
                    (Mode::Mixed, None) => crate::run::shape_vertices(&quads, &mut data),
                    (Mode::Mixed, Some(t)) => {
                        let l = scene
                            .as_ref()
                            .and_then(|s| self.text_layer(s, &m.name))
                            .expect("a run's text is a text layer of the scene");
                        crate::run::glyph_vertices(l, &quads, t.size.get(), &mut data);
                    }
                    _ => data.extend_from_slice(&quads),
                }
            }
        }
        #[cfg(test)]
        self.run_floats.set(self.run_floats.get() + data.len());
        if all {
            self.batch_quads(run.batch, what, &data, nattr)?;
            *run.sent.borrow_mut() = counts;
            return Ok(());
        }
        if data.is_empty() {
            return Ok(());
        }
        let first: usize = counts[..span.start].iter().sum();
        let ok = unsafe {
            ffi::ghee_mp_set_batch_quads_at(
                self.handle,
                run.batch,
                c_int::try_from(first)?,
                data.as_ptr(),
                c_int::try_from(data.len() / per_quad)?,
            )
        };
        if !ok {
            bail!(
                "{}",
                or_else(self.last_error(), &format!("{what} refused its quads"))
            );
        }
        Ok(())
    }

    /// Insert a batch into drawn pass `pass` after batch `after`, -1 for
    /// ahead of the pass's batches, with `program`, or none, and `blend`.
    fn insert_batch(
        &self,
        pass: c_int,
        after: c_int,
        name: &str,
        program: Option<crate::layer::Program>,
        blend: Blend,
    ) -> Result<c_int> {
        let cname = CString::new(name)?;
        let sources = match &program {
            Some((v, f)) => Some((CString::new(v.as_str())?, CString::new(f.as_str())?)),
            None => None,
        };
        let (vs, fs) = match &sources {
            Some((v, f)) => (v.as_ptr(), f.as_ptr()),
            None => (ptr::null(), ptr::null()),
        };
        let b = unsafe {
            ffi::ghee_mp_insert_batch(
                self.handle,
                pass,
                after,
                cname.as_ptr(),
                vs,
                fs,
                blend.engine(),
            )
        };
        if b < 0 {
            bail!("{name}: {}", or_else(self.last_error(), "refused"));
        }
        Ok(b)
    }

    /// A drawn layer named `name`, with no batch yet.
    fn undrawn(name: &str) -> LayerBatches {
        LayerBatches {
            batch: -1,
            colour: -1,
            coloured: std::cell::Cell::new(false),
            boxes: -1,
            name: name.to_string(),
            phases: std::cell::RefCell::new(Vec::new()),
            shared: std::cell::Cell::new(false),
            shown: std::cell::RefCell::new(None),
        }
    }

    /// Draw shape layer `l`, new at `index` of drawn pass `pass`, in the
    /// runs of the pass.
    fn add_shape(&mut self, pass: &str, index: usize, l: &ShapeLayer) -> Result<()> {
        let size = self.pass_size(pass);
        self.put_drawn(pass, index, Self::undrawn(&l.name));
        self.rerun(pass, vec![Member::shape(l, size)])
    }

    /// Draw image layer `l`, new at `index` of drawn pass `pass`.
    fn add_image(
        &mut self,
        pass_name: &str,
        pass: c_int,
        index: usize,
        l: &ImageLayer,
    ) -> Result<()> {
        let after = self.after(pass_name, index);
        let name = format!("layer '{}'", l.name);
        let batch = self.insert_batch(pass, after, &name, Some(l.shaders()), l.blend)?;
        let dl = LayerBatches {
            batch,
            ..Self::undrawn(&l.name)
        };
        for (u, ty, value) in Layer::Image(l.clone()).uniforms() {
            self.set_layer_uniform(&dl, &u, ty, value)?;
        }
        self.bind_image(batch, &self.image_asset(l)?)?;
        self.batch_quads(batch, &name, &IMAGE_QUAD, 1)?;
        self.put_drawn(pass_name, index, dl);
        self.rerun(pass_name, Vec::new())
    }

    /// Draw text layer `l`, new at `index` of drawn pass `pass`: its glyphs
    /// in the runs of the pass, or after the layer before it, its colour
    /// glyphs and its error boxes among those of the pass in the order of
    /// the layers, its atlas, and its words.
    fn add_text(
        &mut self,
        pass_name: &str,
        pass: c_int,
        index: usize,
        l: &TextLayer,
    ) -> Result<()> {
        let size = self.pass_size(pass_name);
        if let Some(b) = l.place(&l.text).ok().and_then(|laid| laid.built) {
            let px = l.font_px(size, b.size);
            self.px_em
                .insert(l.name.clone(), TextLayer::atlas_px_em(px));
        }
        let shaders = l.shaders(self.detail(l));
        let name = format!("layer '{}'", l.name);
        match joins_runs(&Layer::Text(l.clone())) {
            true => {
                let own = Hold::keep(self.handle, &name, &shaders.glyphs)?;
                self.put_drawn(pass_name, index, Self::undrawn(&l.name));
                self.rerun(pass_name, vec![Member::text(l, Class::Joins, own)])?;
            }
            false => {
                let after = self.after(pass_name, index);
                let batch = self.insert_batch(pass, after, &name, Some(shaders.glyphs), l.blend)?;
                let dl = LayerBatches {
                    batch,
                    ..Self::undrawn(&l.name)
                };
                self.put_drawn(pass_name, index, dl);
                self.rerun(pass_name, Vec::new())?;
            }
        }
        let batch = self.drawn(&l.name).expect("the layer was just drawn").batch;
        let names = self.pass_layers(pass_name);
        let text_before = |k: usize| {
            names[..k]
                .iter()
                .rev()
                .filter_map(|n| self.drawn(n))
                .find(|d| d.colour >= 0)
        };
        // Every colour batch of a pass draws after every layer of it, and
        // every error-box batch after them, each in the order of the layers.
        let colour_after = match text_before(index) {
            Some(d) => d.colour,
            None => names
                .iter()
                .rev()
                .find_map(|n| self.drawn(n).map(|d| d.batch))
                .unwrap_or(batch),
        };
        let colour = self.insert_batch(
            pass,
            colour_after,
            &format!("{name} colour glyphs"),
            None,
            l.blend,
        )?;
        let boxes_after = match text_before(index) {
            Some(d) => d.boxes,
            None => names[index + 1..]
                .iter()
                .rev()
                .filter_map(|n| self.drawn(n))
                .find(|d| d.colour >= 0)
                .map_or(colour, |d| d.colour),
        };
        let boxes = self.insert_batch(
            pass,
            boxes_after,
            &format!("{name} error boxes"),
            Some(shaders.boxes),
            Blend::Normal,
        )?;
        let k = self.places().drawn[&l.name];
        self.layers[k].colour = colour;
        self.layers[k].boxes = boxes;
        let dl = &self.layers[k];
        for (u, ty, value) in l.uniforms() {
            self.set_layer_uniform(dl, &u, ty, value)?;
        }
        if l.render == Render::Slug {
            self.curves.insert(l.name.clone());
        }
        let layer = self
            .scene
            .borrow()
            .as_ref()
            .and_then(|s| s.text_layer(&l.name).cloned())
            .expect("the layer joined the scene");
        self.add_atlas(&mut self.atlases.borrow_mut(), &mut Vec::new(), &layer)?;
        self.lay_text(&l.name, &l.text)?;
        Ok(())
    }

    /// Give the run of shape layer `l` its quads as `l` draws them, for the
    /// next settle().
    fn reshape(&self, l: &ShapeLayer) {
        let Some((r, k)) = self.member_of(&l.name) else {
            return;
        };
        let run = &self.runs[r];
        let m = &run.members[k];
        let mut quads = Vec::new();
        l.quads(run.size, &mut quads);
        *m.quads.borrow_mut() = quads;
        m.dirty.set(true);
        m.class.set(shape_class(l, run.size));
        self.unsettled.set(true);
        if m.class.get() != m.placed {
            self.reclassed.set(true);
        }
    }

    /// Build the atlas of the text layer `name`, which shows for the first
    /// time since the setup, and lay its words out again on it. A layer
    /// that holds an atlas, or that draws none, keeps what it has.
    fn first_shown(&self, name: &str) -> Result<()> {
        if self.atlas_of(name).is_some() {
            return Ok(());
        }
        let words = {
            let scene = self.scene.borrow();
            let Some(scene) = scene.as_ref() else {
                return Ok(());
            };
            let Some(layer) = self.text_layer(scene, name) else {
                return Ok(());
            };
            self.add_atlas(&mut self.atlases.borrow_mut(), &mut Vec::new(), layer)?;
            layer.text.clone()
        };
        if self.atlas_of(name).is_none() {
            return Ok(());
        }
        self.words.borrow_mut().remove(name);
        self.lay_text(name, &words)?;
        Ok(())
    }

    /// The index in `atlases` of the atlas that layer `name` reads, None for a
    /// layer that reads none.
    fn atlas_of(&self, name: &str) -> Option<usize> {
        let tex = |t: &str| {
            t.strip_prefix(crate::scene::PREFIX)
                .and_then(|t| t.strip_suffix("_atlas"))
                == Some(name)
        };
        self.atlases
            .borrow()
            .iter()
            .position(|s| s.textures.iter().any(|t| tex(&t.0)))
    }

    /// Draw `laid`, new words of `layer`, from now on, and log their problems.
    /// Words that did not lay out leave the last words drawn.
    fn show(&self, layer: &TextLayer, laid: Layout) -> Result<()> {
        let name = &layer.name;
        let Layout { built, mut faults } = laid;
        let Some(built) = built else {
            self.text_faults(layer, &faults, false);
            return Ok(());
        };
        // The quads of the layer's render, and under MTSDF an error box for
        // each glyph that the atlas holds no field for. An MTSDF layer with no
        // atlas is hidden or has no font with outlines left, and still draws
        // its colour glyphs.
        let mut boxes = built.boxes.floats.clone();
        let quads = match layer.render {
            Render::Slug => layer.copies(&built.quads.floats),
            Render::Mtsdf => {
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
                        layer.copies(&quads.floats)
                    }
                    _ => Vec::new(),
                }
            }
        };
        let colour = self.colour_quads(layer, &built.colour, &mut faults)?;
        boxes.extend(&colour.1);
        self.text_faults(layer, &faults, true);
        // A layout with no texture of curves leaves the texture alone. A
        // run binds the texture of its texts as it settles.
        let drawn = self.drawn(name);
        let member = self.member_of(name);
        if built.shared {
            self.send_glyphs()?;
            let rows = self.glyphs.borrow().rows();
            // The engine binds an asset that it holds, and words with no
            // glyph draw no quad that reads one.
            if let Some(d) = drawn.filter(|d| member.is_none() && !d.shared.get() && rows > 0) {
                self.bind_asset(d.batch, &text::internal(GLSL_NAME, "_curves"), GLYPHS)?;
                d.shared.set(true);
                for (suffix, v) in text::Glyphs::uniforms(rows) {
                    let own = text::internal(GLSL_NAME, suffix);
                    self.set_layer_uniform(d, &own, "float", Value::from(v))?;
                }
            }
        } else if built.width > 0 {
            let curves = text::internal(name, "_curves");
            self.set_asset(
                &curves,
                built.width,
                built.height,
                4,
                false,
                AssetData::U8(&built.texels),
            )?;
            self.scene_asset(&curves);
            if let Some(d) = drawn.filter(|_| member.is_none()) {
                self.bind_asset(d.batch, &text::internal(GLSL_NAME, "_curves"), &curves)?;
                d.shared.set(false);
            }
        }
        // The names of a layer reach the passes alone: a layer's batches
        // read their own under GLSL_NAME. A pass or a batch that holds
        // these values already takes none.
        let shown: Shown = (built.uniforms, built.size);
        let size = Value::from(built.size.to_vec());
        if self.pass_shown.borrow().get(name) != Some(&shown) {
            for (suffix, v) in &shown.0 {
                let uniform = text::internal(name, suffix);
                for (_, idx) in &self.mp {
                    self.set_pass_uniform(*idx, &uniform, "float", Value::from(*v))?;
                }
            }
            for (_, idx) in &self.mp {
                self.set_pass_uniform(*idx, &format!("{name}_size"), "vec2", size.clone())?;
            }
            self.pass_shown
                .borrow_mut()
                .insert(name.clone(), shown.clone());
        }
        let Some(d) = drawn else {
            return Ok(());
        };
        if d.shown.borrow().as_ref() != Some(&shown) {
            for (suffix, v) in &shown.0 {
                let own = text::internal(GLSL_NAME, suffix);
                self.set_layer_uniform(d, &own, "float", Value::from(*v))?;
            }
            let own = text::internal(GLSL_NAME, "_size");
            self.set_layer_uniform(d, &own, "vec2", size)?;
            *d.shown.borrow_mut() = Some(shown);
        }
        for (batch, floats, part) in [
            (d.batch, &quads, ""),
            (d.colour, &colour.0, " colour glyphs"),
            (d.boxes, &boxes, " error boxes"),
        ] {
            if batch >= 0 && !(member.is_some() && batch == d.batch) {
                self.text_quads(batch, name, part, floats)?;
            }
        }
        if let Some((r, k)) = member {
            let m = &self.runs[r].members[k];
            let t = m.text.as_ref().expect("a text's member");
            t.size.set(built.size);
            t.shared.set(built.shared);
            if !built.shared && built.width > 0 {
                t.texture.set(true);
                t.fresh.set(true);
            }
            *m.quads.borrow_mut() = quads;
            m.dirty.set(true);
            m.reclass.set(true);
            self.unsettled.set(true);
            self.reclassed.set(true);
        }
        Ok(())
    }

    /// Hand the engine batch `batch` of text layer `name` its quads, the
    /// batch named in an error as the layer and `part`. A batch that holds
    /// as many quads takes them in place, and an empty batch that stays
    /// empty takes no call.
    fn text_quads(&self, batch: c_int, name: &str, part: &str, floats: &[f32]) -> Result<()> {
        let nattr = text::QUAD_FLOATS / 4;
        let quads = c_int::try_from(floats.len() / (16 * nattr))?;
        let held = unsafe { ffi::ghee_mp_batch_quads(self.handle, batch, nattr as c_int) };
        if held == quads && quads == 0 {
            return Ok(());
        }
        if held == quads
            && floats.len().is_multiple_of(16 * nattr)
            && unsafe {
                ffi::ghee_mp_set_batch_quads_at(self.handle, batch, 0, floats.as_ptr(), quads)
            }
        {
            return Ok(());
        }
        self.batch_quads(batch, &format!("layer '{name}'{part}"), floats, nattr)
    }

    /// The quads of the colour glyphs `glyphs` of `layer`, for its colour
    /// batch, and an error box for each glyph whose bitmap failed, with a
    /// fault in `faults`. The first colour glyph of the layer makes its colour
    /// atlas, and every new one decodes into it and uploads.
    fn colour_quads(
        &self,
        layer: &TextLayer,
        glyphs: &[text::Placement],
        faults: &mut Vec<Fault>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let name = &layer.name;
        if glyphs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let Some(drawn) = self.drawn(name).filter(|d| d.colour >= 0) else {
            return Ok((Vec::new(), Vec::new()));
        };
        if !drawn.coloured.get() {
            self.build_colour(layer, drawn)?;
        }
        let mut colours = self.colours.borrow_mut();
        let slot = colours.entry(name.clone()).or_default();
        let mut cells: Vec<crate::colour::Rect> = Vec::new();
        for p in glyphs {
            let id = text::atlas_id(p.font, p.gid);
            if slot.atlas.cells.contains_key(&id) || slot.failed.contains_key(&id) {
                continue;
            }
            let font = layer.font(p.font);
            let face = font
                .as_deref()
                .and_then(|d| ttf_parser::Face::parse(d, 0).ok());
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
                self.bind_asset(drawn.colour, &text::internal(GLSL_NAME, "_emoji"), &tex)?;
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

    /// Compile the program of the colour batch of `layer`, drawn as `dl`,
    /// and hand it the layer's uniforms.
    fn build_colour(&self, layer: &TextLayer, dl: &LayerBatches) -> Result<()> {
        let (vs, fs) = layer.shaders(self.detail(layer)).colour;
        let (cvs, cfs) = (CString::new(vs)?, CString::new(fs)?);
        if !unsafe {
            ffi::ghee_mp_set_batch_program(self.handle, dl.colour, cvs.as_ptr(), cfs.as_ptr())
        } {
            let err = or_else(self.last_error(), "compile error");
            bail!("layer '{}' colour glyphs: {err}", layer.name);
        }
        dl.coloured.set(true);
        Ok(())
    }

    /// The atlas detail of `layer`, as setup() worked it out.
    fn detail(&self, layer: &TextLayer) -> Detail {
        let px_em = self.px_em.get(&layer.name).copied();
        layer.detail(px_em.unwrap_or(atlas::PX_EM))
    }

    /// The texels per em of the atlas of each layer of `drawn`, by its
    /// name: the step at or above its font_size, or the size at which its
    /// words fit its pass, laid out on the CPU. A layer whose words lay out
    /// in no font takes atlas::PX_EM.
    fn atlas_sizes(&self, drawn: &[Pass]) -> HashMap<String, u32> {
        let mut out = HashMap::new();
        let (sw, sh) = self.scene_size();
        let frame = (i64::from(sw), i64::from(sh));
        for p in drawn {
            let (w, h) = p.size_in(frame).unwrap_or(frame);
            for l in p.layers.iter().filter_map(Layer::as_text) {
                let block = l.place(&l.text).ok().and_then(|laid| laid.built);
                let Some(b) = block else { continue };
                let px = l.font_px([w as f64, h as f64], b.size);
                out.insert(l.name.clone(), TextLayer::atlas_px_em(px));
            }
        }
        out
    }

    /// Set up the atlases of the layers of `drawn`: one generator per font,
    /// px/em and range among the MTSDF layers that show. A hidden layer takes
    /// none. A generator that the last scene used carries on, and the others
    /// stop.
    fn set_atlases(&self, drawn: &[Pass]) -> Result<()> {
        let mut old = std::mem::take(&mut *self.atlases.borrow_mut());
        let mut slots: Vec<AtlasSlot> = Vec::new();
        for l in drawn
            .iter()
            .flat_map(|p| &p.layers)
            .filter_map(Layer::as_text)
        {
            self.add_atlas(&mut slots, &mut old, l)?;
        }
        drop(old);
        *self.atlases.borrow_mut() = slots;
        Ok(())
    }

    /// Hand `layer` its atlas in `slots`, when it is an MTSDF layer that
    /// shows and has a font with outlines: the slot of its font, px/em and
    /// range, else the generator of `old` that matches, else a new one.
    fn add_atlas(
        &self,
        slots: &mut Vec<AtlasSlot>,
        old: &mut Vec<AtlasSlot>,
        l: &TextLayer,
    ) -> Result<()> {
        let Some(fonts) = &l.data else {
            return Ok(());
        };
        if l.render != Render::Mtsdf || !l.visible || !l.has_outlines() {
            return Ok(());
        }
        let Detail { px_em, range } = self.detail(l);
        let same = |s: &AtlasSlot| s.px_em == px_em && s.range == range && s.fonts == *fonts;
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
                        px_em,
                        range,
                        gen: atlas::Generator::new(
                            (self.atlas_font)(fonts),
                            atlas::Config {
                                px_em,
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
            .push((text::internal(&l.name, "_atlas"), true));
        Ok(())
    }

    /// Advance the phase of every effect that moves by one by its pace times
    /// the time since the last render: the step of the pinned time, from 0
    /// at the setup, else the time step of set_dt().
    fn advance(&self) -> Result<()> {
        let step = match self.pinned.get() {
            Some(t) => t - self.drawn_at.replace(t),
            None => self.dt.get(),
        };
        let scene = self.scene.borrow();
        let Some(scene) = scene.as_ref() else {
            return Ok(());
        };
        for pass in &scene.passes {
            let size = self.pass_size(&pass.name);
            let texts = pass.layers.iter().filter_map(Layer::as_text);
            for layer in texts.filter(|l| !l.effects.is_empty()) {
                let Some(dl) = self.drawn(&layer.name) else {
                    continue;
                };
                let block = dl.shown.borrow().as_ref().map(|s| s.1);
                let mut phases = dl.phases.borrow_mut();
                for (kind, uniform, pace) in layer.paces() {
                    let i = match phases.iter().position(|(k, _)| *k == kind) {
                        Some(i) => i,
                        None => {
                            phases.push((kind, 0.0));
                            phases.len() - 1
                        }
                    };
                    let mut phase = phases[i].1 + pace * step;
                    // A phase stays under its cycle, so its float uniform
                    // keeps its precision however long the scene runs.
                    if let Some(cycle) = layer.cycle(kind, size, block) {
                        phase = phase.rem_euclid(cycle);
                    }
                    phases[i].1 = phase;
                    self.set_layer_uniform(dl, &uniform, "float", Value::from(phase))?;
                }
            }
        }
        Ok(())
    }

    /// Hand the engine what the atlases gained, then draw the held words
    /// whose glyphs are all in. With `wait`, or with the time pinned, every
    /// held text waits for its glyphs first.
    fn tick_texts(&self, wait: bool) -> Result<()> {
        if self.atlases.borrow().is_empty() {
            return Ok(());
        }
        let wait = wait || self.pinned.get().is_some();
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
            let scene = self.scene.borrow();
            if let Some(layer) = scene.as_ref().and_then(|s| self.text_layer(s, &name)) {
                self.show(layer, h.laid)?;
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
                    let layer = tex
                        .strip_prefix(crate::scene::PREFIX)
                        .and_then(|t| t.strip_suffix("_atlas"));
                    if let Some(d) = self.layers.iter().find(|d| Some(d.name.as_str()) == layer) {
                        self.bind_asset(d.batch, &text::internal(GLSL_NAME, "_atlas"), tex)?;
                    }
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

    /// Log each new cause among the `faults` of a layout of `layer`, and hold
    /// them for text_errors(). Words that did not lay out, `drawn` false,
    /// leave the problems of the words still on screen.
    fn text_faults(&self, layer: &TextLayer, faults: &[Fault], drawn: bool) {
        let name = &layer.name;
        let mut reported = self.reported.borrow_mut();
        // True the first time since the setup that a cause comes up.
        let mut fresh = |cause: String| reported.insert(cause);
        let no_outline = |f: usize, cs: &[char]| {
            let list: Vec<String> = cs.iter().map(|c| format!("{c:?}")).collect();
            format!(
                "layer '{name}': {} has no outline for {}",
                layer.font_name(f),
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
                // The reason names the asset, which every layer in the font
                // shares.
                Fault::Font(why) => {
                    if fresh(format!("font\0{why}")) {
                        crate::report::error(why);
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
                        crate::report::error(no_outline(*f, &new));
                    }
                    no_outline(*f, chars)
                }
                Fault::Glyph(ch, why) => format!("layer '{name}': {ch:?} {why}"),
                Fault::Layout(why) => format!("layer '{name}': {why}"),
            };
            if !matches!(f, Fault::Font(_) | Fault::NoOutline(..)) && fresh(line.clone()) {
                crate::report::error(&line);
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

    /// The problems of the scene's text layers, a line each: a font that
    /// failed to load, characters drawn as error boxes, words that do not
    /// fit.
    pub fn text_errors(&self) -> Vec<String> {
        let all = self.text_errors.borrow();
        let scene = self.scene.borrow();
        let Some(scene) = scene.as_ref() else {
            return Vec::new();
        };
        let mut out: Vec<String> = Vec::new();
        for l in scene.text_layers() {
            if let Some(e) = all.get(&l.name) {
                for line in e.drawn.iter().chain(&e.refused) {
                    // Layers in one font share its failure.
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

    /// Bind engine asset `asset` to the batch `batch` of an image layer,
    /// which samples it as its image.
    fn bind_image(&self, batch: c_int, asset: &str) -> Result<()> {
        self.bind_asset(batch, &text::internal(GLSL_NAME, "_image"), asset)
    }

    /// Bind engine asset `asset` to the batch `batch`, which samples it as
    /// `<name>_tex` and reads its size as `<name>_size`.
    fn bind_asset(&self, batch: c_int, name: &str, asset: &str) -> Result<()> {
        let (cname, casset) = (CString::new(name)?, CString::new(asset)?);
        if !unsafe { ffi::ghee_mp_bind_asset(self.handle, batch, cname.as_ptr(), casset.as_ptr()) }
        {
            bail!(
                "{}",
                or_else(self.last_error(), &format!("{asset} refused its batch"))
            );
        }
        Ok(())
    }

    /// Set the uniform `name` of `ptype` to `value` on batch `batch` alone.
    fn set_batch_uniform(&self, batch: c_int, name: &str, ptype: &str, value: Value) -> Result<()> {
        let mut cp = create_c_parameter(ptype, &value)?;
        cp.set_c_name(name)?;
        if !unsafe { ffi::ghee_mp_set_batch_uniform(self.handle, batch, cp.as_ptr()) } {
            bail!(
                "{}",
                or_else(self.last_error(), &format!("batch {batch} refused {name}"))
            );
        }
        Ok(())
    }

    /// Set the uniform `name` of `ptype` to `value` on each batch of the
    /// drawn layer `dl`. A text of a run of more members holds its uniforms
    /// in its vertices, which its run takes again at the next settle().
    fn set_layer_uniform(
        &self,
        dl: &LayerBatches,
        name: &str,
        ptype: &str,
        value: Value,
    ) -> Result<()> {
        let mut own = true;
        if let Some((r, k)) = self.member_of(&dl.name) {
            let run = &self.runs[r];
            let m = &run.members[k];
            m.reclass.set(true);
            self.reclassed.set(true);
            if run.installed.get() != Some(Mode::Alone) {
                own = false;
                m.dirty.set(true);
                self.unsettled.set(true);
            }
        }
        for batch in [dl.batch, dl.colour, dl.boxes] {
            if batch >= 0 && (own || batch != dl.batch) {
                self.set_batch_uniform(batch, name, ptype, value.clone())?;
            }
        }
        Ok(())
    }

    /// Record `name` among the engine assets uploaded for the scene.
    fn scene_asset(&self, name: &str) {
        let mut names = self.scene_assets.borrow_mut();
        if !names.iter().any(|n| n == name) {
            names.push(name.to_string());
        }
    }

    /// Stage batch `name` of the shaders `program` into pending pass `pass`,
    /// blended by `blend`. Returns the engine's index of the batch, -1 when
    /// the engine refuses it.
    fn add_batch(
        &self,
        pass: c_int,
        name: &str,
        program: &crate::layer::Program,
        blend: Blend,
    ) -> Result<c_int> {
        let (cname, cvs, cfs) = (
            self.staged_cstring(name)?,
            self.staged_cstring(&program.0)?,
            self.staged_cstring(&program.1)?,
        );
        Ok(unsafe {
            ffi::ghee_mp_add_batch(
                self.handle,
                pass,
                cname.as_ptr(),
                cvs.as_ptr(),
                cfs.as_ptr(),
                blend.engine(),
            )
        })
    }

    /// Stage batch `name` into pending pass `pass` with no program yet,
    /// blended by `blend`. Returns the engine's index of the batch, -1 when
    /// the engine refuses it.
    fn add_batch_unbuilt(&self, pass: c_int, name: &str, blend: Blend) -> Result<c_int> {
        let cname = self.staged_cstring(name)?;
        Ok(unsafe {
            ffi::ghee_mp_add_batch(
                self.handle,
                pass,
                cname.as_ptr(),
                ptr::null(),
                ptr::null(),
                blend.engine(),
            )
        })
    }

    /// Drop the staged scene after the engine refused batch `name` of the
    /// pass at index `pass` of the drawn passes, and return the error.
    fn batch_refused(&mut self, name: &str, pass: usize) -> Result<String> {
        let err = or_else(self.last_error(), "compile error");
        self.staged_refused(format!("{name}: {err}"), pass)
    }

    /// Drop the staged scene after the engine refused a program of the pass
    /// at index `pass` of the drawn passes, and return `error`, which names
    /// the program and holds the engine's error.
    fn staged_refused(&mut self, error: String, pass: usize) -> Result<String> {
        self.compile_error = unsafe { ffi::ghee_last_error_is_compile(self.handle) };
        unsafe { ffi::ghee_mp_abort(self.handle) };
        self.error = error;
        self.error_pass = Some(ErrorPass::Index(pass));
        crate::report::progress_done();
        Ok(self.error.clone())
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

    /// Pin the `iTime` uniform to a fixed value, for a reproducible render.
    /// The default is the monotonic clock since the engine start.
    /// None returns the uniform to that clock.
    pub fn set_time(&self, t: Option<f64>) {
        let t = t.map(|t| match self.time_base.get() {
            (_, true) => {
                self.time_base.set((t, false));
                0.0
            }
            (base, false) => t - base,
        });
        self.pinned.set(t);
        match t {
            None => unsafe { ffi::ghee_clear_time(self.handle) },
            Some(t) => unsafe { ffi::ghee_set_time(self.handle, t as f32) },
        }
    }

    /// Restart the clock: the next frame draws as the first one does, as
    /// iFrame 0 at iTime 0 with iTimeDelta at 1/60 until set_dt() sets
    /// another. The monotonic clock counts from that frame. With the time
    /// pinned, the next pin reads 0, and later pins count from it.
    pub fn restart_clock(&self) {
        unsafe { ffi::ghee_restart_clock(self.handle) };
        self.time_base.set((0.0, self.pinned.get().is_some()));
        self.drawn_at.set(0.0);
        self.set_dt(DT);
        self.restarted.set(true);
    }

    /// Set the `iTimeDelta` uniform, this frame's advance in seconds of scene
    /// time. iTimeDelta rides the draw like `iTime`: one store here, one
    /// upload per pass.
    /// The default is 1/60 until a caller feeds a measured period.
    pub fn set_dt(&self, value: f64) {
        self.dt.set(value);
        unsafe { ffi::ghee_set_dt(self.handle, value as f32) };
    }

    /// The texture units one program of a pass binds at most: the pass's
    /// channels and arrays, then the assets and the texts that the program
    /// samples. It is the GPU's own count, capped at 16.
    pub fn max_texture_units(&self) -> i32 {
        unsafe { ffi::ghee_max_texture_units(self.handle) }
    }

    /// The most pixels that a texture of this GPU holds on a side,
    /// GL_MAX_TEXTURE_SIZE.
    pub fn max_texture_size(&self) -> i64 {
        i64::from(unsafe { ffi::ghee_max_texture_size(self.handle) })
    }

    /// The error for `what`, an array or an image of `w` x `h` pixels, when a
    /// side is past max_texture_size(). GL refuses a texture past it, and a
    /// pass then samples black.
    fn past_max(&self, what: &str, w: i64, h: i64) -> Option<String> {
        let max = self.max_texture_size();
        (w.max(h) > max)
            .then(|| format!("{what} is {w}x{h}, over this GPU's {max} pixel cap of a side"))
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

    /// Turn the frame `degrees` clockwise, 0, 90, 180 or 270, on its way to
    /// every output: the readback, the display and the H264 encoder. width
    /// and height stay the frame's, and at 90 and 270 the scene renders at
    /// that size turned, as scene_size() gives it. Call it before the first
    /// setup().
    pub fn set_rotate(&mut self, degrees: i64) -> Result<()> {
        let turns = quarter_turns(degrees)?;
        if !unsafe { ffi::ghee_set_rotate(self.handle, turns) } {
            bail!("{}", or_else(self.last_error(), "rotate refused"));
        }
        self.turns = turns;
        Ok(())
    }

    /// The backlight level of the display, 0 to 1. None outside display
    /// mode, or for a display with no backlight device.
    pub fn brightness(&self) -> Result<Option<f64>> {
        crate::panel::brightness(Path::new(crate::panel::SYS), self.card.as_deref())
    }

    /// Set the backlight of the display, 0 to 1, through the backlight
    /// device that the kernel registered under the card's device, and give
    /// back the level that it then reads. close() puts back the level from
    /// before the first set.
    pub fn set_brightness(&self, level: f64) -> Result<f64> {
        crate::panel::set_brightness(
            Path::new(crate::panel::SYS),
            self.card.as_deref(),
            level,
            &mut self.backlight.borrow_mut(),
        )
    }

    /// The size that the scene renders at: the frame's size, turned at 90
    /// and 270 degrees.
    pub fn scene_size(&self) -> (i32, i32) {
        match self.turns % 2 {
            1 => (self.height, self.width),
            _ => (self.width, self.height),
        }
    }

    /// Hand `read` the scene that the renderer draws: the scene of its last
    /// setup, with the layers and the live values set since. None before the
    /// first setup.
    pub fn with_scene<R>(&self, read: impl FnOnce(&Scene) -> R) -> Option<R> {
        self.scene.borrow().as_ref().map(read)
    }

    /// `read` of the scene and its layer named `name`, None when the scene
    /// has no such layer.
    pub fn with_layer<R>(&self, name: &str, read: impl FnOnce(&Scene, &Layer) -> R) -> Option<R> {
        let scene = self.scene.borrow();
        let scene = scene.as_ref()?;
        let (p, i) = *self.places().scene.get(name)?;
        Some(read(scene, &scene.passes[p].layers[i]))
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

    /// The programs that the renderer linked since it was created: each
    /// pass's, each distinct batch's, and its own.
    pub fn compiled(&self) -> i32 {
        unsafe { ffi::ghee_compiled(self.handle) }
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
        #[cfg(test)]
        {
            self.batches = Vec::new();
        }
        self.layers = Vec::new();
        self.runs = Vec::new();
        self.moved();
        self.feeds = Vec::new();
        self.held.borrow_mut().clear();
        self.words.borrow_mut().clear();
        self.atlases.borrow_mut().clear();
        self.colours.borrow_mut().clear();
        *self.glyphs.borrow_mut() = text::Glyphs::default();
        self.pass_shown.borrow_mut().clear();
    }

    pub fn close(&mut self) {
        self.backlight.get_mut().take();
        // The runs let go of their texts' programs while the engine lives.
        self.runs.clear();
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
    use crate::layer::Rgb;

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
        scene_of(json!([{"name": "main", "fragment": FLAT}]))
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
    #[cfg_attr(not(ghee_display), ignore = "needs a connected display")]
    fn probe_display_device_names_a_probed_card_and_its_size() {
        let _g = lock();
        env::remove_var("GHEE_DRI");
        let (dev, w, h) = probe_display_device(None).expect("no connected display");
        // The connectors do not always sit on card0. A Pi 5 puts them on card1.
        assert!(probe_dri_cards().contains(&dev));
        assert!(w > 0 && h > 0);
        assert_eq!(probe_display(None), Some((w, h)));
    }

    #[test]
    #[cfg_attr(
        any(ghee_no_gpu, not(ghee_display)),
        ignore = "needs a GL device and a connected display"
    )]
    fn display_create_fails_once_without_drm_master() {
        let _g = lock();
        assert!(
            !fs::metadata("/proc/self").is_ok_and(|m| m.uid() == 0),
            "runs as a user: as root this test takes the session's card"
        );
        let (dev, w, h) = probe_display_device(None).expect("no connected display");
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
        assert!(
            !before_the_master_attempt.iter().any(|s| msg.contains(s)),
            "{dev} failed before it asked for the DRM master: {msg}"
        );
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
    fn a_compile_error_names_its_line_in_the_log_of_each_driver() {
        // Mesa's row is the form of the tests above. NVIDIA's is the log of
        // an undefined name on NVIDIA 615.71.09.
        for (log, line) in [
            ("0:6(3): error: no function nope", 6),
            ("0(18) : error C1503: undefined variable \"nope\"", 18),
        ] {
            assert_eq!(error_line_number(log), Some(line), "{log}");
        }
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
            {"name": "a", "fragment": FLAT},
            {"name": "b", "channels": [{"buffer": "a"}, {"buffer": "typo"}], "fragment": FLAT},
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
            {"name": "a", "channels": [{"buffer": null}], "fragment": FLAT},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn setup_rejects_a_scene_in_a_fixed_order() {
        let (_g, mut r) = gpu();
        let clash = json!([
            {"name": "a", "fragment": FLAT, "arrays": [{"name": "v", "size": 4}]},
            {"name": "b", "fragment": FLAT, "arrays": [{"name": "v", "size": [2, 2]}]},
        ]);
        assert_eq!(
            r.setup(&scene_of(clash)).unwrap(),
            "array 'v' declared at conflicting sizes (4 and (2, 2))"
        );
        // The size clash outranks the bad channel below it.
        let both = json!([
            {"name": "a", "fragment": FLAT, "arrays": [{"name": "v", "size": 4}]},
            {"name": "b", "fragment": FLAT, "channels": [{"buffer": "typo"}],
             "arrays": [{"name": "v", "size": [2, 2]}]},
        ]);
        assert!(r.setup(&scene_of(both)).unwrap().starts_with("array 'v'"));
    }

    /// Pass `a` is 8x8, or 8x6 at a scale of 1/8, and paints red right of
    /// x = 4. Read at x = 16, it is black when `a` keeps its size and a later
    /// pass stretches it over the frame, and red when `a` renders at the
    /// screen size.
    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_sized_or_scaled_pass_renders_at_the_screen_size_unless_a_later_pass_reads_it() {
        let (_g, mut r) = gpu();
        let a = json!({"name": "a", "size": [8, 8], "fragment":
            "void main(){ gl_FragColor = vec4(gl_FragCoord.x > 4.0 ? 1.0 : 0.0, 0.0, 0.0, 1.0); }"});
        let through = "void main(){ gl_FragColor = texture2D(iChannel0, uv); }";
        let b = |visible: bool| {
            json!({"name": "b", "fragment": through, "channels": [{"buffer": "a"}],
                   "visible": visible})
        };
        let mut scaled = a.clone();
        scaled["scale"] = json!(0.125);
        scaled.as_object_mut().unwrap().remove("size");
        for (passes, red) in [
            (json!([scaled]), 255.0),
            (json!([scaled, b(true)]), 0.0),
            (json!([a]), 255.0),
            (json!([a, b(true)]), 0.0),
            (json!([a, b(false)]), 255.0),
        ] {
            assert_eq!(r.setup(&scene_of(passes.clone())).unwrap(), "", "{passes}");
            r.set_time(Some(0.0));
            let f = r.render(None).unwrap().to_vec();
            assert_eq!(px(&f, W, 16, H / 2, 0), red, "{passes}");
            assert_eq!(px(&f, W, 2, H / 2, 0), 0.0, "{passes}");
            assert_eq!(px(&f, W, W - 1, H / 2, 0), 255.0, "{passes}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn unbound_channel_reads_black() {
        // 'leak' leaves a white texture bound on unit 0.
        // Channel 0 of 'main' is unbound, and it reads black.
        let (_g, mut r) = gpu();
        let sample = "void main(){ gl_FragColor = texture2D(iChannel0, uv); }";
        let sc = scene_of(json!([
            {"name": "bright", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"},
            {"name": "leak", "channels": [{"buffer": "bright"}], "fragment": sample},
            {"name": "main", "channels": [{"buffer": null}], "fragment": sample},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let f = r.render(None).unwrap().to_vec();
        for c in 0..3 {
            assert!(channel(&f, c).iter().fold(0.0, |a: f64, b| a.max(*b)) <= 2.0);
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_hidden_pass_draws_the_pass_before_it_and_the_last_visible_pass_shows() {
        // a is red, b adds green to a, and c halves the red of b and adds
        // blue. A hidden pass copies the pass before it, a hidden first pass
        // is black, and the frame is the last visible pass. Each row is
        // worked by hand, in levels of 255.
        let (_g, mut r) = gpu();
        let read =
            |more: &str| format!("void main(){{ gl_FragColor = texture2D(iChannel0, uv){more}; }}");
        let passes = |hidden: &[&str]| {
            let mut ps = json!([
                {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
                 "params": {"gain": {"type": "float", "value": 1.0}},
                 "arrays": [{"name": "level", "size": 4}]},
                {"name": "b", "channels": [{"buffer": "a"}],
                 "fragment": read(" + vec4(0.0, 0.5, 0.0, 0.0)"),
                 "params": {"gain": {"type": "float", "value": 1.0}},
                 "arrays": [{"name": "level", "size": 4}]},
                {"name": "c", "channels": [{"buffer": "b"}],
                 "fragment": read(" * vec4(0.5, 1.0, 1.0, 1.0) + vec4(0.0, 0.0, 0.25, 0.0)")},
            ]);
            for p in ps.as_array_mut().unwrap() {
                p["visible"] = json!(!hidden.contains(&p["name"].as_str().unwrap()));
            }
            scene_of(ps)
        };
        for (hidden, want) in [
            (vec![], [128.0, 128.0, 64.0]),
            (vec!["b"], [128.0, 0.0, 64.0]),
            (vec!["a"], [0.0, 128.0, 64.0]),
            (vec!["c"], [255.0, 128.0, 0.0]),
            (vec!["b", "c"], [255.0, 0.0, 0.0]),
            (vec!["a", "c"], [0.0, 128.0, 0.0]),
        ] {
            assert_eq!(r.setup(&passes(&hidden)).unwrap(), "", "{hidden:?}");
            r.set_time(Some(0.0));
            // A hidden pass keeps its params and arrays, so a set and a feed
            // still land.
            r.set_param("b", "gain", "float", &ParamValue::Float(0.5))
                .unwrap();
            let mut feed = |_: &str, _: ArraySize| Feed::Data(vec![0.5; 4]);
            let f = r.render(Some(&mut feed)).unwrap().to_vec();
            let got = [0, 1, 2].map(|c| mean(&channel(&f, c)));
            assert!(near3(got, want), "{hidden:?}: {got:?}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_channel_that_reads_a_pass_past_the_last_visible_one_reads_that_pass() {
        // b adds a to the last frame of c, a copy of b. With c hidden, b reads
        // its own last frame, as c's copy shows it: 0.25 at frame 1, 0.5 at
        // frame 2.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([
            {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(0.25, 0.0, 0.0, 1.0); }"},
            {"name": "b", "channels": [{"buffer": "a"}, {"buffer": "c"}],
             "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) \
                      + texture2D(iChannel1, uv); }"},
            {"name": "c", "channels": [{"buffer": "b"}], "visible": false,
             "fragment": "void main(){ gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0); }"},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        for want in [64.0, 128.0] {
            let f = r.render(None).unwrap().to_vec();
            let got = [0, 1, 2].map(|c| mean(&channel(&f, c)));
            assert!(near3(got, [want, 0.0, 0.0]), "{got:?}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn compile_error_leaves_the_running_scene() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();

        let bad = scene_of(json!([
            {"name": "main", "fragment": FLAT},
            {"name": "broken", "channels": [{"buffer": "main"}], "fragment": BROKEN},
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
    fn a_glsl_compile_error_alone_reads_as_a_compile_error() {
        let (_g, mut r) = gpu();
        let broken = scene_of(json!([{"name": "main", "fragment": BROKEN}]));
        assert!(!r.setup(&broken).unwrap().is_empty());
        assert!(r.compile_error);
        // A pass buffer that the GL refuses, after the compile error.
        let (name, raw) = (
            CString::new("pass 'raw'").unwrap(),
            CString::new(RAW).unwrap(),
        );
        let idx = unsafe { ffi::ghee_mp_add_pass(r.handle(), name.as_ptr(), raw.as_ptr(), 1) };
        assert!(!unsafe { ffi::ghee_mp_set_pass_size(r.handle(), idx, 1 << 17, 1 << 17) });
        assert!(!unsafe { ffi::ghee_last_error_is_compile(r.handle()) });
        unsafe { ffi::ghee_mp_abort(r.handle()) };
        let typo = scene_of(json!([
            {"name": "main", "channels": [{"buffer": "typo"}], "fragment": FLAT},
        ]));
        assert!(!r.setup(&typo).unwrap().is_empty());
        assert!(!r.compile_error);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_text_layer_that_fails_to_compile_gives_the_gls_message() {
        let (_g, mut r) = gpu();
        r.glyphs_fragment = Some(BROKEN);
        let scene = run_scene(&[words("t", "A", 0.0, 0.0, 8.0)]);
        let err = r.setup(&scene).unwrap();
        // The fragment shader fails before a link, so its log alone is the
        // GL's message, whatever vertex shader comes with it.
        let vertex = CString::new(crate::shape::program().0).unwrap();
        let broken = CString::new(BROKEN).unwrap();
        assert!(!unsafe {
            ffi::ghee_mp_keep_program(r.handle(), vertex.as_ptr(), broken.as_ptr())
        });
        let log = r.last_error();
        assert!(!log.is_empty());
        assert_eq!(err, format!("layer 't': {log}"));
        assert!(r.compile_error);
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
            r.setup(&scene_of(json!([{"name": "main", "fragment": SPECKLE}])))
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
        // Frame 0 draws one-pixel stripes, and each later frame reads the last
        // one half a pixel over. A linear fetch there blends two stripes.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{
            "name": "main",
            "channels": [{"buffer": "main"}],
            "fragment": "void main(){ float s = mod(floor(gl_FragCoord.x), 2.0); \
                     vec4 last = texture2D(iChannel0, uv + vec2(0.5 / iResolution.x, 0.0)); \
                     gl_FragColor = iFrame == 0 ? vec4(vec3(s), 1.0) : last; }",
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
            "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) \
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
                "fragment": "void main(){ gl_FragColor = vec4(shared_at(0), 0.0, 0.0, 1.0); }",
            },
            {
                "name": "b",
                "channels": [{"buffer": "a"}],
                "arrays": [{"name": "shared", "size": 4}],
                "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) \
                         + vec4(0.0, shared_at(1), 0.0, 0.0); }",
            },
            {
                "name": "c",
                "channels": [{"buffer": "b"}],
                "arrays": [{"name": "shared", "size": 4}],
                "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) \
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
        let pass_reads = "void main(){ gl_FragColor = vec4(t_cover(uv, 20.0)); }";
        let black = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";
        let cache = crate::testing::TempDir::new("engine-curves");
        let (_g, mut r) = gpu();
        r.atlas_cache = Some(cache.join("atlas"));
        // An MTSDF layer, a slug layer, and a pass that reads the words.
        for (code, render, reads) in [
            (black, "mtsdf", false),
            (black, "slug", true),
            (pass_reads, "mtsdf", true),
        ] {
            let layer = json!({"name": "t", "text": "AV", "font_size": 20, "render": render,
                               "font_family": "DejaVuSans-ASCII.ttf"});
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "passes": [{"name": "main", "fragment": code,
                                                  "layers": [layer]}]}),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(fonts).unwrap();
            assert_eq!(r.setup(&sc).unwrap(), "");
            let assets = r.scene_assets.borrow();
            let held = assets.iter().any(|n| n == "ghee_t_curves" || n == GLYPHS);
            assert_eq!(held, reads, "{code} {render}");
        }
    }

    extern "C" {
        fn glIsTexture(name: u32) -> u8;
        fn glIsFramebuffer(name: u32) -> u8;
        fn glIsRenderbuffer(name: u32) -> u8;
        fn glIsBuffer(name: u32) -> u8;
        fn glIsProgram(name: u32) -> u8;
        fn glIsShader(name: u32) -> u8;
    }

    /// The live GL objects of each kind among the first 4096 names, in the
    /// context that the last engine call left current.
    fn gl_objects() -> [usize; 6] {
        let kinds: [unsafe extern "C" fn(u32) -> u8; 6] = [
            glIsTexture,
            glIsFramebuffer,
            glIsRenderbuffer,
            glIsBuffer,
            glIsProgram,
            glIsShader,
        ];
        kinds.map(|is| (1..=4096).filter(|&n| unsafe { is(n) } != 0).count())
    }

    /// The bytes of the buffers that the DRM client of `r`'s GL holds. None
    /// where the driver reports none.
    fn gpu_bytes(r: &Renderer) -> Option<u64> {
        let fd = unsafe { ffi::ghee_gl_fd(r.handle) };
        let text = std::fs::read_to_string(format!("/proc/self/fdinfo/{fd}")).ok()?;
        crate::usage::client(&text).memory
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn setups_in_a_row_hold_what_one_setup_holds_on_the_gpu() {
        // Each scene draws frames that nothing reads before the next setup
        // replaces it, as a console does between switches. A driver that
        // reports no buffer memory leaves the GL objects to check.
        let (_g, mut r) = gpu_at(640, 480);
        let fed = scene_of(json!([
            {"name": "buf", "fragment": FLAT},
            {"name": "main", "channels": [{"buffer": "buf"}],
             "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv); }"},
        ]));
        let flat = flat_scene();
        let round = |r: &mut Renderer| {
            for scene in [&fed, &flat] {
                assert_eq!(r.setup(scene).unwrap(), "");
                r.draw(None).unwrap();
                r.draw(None).unwrap();
            }
        };
        round(&mut r);
        let (objects, bytes) = (gl_objects(), gpu_bytes(&r));
        for _ in 0..40 {
            round(&mut r);
        }
        assert_eq!(gl_objects(), objects);
        if let (Some(before), Some(after)) = (bytes, gpu_bytes(&r)) {
            assert!(
                after <= 2 * before,
                "the GPU held {} MiB after one round of setups and {} MiB after 40 more",
                before >> 20,
                after >> 20
            );
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_setup_frees_the_textures_of_the_last_scenes_assets_and_layers() {
        // Each scene of a row names its image asset, or its text layer read
        // by the pass, apart from every other scene's, so a setup that kept
        // the last scene's textures would hold one more at each scene. An
        // asset set by the host stays.
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let cache = crate::testing::TempDir::new("engine-frees");
        let (_g, mut r) = gpu();
        r.atlas_cache = Some(cache.join("atlas"));
        r.set_asset("host", 1, 1, 1, false, AssetData::U8(&[200u8]))
            .unwrap();
        let image = |i: u8| {
            let name = format!("img{i}");
            let code = format!(
                "uniform sampler2D host_tex; void main(){{ \
                 gl_FragColor = vec4(texture2D({name}_tex, vec2(0.5)).r, \
                 texture2D(host_tex, vec2(0.5)).r, 0.0, 1.0); }}"
            );
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "assets": {name.clone(): {"file": format!("{name}.png")}},
                        "passes": [{"name": "main", "fragment": code}]}),
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
            (sc, Some(i * 12))
        };
        let text = |i: u8| {
            let name = format!("t{i}");
            let code = format!(
                "uniform sampler2D host_tex; void main(){{ \
                 gl_FragColor = vec4({name}_cover(uv * {name}_size, 48.0), \
                 texture2D(host_tex, vec2(0.5)).r, 0.0, 1.0); }}"
            );
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
                        "passes": [{"name": "main", "fragment": code,
                                    "layers": [{"name": name, "text": "I",
                                                "font_family": "sans", "visible": false}]}]}),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(fonts).unwrap();
            (sc, None)
        };
        for kind in ["image", "text"] {
            let mut held = None;
            for i in 0..20u8 {
                let (sc, own) = match kind {
                    "image" => image(i),
                    _ => text(i),
                };
                assert_eq!(r.setup(&sc).unwrap(), "", "{kind} {i}");
                r.set_time(Some(0.0));
                let f = r.render(None).unwrap();
                assert_eq!(f[1], 200, "{kind} {i}");
                match own {
                    Some(v) => assert_eq!(f[0], v, "{kind} {i}"),
                    None => assert!(f.chunks_exact(4).any(|p| p[0] == 255), "{kind} {i}"),
                }
                let textures = gl_objects()[0];
                assert_eq!(*held.get_or_insert(textures), textures, "{kind} {i}");
            }
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_nul_in_a_shader_drops_the_passes_staged_before_it() {
        // The engine stages 16 passes at most, so a stale stage of 15 would
        // refuse the second pass of the next scene.
        let (_g, mut r) = gpu();
        let mut passes: Vec<Value> = (0..15)
            .map(|i| json!({"name": format!("p{i}"), "fragment": FLAT}))
            .collect();
        passes.push(json!({"name": "main", "fragment": "void main(){}\u{0}"}));
        assert!(r.setup(&scene_of(Value::from(passes))).is_err());
        let two = scene_of(json!([{"name": "a", "fragment": FLAT},
            {"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]));
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
            {"name": "a", "fragment": FLAT},
            {
                "name": "main",
                "channels": channels,
                "arrays": [{"name": "v", "size": 4}],
                "fragment": "void main(){ gl_FragColor = vec4(v_at(0)); }",
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
             "fragment": "void main(){ gl_FragColor = vec4(a11_at(0), 0.0, 0.0, 1.0); }"},
            {"name": "main", "channels": channels, "arrays": arrays, "fragment": format!(
                "void main(){{ gl_FragColor = vec4(texture2D(iChannel{}, uv).r, \
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
                    "fragment": format!("void main(){{ gl_FragColor = vec4({p}, {a}_at(0), \
                        texture2D({i}_tex, vec2(0.5)).r, 1.0); }}")}]}),
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
                .map(|i| json!({"name": format!("p{i}"), "fragment": FLAT}))
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
                .map(|n| format!("texture2D({n}_tex, vec2(0.5)).r"))
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
                    {"name": "a", "fragment": format!(
                        "void main(){{ gl_FragColor = vec4(0.0, {}, 0.0, 1.0); }}",
                        sum(&names[units..]))},
                    {"name": "main", "channels": channels, "fragment": format!(
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
        let sc = scene_of(json!([{"name": "main", "fragment": HALF}]));
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
        // A GL reads a value v back as v * 255, rounded either way.
        for (got, v) in rgb[0][..3].iter().zip([0.25, 0.5, 0.0]) {
            assert!((f64::from(*got) - v * 255.0).abs() < 1.0, "{:?}", &rgb[0]);
        }
        assert_eq!(rgb[0], rgb[1]);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_channel_reads_the_alpha_its_pass_wrote() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([
            {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(1.0, 1.0, 1.0, 0.25); }"},
            {"name": "main", "channels": [{"buffer": "a"}], "fragment":
                "void main(){ gl_FragColor = vec4(texture2D(iChannel0, uv).a, 0.0, 0.0, 1.0); }"},
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
            json!([{"name": "main", "channels": [{"buffer": "main"}], "fragment":
            "void main(){ gl_FragColor = vec4(texture2D(iChannel0, uv).a, 0.0, 0.0, 0.5); }"}]),
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
        sc.passes[0].fragment = String::new();
        assert_eq!(
            r.setup(&sc).unwrap(),
            "pass 'main' has no 'fragment' and no 'layers'"
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
            "fragment": "void main(){ gl_FragColor = vec4(mod(iTime * 37.0, 1.0), 0.0, 0.0, 1.0); }",
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
        // iTimeDelta rides the draw like iTime, so one store lands in every
        // pass. Red carries the first pass's iTimeDelta through a channel.
        // Green carries the second pass's iTimeDelta.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([
            {"name": "src", "fragment": "void main(){ gl_FragColor = vec4(iTimeDelta, 0.0, 0.0, 1.0); }"},
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(\
                         texture2D(iChannel0, uv).r, iTimeDelta, 0.0, 1.0); }",
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
    fn i_resolution_holds_each_pass_size_and_a_pixel_aspect_of_one() {
        // The left half shows the 8x4 pass `a`, the right half writes its own:
        // red and green the size, blue the pixel aspect.
        let (_g, mut r) = gpu();
        let own = "vec4(iResolution.x / 255.0, iResolution.y / 255.0, iResolution.z, 1.0)";
        let sc = scene_of(json!([
            {"name": "a", "size": [8, 4],
             "fragment": format!("void main(){{ gl_FragColor = {own}; }}")},
            {"name": "main", "channels": [{"buffer": "a"}],
             "fragment": format!("void main(){{ gl_FragColor = uv.x < 0.5 ? \
                              texture2D(iChannel0, uv) : {own}; }}")},
        ]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        let f = r.render(None).unwrap().to_vec();
        let rgb = |x| [0, 1, 2].map(|c| px(&f, W, x, H / 2, c));
        assert_eq!(rgb(1), [8.0, 4.0, 255.0]);
        assert_eq!(rgb(W - 2), [W as f64, H as f64, 255.0]);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scaled_pass_renders_at_its_share_of_the_turned_frame_rounded_and_1_at_least() {
        // The frame turns a quarter, so the scene renders at 48x64. Pass `a`
        // writes its iResolution, and main shows it over the whole frame.
        let (_g, mut r) = gpu();
        r.set_rotate(90).unwrap();
        let (fw, fh) = (H as f64, W as f64);
        let own = "vec4(iResolution.x / 255.0, iResolution.y / 255.0, 0.0, 1.0)";
        for (scale, (x, y)) in [
            (json!(0.5), (0.5, 0.5)),
            (json!([1, 0.25]), (1.0, 0.25)),
            (json!(0.7), (0.7, 0.7)),
            (json!(0.001), (0.001, 0.001)),
            (json!(2), (2.0, 2.0)),
        ] {
            let sc = scene_of(json!([
                {"name": "a", "scale": scale,
                 "fragment": format!("void main(){{ gl_FragColor = {own}; }}")},
                {"name": "main", "channels": [{"buffer": "a"}],
                 "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv); }"},
            ]));
            assert_eq!(r.setup(&sc).unwrap(), "", "{scale}");
            let f = r.render(None).unwrap().to_vec();
            let side = |n: f64, s: f64| (n * s).round().max(1.0);
            assert_eq!(
                [0, 1].map(|c| px(&f, W, W / 2, H / 2, c)),
                [side(fw, x), side(fh, y)],
                "{scale}"
            );
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scale_past_the_caps_of_a_sized_pass_is_refused_and_the_scene_runs_on() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        let before = r.render(None).unwrap().to_vec();
        for (scale, want) in [
            (
                json!(300),
                "pass 'a' is 19200x14400, over the 16384 pixel cap of a side",
            ),
            (
                json!(250),
                "pass 'a' is 16000x12000, over the 16777216 pixel cap",
            ),
        ] {
            let sc = scene_of(json!([
                {"name": "a", "scale": scale, "fragment": FLAT},
                {"name": "main", "channels": [{"buffer": "a"}], "fragment": FLAT},
            ]));
            assert_eq!(r.setup(&sc).unwrap(), want);
            assert_eq!(r.render(None).unwrap(), before);
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_array_or_an_image_past_the_gpus_texture_size_is_refused_and_the_scene_runs_on() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        let before = r.render(None).unwrap().to_vec();
        let max = r.max_texture_size();
        let over = max + 1;
        // A gray PNG one pixel past the GPU's size on a side.
        let dir = crate::testing::TempDir::new("engine-max-size");
        let mut png = Vec::new();
        let mut enc = png::Encoder::new(&mut png, over as u32, 1);
        enc.set_color(png::ColorType::Grayscale);
        let row = vec![128u8; over as usize];
        enc.write_header().unwrap().write_image_data(&row).unwrap();
        std::fs::write(dir.join("wide.png"), png).unwrap();
        let cap = format!("over this GPU's {max} pixel cap of a side");
        // White, where the running scene draws FLAT.
        let white = "void main(){ gl_FragColor = vec4(1.0); }";
        // The scene caps an array at 16384 on a side, at or below a desktop
        // GPU's size, so the rows set the array's size past the parse.
        let arrays = [ArraySize::Linear(over), ArraySize::Grid(1, over)];
        let mut rows: Vec<(Scene, String)> = arrays
            .into_iter()
            .map(|size| {
                let mut sc = scene_of(json!([
                    {"name": "main", "arrays": [{"name": "a", "size": 4}], "fragment": white},
                ]));
                sc.passes[0].arrays[0].size = size;
                let (w, h) = match size {
                    ArraySize::Linear(n) => (n, 1),
                    ArraySize::Grid(w, h) => (w, h),
                };
                (sc, format!("array 'a' is {w}x{h}, {cap}"))
            })
            .collect();
        for (doc, want) in [
            (
                json!({"version": 1, "assets": {"sky": {"file": "wide.png"}},
                       "passes": [{"name": "main", "fragment": white}]}),
                format!("asset 'sky' is {over}x1, {cap}"),
            ),
            (
                json!({"version": 1, "passes": [{"name": "main", "fragment": white,
                       "layers": [{"name": "pic", "image": "wide.png"}]}]}),
                format!("layer 'pic': 'wide.png' is {over}x1, {cap}"),
            ),
        ] {
            let mut sc = Scene::from_dict(&doc, None, None).unwrap();
            sc.load_assets(&dir).unwrap();
            rows.push((sc, want));
        }
        for (sc, want) in rows {
            assert_eq!(r.setup(&sc).unwrap(), want);
            assert_eq!(r.render(None).unwrap(), before, "{want}");
        }
        // A host's asset, and a file set live as a layer's image.
        let e = r.set_asset("sky", over as u32, 1, 1, false, AssetData::U8(&row));
        assert_eq!(
            e.unwrap_err().to_string(),
            format!("asset 'sky' is {over}x1, {cap}")
        );
        let mut enc = png::Encoder::new(std::fs::File::create(dir.join("dot.png")).unwrap(), 1, 1);
        enc.set_color(png::ColorType::Grayscale);
        enc.write_header()
            .unwrap()
            .write_image_data(&[255])
            .unwrap();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "main", "fragment": FLAT,
                    "layers": [{"name": "pic", "image": "dot.png"}]}]}),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(&dir).unwrap();
        assert_eq!(r.setup(&sc).unwrap(), "");
        let shown = r.render(None).unwrap().to_vec();
        let at = Address {
            layer: "pic",
            effect: None,
            field: "image",
        };
        let e = r.set_layer(&at, Given::Text("wide.png")).unwrap_err();
        assert_eq!(
            e.downcast_ref::<Refused>().unwrap().0,
            format!("layer 'pic': 'wide.png' is {over}x1, {cap}")
        );
        assert_eq!(r.render(None).unwrap(), shown);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_param_set_live_is_the_value_of_the_renderers_scene() {
        let (_g, mut r) = gpu();
        let sc = Scene::from_dict(
            &json!({"version": 1,
                "params": {"speed": {"type": "float", "value": 0.5}},
                "passes": [{"name": "main",
                    "params": {"gain": {"type": "float", "value": 1.0},
                               "mode": {"type": "string", "value": "a"}},
                    "fragment": "void main(){ gl_FragColor = vec4(vec3(gain * speed), 1.0); }",
                    "layers": [{"name": "title", "text": "hi", "font_family": "pixel"}]}]}),
            None,
            None,
        )
        .unwrap();
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_param("main", "gain", "float", &ParamValue::Float(0.25))
            .unwrap();
        r.set_param("main", "mode", "string", &ParamValue::Str("b".into()))
            .unwrap();
        r.set_scene_param("speed", "float", &ParamValue::Float(2.0))
            .unwrap();
        set_words(&r, "title", "ho");
        let doc = r.with_scene(Scene::to_dict).unwrap();
        let main = &doc["passes"][0];
        assert_eq!(main["params"]["gain"]["value"], json!(0.25));
        assert_eq!(main["params"]["mode"]["value"], json!("b"));
        assert_eq!(doc["params"]["speed"]["value"], json!(2.0));
        assert_eq!(main["layers"][0]["text"], json!("ho"));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_restarted_clock_draws_its_next_frame_as_the_first_one() {
        // Red is iFrame, green iTime / 5 and blue iTimeDelta * 12, in levels:
        // the first frame's 1/60 reads 51.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{
            "name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(float(iFrame) / 255.0, iTime * 0.2, iTimeDelta * 12.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        let drawn = |r: &mut Renderer| {
            let f = r.render(None).unwrap().to_vec();
            [0, 1, 2].map(|c| px(&f, W, 0, 0, c))
        };
        // the monotonic clock
        for _ in 0..3 {
            drawn(&mut r);
        }
        sleep(Duration::from_millis(50));
        r.set_dt(0.05);
        let before = drawn(&mut r);
        assert!(
            before[0] == 3.0 && before[1] > 0.0 && before[2] == 153.0,
            "{before:?}"
        );
        r.restart_clock();
        assert_eq!(drawn(&mut r), [0.0, 0.0, 51.0]);
        assert_eq!(drawn(&mut r)[0], 1.0);
        // a pinned time
        r.set_dt(0.05);
        r.set_time(Some(4.0));
        assert_eq!(drawn(&mut r), [2.0, 204.0, 153.0]);
        r.restart_clock();
        r.set_time(Some(5.5));
        assert_eq!(drawn(&mut r), [0.0, 0.0, 51.0]);
        r.set_time(Some(6.5));
        assert_eq!(drawn(&mut r), [1.0, 51.0, 51.0]);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn i_frame_reads_0_on_the_first_frame_drawn_and_counts_up_as_an_int() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{
            "name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(float(iFrame) / 255.0, 0.0, 0.0, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        for want in [0.0, 1.0, 2.0] {
            let f = r.render(None).unwrap().to_vec();
            assert_eq!(px(&f, W, 0, 0, 0), want);
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn gl_frag_coord_reads_the_pixel_beside_a_local_named_uv() {
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{
            "name": "main",
            "fragment": "void main(){ vec2 uv = vec2(0.0); \
                     gl_FragColor = vec4(floor(gl_FragCoord.x) / 255.0, uv, 1.0); }",
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        let f = r.render(None).unwrap().to_vec();
        for x in [0, 5, W - 1] {
            assert_eq!(px(&f, W, x, H / 2, 0), x as f64);
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn closing_a_second_renderer_leaves_the_first_alive() {
        let (_g, mut r) = gpu();
        let sc = flat_scene();
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        let before = r.render(None).unwrap().to_vec();
        let mut other = Renderer::new(16, 16, COMPONENTS as i32, None, false, false)
            .expect("a second renderer on the GPU");
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
            "fragment": "void main(){ gl_FragColor = vec4(mask_sample2(vec2(0.75, 0.75)), \
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
            "fragment": "void main(){ gl_FragColor = vec4(vals_at(0), 0.0, 0.0, 1.0); }",
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
                "common": bad,
                "passes": [{
                    "name": "main",
                    "fragment": "void main(){ gl_FragColor = vec4(sat(uv.x), 0.0, 0.0, 1.0); }",
                }],
            }),
            None,
            None,
        )
        .unwrap();
        let e = r.setup(&sc).unwrap();
        // The GL's own log follows, once.
        let log = e
            .strip_prefix("pass 'main', line 2 of the common block: ")
            .unwrap_or_else(|| panic!("{e}"));
        assert!(log.contains("nope") && !log.contains("pass '"), "{e}");
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
                "common": "float sat(float x){ return clamp(x, 0.0, 1.0); }",
                "passes": [
                    {"name": "a", "fragment": FLAT},
                    {"name": "b", "fragment": "void main(){\n  float v = 1.0;\n  nope(v);\n}"},
                ],
            }),
            None,
            None,
        )
        .unwrap();
        let e = r.setup(&sc).unwrap();
        // The GL's own log follows, once.
        let log = e
            .strip_prefix("pass 'b', line 3 of its code: ")
            .unwrap_or_else(|| panic!("{e}"));
        assert!(log.contains("nope") && !log.contains("pass '"), "{e}");
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
            "fragment": code,
            "arrays": [{"name": name, "size": size}],
        }]))
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn array_samples_correctly() {
        let (_g, mut r) = gpu();
        let sc = array_scene(
            "void main(){ vec2 uv=gl_FragCoord.xy/iResolution.xy; \
             gl_FragColor=vec4(vec3(vals_at(int(uv.x*float(vals_count-1)))),1.0); }",
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
            "void main(){ vec2 uv=gl_FragCoord.xy/iResolution.xy; \
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
                "void main(){ gl_FragColor=vec4(vec3(vals_sample(uv.x)),1.0); }",
                json!(N),
            ),
            (
                "void main(){ gl_FragColor=vec4(vec3(vals_sample2(uv)),1.0); }",
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
    fn an_asset_binds_as_its_name_with_tex_and_carries_its_size() {
        let (_g, mut r) = gpu();
        // logo lands before setup, so the commit resolves it.
        // late lands on the running scene.
        let code = "uniform sampler2D logo_tex; uniform vec2 logo_size; \
                    uniform sampler2D late_tex; \
                    void main(){ gl_FragColor = vec4(\
                    texture2D(logo_tex, uv).r, \
                    texture2D(late_tex, uv).r, \
                    logo_size.x / 8.0, 1.0); }";
        r.set_asset("logo", 2, 2, 4, false, AssetData::U8(&[128u8; 16]))
            .unwrap();
        let sc = scene_of(json!([{"name": "main", "fragment": code}]));
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
            "void main(){ gl_FragColor=vec4(vec3(vals_sample(uv.x)),1.0); }",
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
            {"name": "a", "fragment": FLAT, "arrays": [{"name": "v", "size": [3, 5]}]},
            {"name": "b", "fragment": FLAT, "arrays": [{"name": "v", "size": 7}]},
        ]);
        assert_eq!(
            r.setup(&scene_of(clash)).unwrap(),
            "array 'v' declared at conflicting sizes ((3, 5) and 7)"
        );
        // The bad channel outranks the texture-unit budget of the same pass.
        let units = r.max_texture_units() as usize;
        let typos: Vec<Value> = (0..units).map(|_| json!({"buffer": "typo"})).collect();
        let both = json!([
            {"name": "a", "fragment": FLAT},
            {"name": "main", "channels": typos, "arrays": [{"name": "v", "size": 4}],
             "fragment": "void main(){ gl_FragColor = vec4(v_at(0)); }"},
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
            {"name": "a", "fragment": FLAT},
            {"name": "main", "channels": channels,
             "arrays": [{"name": "v", "size": 4}],
             "fragment": "void main(){ gl_FragColor = vec4(v_at(0)); }"},
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
            "fragment": "void main(){ gl_FragColor = vec4(tint.r, off.x, off.y, 1.0); }",
            "params": {"tint": {"type": "color", "value": [0.0, 0.0, 0.0]},
                       "off": {"type": "vec2", "value": [0.0, 0.0]},
                       "label": {"type": "string", "value": "hi"}},
        }]));
        assert_eq!(r.setup(&sc).unwrap(), "");
        r.set_time(Some(0.0));
        // A colour written as hex sets the same vec3 as one written as a list.
        let color = ParamValue::Color(Rgb::from_json(&json!("#801a33")).unwrap());
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
            .set_param("main", "ghost", "color", &ParamValue::Color(Rgb::WHITE))
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
        let code = "void main(){ float x = uv.x; \
                    gl_FragColor = x < 0.25 ? vec4(a, 0.0, 1.0) : x < 0.5 ? vec4(b, 1.0) \
                    : x < 0.75 ? vec4(c.xyz * c.w, 1.0) : vec4(tint, 1.0); }";
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "params": {"a": {"type": "vec2", "value": [0.25, 0.5]},
                           "tint": {"type": "color", "value": [0.5, 0.25, 1.0]}},
                "passes": [{"name": "main", "fragment": code, "params": {
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
        let tint = ParamValue::Color(Rgb::numbers([0.0, 1.0, 0.5]));
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
        let sc = scene_of(json!([{"name": "main", "fragment": code,
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
             "fragment": "void main(){ vec2 t = floor(uv * 256.0); \
                      gl_FragColor = vec4(vec3(mod(t.x + t.y, 2.0) * level), 1.0); }"},
            {"name": "main", "channels": [{"buffer": "check"}], "fragment": main},
        ]))
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_with_mipmaps_read_small_reads_its_average() {
        let (_g, mut r) = gpu_at(64, 64);
        let read = "void main(){ gl_FragColor = texture2D(iChannel0, uv + 0.5 / 256.0); }";
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
        let fs = "uniform sampler2D iChannel0; uniform vec3 iResolution;\n\
                  void main(){ gl_FragColor = texture2D(iChannel0, v_px / iResolution.xy + 0.5 / 256.0); }";
        let batches = [batch("main", fs, Blend::Normal)];
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
                    {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(P, 0.0, 0.0, 1.0); }"},
                    {"name": "main", "channels": [{"buffer": "a"}],
                     "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) \
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
            "fragment": "void main(){ gl_FragColor = vec4(mask_sample2(vec2(0.25, 0.25)), \
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
            "fragment": "void main(){ gl_FragColor = vec4(uv.x, uv.y, 0.25, 1.0); }",
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

    /// ghee_a0.xy is the vertex in pass pixels, y down like uv, and
    /// ghee_a1 the premultiplied colour.
    fn batch_vs(extra: &str) -> String {
        format!(
            "{PRECISION}attribute vec4 ghee_a0; attribute vec4 ghee_a1;\n\
             uniform vec3 iResolution; uniform float ghee_vflip;\n\
             varying vec4 v_color; varying vec2 v_px;\n{extra}\n\
             void main() {{\n\
               vec2 ndc = ghee_a0.xy / iResolution.xy * 2.0 - 1.0;\n\
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
        // src (0.25, 0, 0, 0.5) premultiplied, over a gray pass of 0.5. Each
        // row is its formula worked by hand, in levels of 255.
        let src = [0.25, 0.0, 0.0, 0.5];
        for (blend, want) in [
            // src + dst * (1 - src.a): (0.5, 0.25, 0.25)
            (Blend::Normal, [128.0, 64.0, 64.0]),
            // src + dst: (0.75, 0.5, 0.5)
            (Blend::Add, [191.0, 128.0, 128.0]),
            // src * dst + dst * (1 - src.a): (0.375, 0.25, 0.25)
            (Blend::Multiply, [96.0, 64.0, 64.0]),
            // src + dst * (1 - src): (0.625, 0.5, 0.5)
            (Blend::Screen, [159.0, 128.0, 128.0]),
        ] {
            let sc = scene_of(json!([{"name": "main", "fragment": GRAY}]));
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
        // The pass paints the top rows, uv.y < 0.25, red. The batch
        // covers the same rows by pixel and adds green.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "fragment":
            "void main(){ gl_FragColor = vec4(uv.y < 0.25 ? 1.0 : 0.0, 0.0, 0.0, 1.0); }"}]));
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
        let sc = scene_of(json!([{"name": "main", "fragment":
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
    fn a_layer_built_again_and_again_holds_the_batches_of_one_layer() {
        // The engine hands the index of a batch taken out to the next batch
        // that comes in.
        let (_g, mut r) = gpu();
        let words = |px: i32| json!({"name": "t", "text": "Hi", "font_size": px});
        let doc = json!([{"name": "main", "fragment": FLAT, "layers": [words(12)]}]);
        assert_eq!(r.setup(&scene_of(doc)).unwrap(), "");
        let held = |r: &Renderer| {
            let d = r.drawn("t").unwrap();
            let mut b = vec![d.batch, d.colour, d.boxes];
            b.sort_unstable();
            b
        };
        let first = held(&r);
        for px in [14, 16, 18, 20] {
            r.replace_layer(&words(px)).unwrap();
        }
        assert_eq!(held(&r), first);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_pass_draws_a_hundred_batches() {
        // Each batch adds 1/255 of red over black.
        let (_g, mut r) = gpu();
        let black = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";
        let sc = scene_of(json!([{"name": "main", "fragment": black}]));
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
        let sc = scene_of(json!([{"name": "main", "fragment": GRAY}]));
        let batches = [
            batch("main", PLAIN_FS, Blend::Normal),
            batch("main", PLAIN_FS, Blend::Normal),
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
        // texel and the channel. `level` and `iResolution` sit in both stages.
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
                    {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.75, 1.0); }"},
                    {"name": "main", "channels": [{"buffer": "a"}],
                     "arrays": [{"name": "vals", "size": 4}],
                     "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"},
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
                "v_px = ghee_a0.xy; gl_Position.x += step(0.25, level) * shift * 2.0 / iResolution.x;",
            ),
            fragment: batch_fs(
                "uniform float level; uniform sampler2D vals_tex; uniform sampler2D pad_tex;\n\
                 uniform sampler2D iChannel0; uniform vec2 pad_size; uniform vec3 iResolution;\n\
                 void main(){ gl_FragColor = vec4(level, \
                   texture2D(vals_tex, vec2(0.125, 0.5)).r * 0.5 + texture2D(pad_tex, vec2(0.5)).r, \
                   texture2D(iChannel0, v_px / iResolution.xy).b * pad_size.x, 1.0); }",
            ),
            blend: Blend::Normal,
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
        let sc = scene_of(json!([{"name": "main", "fragment": GRAY}]));
        let l = batch(
            "main",
            "uniform sampler2D early_tex; uniform sampler2D late_tex;\n\
             void main(){ gl_FragColor = vec4(texture2D(early_tex, vec2(0.5)).r, \
               texture2D(late_tex, vec2(0.5)).r, 0.0, 1.0); }",
            Blend::Normal,
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
            {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"},
            {"name": "main", "channels": [{"buffer": "a"}],
             "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv); }"},
        ]));
        assert_eq!(
            r.setup_batches(&sc, &[batch("a", PLAIN_FS, Blend::Normal)])
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
        let sc = scene_of(json!([{"name": "main", "fragment": GRAY}]));
        assert_eq!(
            r.setup_batches(&sc, &[batch("main", PLAIN_FS, Blend::Normal)])
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
        let mut bad = batch("main", PLAIN_FS, Blend::Normal);
        bad.vertex = bad
            .vertex
            .replace("v_color = ghee_a1;", "v_color = no_such_symbol;");
        let err = r.setup_batches(&flat_scene(), &[bad]).unwrap();
        assert!(err.starts_with("batch 0: vertex shader:"), "{err}");
        assert_eq!(r.error_pass, Some(ErrorPass::Index(0)));
        let bad = batch(
            "main",
            "void main(){ gl_FragColor = no_such_symbol; }",
            Blend::Normal,
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
            .setup_batches(&flat_scene(), &[batch("typo", PLAIN_FS, Blend::Normal)])
            .unwrap();
        assert_eq!(
            err,
            "batch 0 draws into pass 'typo', which is not a pass in this scene"
        );
        assert_eq!(
            r.setup_batches(&flat_scene(), &[batch("main", PLAIN_FS, Blend::Normal)])
                .unwrap(),
            ""
        );
        let one = quad(0.0, 0.0, 1.0, 1.0, [1.0; 4]);
        assert_eq!(
            r.set_batch_quads(1, &one, 2).unwrap_err().to_string(),
            "no batch 1: setup_batches() took 1 batch"
        );
        assert_eq!(
            r.set_batch_quads(0, &one, 9).unwrap_err().to_string(),
            "batch 0: a vertex holds 1 to 8 attributes, got 9"
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
    fn a_batch_with_eight_attributes_reads_each_one() {
        // GLES2 guarantees eight attributes, and VC4 holds no more. Each
        // channel is the sum of one of ghee_a1 to ghee_a3 and one of ghee_a5
        // to ghee_a7, and ghee_a4 holds the alpha.
        let (_g, mut r) = gpu();
        let sc = scene_of(json!([{"name": "main", "fragment": GRAY}]));
        let attributes: String = (0..8)
            .map(|k| format!("attribute vec4 ghee_a{k};\n"))
            .collect();
        let vs = format!(
            "{PRECISION}{attributes}\
             uniform vec3 iResolution; uniform float ghee_vflip; varying vec4 v_color;\n\
             void main() {{ vec2 ndc = ghee_a0.xy / iResolution.xy * 2.0 - 1.0;\n\
               gl_Position = vec4(ndc.x, ndc.y * ghee_vflip, 0.0, 1.0);\n\
               v_color = vec4(ghee_a1.x + ghee_a5.x, ghee_a2.y + ghee_a6.y, ghee_a3.z + ghee_a7.z,\n\
                              ghee_a4.w); }}"
        );
        let l = Batch {
            pass: "main".to_string(),
            vertex: vs,
            fragment: format!("{PRECISION}varying vec4 v_color;\n{PLAIN_FS}"),
            blend: Blend::Normal,
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
                [x, y, 0.0, 0.0],
                [0.125, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.375, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.125, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.375, 0.0],
            ]
        })
        .flatten()
        .collect();
        r.set_batch_quads(0, &data, 8).unwrap();
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
        let load = |layers: Value| {
            let mut sc = Scene::from_dict(
                &json!({
                    "version": 1,
                    "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
                    "passes": [{"name": "main", "layers": layers,
                        "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 1.0); }"}],
                }),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(fonts).unwrap();
            sc
        };
        let with = load(
            json!([{"name": "label", "text": "AV", "font_family": "sans",
            "render": "slug", "x": 3, "y": 2, "font_size": 30, "color": [1.0, 0.5, 0.0]}]),
        );
        let alone = load(json!([]));
        let (_g, mut r) = gpu();
        let frame = |r: &mut Renderer| {
            r.set_time(Some(0.0));
            r.render(None).unwrap().to_vec()
        };
        assert_eq!(r.setup(&with).unwrap(), "");
        let layer = with.passes[0].layers[0].as_text().unwrap();
        let built = layer.build("AV").unwrap().built.unwrap();
        assert_eq!((built.quads.count, built.boxes.count), (2, 0));
        let drawn = frame(&mut r);
        assert!(r.text_errors().is_empty());
        // The error-box batch is live: given the glyphs' quads, it covers
        // them.
        let boxes = r.layers[0].boxes;
        r.batch_quads(boxes, "boxes", &built.quads.floats, 4)
            .unwrap();
        assert!(frame(&mut r) != drawn, "the error-box batch drew nothing");
        let detail = r.detail(layer);
        let l = Batch {
            pass: "main".to_string(),
            vertex: layer.shaders(detail).glyphs.0,
            fragment: layer.shaders(detail).glyphs.1,
            blend: Blend::Normal,
        };
        assert_eq!(r.setup_batches(&alone, &[l]).unwrap(), "");
        // The layer's own uniforms and the words' texture, as a setup sets
        // them for a layer of the scene.
        let b = r.batches[0];
        for (name, ty, value) in layer.uniforms() {
            r.set_batch_uniform(b, &name, ty, value).unwrap();
        }
        r.set_asset(
            "ghee_label_curves",
            built.width,
            built.height,
            4,
            false,
            AssetData::U8(&built.texels),
        )
        .unwrap();
        r.bind_asset(
            b,
            &text::internal(GLSL_NAME, "_curves"),
            "ghee_label_curves",
        )
        .unwrap();
        for (suffix, v) in &built.uniforms {
            let name = text::internal(GLSL_NAME, suffix);
            r.set_batch_uniform(b, &name, "float", Value::from(*v))
                .unwrap();
        }
        let size = Value::from(built.size.to_vec());
        r.set_batch_uniform(b, &text::internal(GLSL_NAME, "_size"), "vec2", size)
            .unwrap();
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
                "passes": [{"name": "main", "fragment": "void main(){ \
                    float c = label_cover(uv * label_size, iResolution.y / label_size.y); \
                    gl_FragColor = vec4(c, c, c, 1.0); }",
                    "layers": [{"name": "label", "text": words, "font_family": "sans",
                                "text_align": align, "visible": false}]}],
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
        set_words(&r, "label", "other words");
        set_words(&r, "label", &words);
        assert!(still(&mut r) == left);
        let laid = r.layouts.get();
        for _ in 0..100 {
            set_words(&r, "label", &words);
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
    fn slug_words_of_glyphs_that_the_renderer_holds_read_no_outline_and_upload_nothing() {
        let layer = json!({"name": "t", "text": "0123456789", "font_family": "pixel",
                           "font_size": 8});
        let sc = scene_of(json!([{"name": "main", "fragment": FLAT, "layers": [layer]}]));
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&sc).unwrap(), "");
        let held = || (r.glyph_bytes.get(), r.glyphs.borrow().reads);
        let (sent, read) = held();
        assert!(sent > 0 && read == 10, "{sent} bytes, {read} outlines");
        for k in 0..100 {
            set_words(&r, "t", &format!("{:02}", k * 37 % 100));
        }
        assert_eq!(held(), (sent, read));
        // New glyphs read their outlines and upload their rows.
        set_words(&r, "t", "AB");
        assert!(r.glyph_bytes.get() > sent);
        assert_eq!(r.glyphs.borrow().reads, read + 2);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_text_given_words_after_words_draws_as_a_scene_built_with_the_last() {
        // Each row gives t, centred by its size, its words by set_text, or a
        // look by update_layer that builds it again, and the frame matches
        // the scene built with that look: words of as many glyphs, none,
        // none again, more, the same words built again at another size.
        let text = |words: &str, px: i32| {
            json!({"name": "t", "text": words, "font_family": "pixel",
                   "font_size": px})
        };
        let scene =
            |l: &Value| scene_of(json!([{"name": "main", "fragment": FLAT, "layers": [l]}]));
        let rows = [
            (text("ab", 8), false),
            (text("cd", 8), false),
            (text("", 8), false),
            (text("", 8), false),
            (text("abc", 8), false),
            (text("abc", 16), true),
            (text("xy", 16), false),
        ];
        let (_g, mut r) = gpu();
        let built: Vec<Vec<u8>> = rows
            .iter()
            .map(|(look, _)| first_frames(&mut r, &scene(look), 1))
            .collect();
        first_frames(&mut r, &scene(&rows[0].0), 1);
        for ((look, rebuild), built) in rows.iter().zip(&built) {
            match rebuild {
                true => assert!(r.update_layer(look).unwrap()),
                false => set_words(&r, "t", look["text"].as_str().unwrap()),
            }
            r.set_time(Some(0.0));
            assert!(still(&mut r) == *built, "{look}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_changed_shapes_of_a_run_hand_the_engine_their_quads_in_one_upload() {
        // Three rects of one run take a frame's calls at once. One rect that
        // changes hands the engine its own quad, two hand it the whole run
        // once, and a rect that hides hands it the run's quads that stay.
        use crate::draw::{Arg, Call, CallKind};
        let rect = |name: &str, x: f64| {
            json!({"name": name, "shape": "rect", "x": x, "y": 2, "width": 4,
                   "height": 4})
        };
        let call = |name: &str, x: f64, visible: bool| {
            let mut args = vec![Arg::None; CallKind::Rect.keys().len()];
            args[0] = Arg::Num(x);
            args[1] = Arg::Num(2.0);
            args[2] = Arg::Num(4.0);
            args[3] = Arg::Num(4.0);
            args[7] = Arg::Bool(visible);
            Call {
                kind: CallKind::Rect,
                layer: name.to_string(),
                args,
            }
        };
        let sc = scene_of(json!([{"name": "main", "fragment": FLAT, "layers": [
            rect("a", 2.0), rect("b", 10.0), rect("c", 18.0)]}]));
        let quad = 16 * crate::shape::ATTRIBUTES;
        let rows = [
            (vec![call("b", 11.0, true)], quad),
            (vec![call("a", 3.0, true), call("c", 19.0, true)], 3 * quad),
            (vec![call("b", 11.0, false)], 2 * quad),
        ];
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&sc).unwrap(), "");
        for (calls, floats) in rows {
            let before = r.run_floats.get();
            let taken = r.update_calls(&calls).unwrap();
            assert_eq!(taken.changed.len(), calls.len(), "{calls:?}");
            assert_eq!(r.run_floats.get() - before, floats, "{calls:?}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_changed_shape_hands_the_engine_its_own_quads_and_draws_as_a_scene_built_with_it() {
        // Three rects of one run. Each row gives b a look, by an update or
        // by a live set, and the frame matches the scene built with it. A
        // look that keeps b's count of quads hands the engine b's quads
        // alone.
        let rect = |name: &str, x: i32, w: i32, color: &str| {
            json!({"name": name, "shape": "rect", "x": x, "y": 2, "width": w, "height": 10,
                   "color": color})
        };
        let scene = |b: &Value| {
            scene_of(json!([{"name": "main", "fragment": FLAT, "layers": [
                rect("a", 2, 10, "red"), b, rect("c", 40, 10, "blue")]}]))
        };
        let quad = 16 * crate::shape::ATTRIBUTES;
        let rows = [
            (rect("b", 20, 16, "lime"), None, Some(quad)),
            (
                rect("b", 20, 10, "yellow"),
                Some(("color", json!("yellow"))),
                Some(quad),
            ),
            (
                json!({"name": "b", "shape": "rect", "x": 20, "y": 2, "width": 10,
                       "height": 10, "color": "lime", "visible": false}),
                Some(("visible", json!(false))),
                None,
            ),
        ];
        let (_g, mut r) = gpu();
        for (look, live, floats) in rows {
            let built = first_frames(&mut r, &scene(&look), 1);
            first_frames(&mut r, &scene(&rect("b", 20, 10, "lime")), 1);
            let before = r.run_floats.get();
            match &live {
                Some((field, value)) => {
                    let at = Address {
                        layer: "b",
                        effect: None,
                        field,
                    };
                    r.set_layer(&at, Given::Json(value)).unwrap();
                }
                None => assert!(r.update_layer(&look).unwrap()),
            }
            if let Some(floats) = floats {
                assert_eq!(r.run_floats.get() - before, floats, "{look}");
            }
            r.set_time(Some(0.0));
            assert!(still(&mut r) == built, "{look}");
        }
    }

    /// A scene of pass `main` of `layers` over FLAT, with `pic`, an image
    /// asset of 2x2 pixels.
    fn run_scene(layers: &[Value]) -> Scene {
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "assets": {"pic": {"file": "pic.png"}},
                    "passes": [{"name": "main", "fragment": FLAT, "layers": layers}]}),
            None,
            None,
        )
        .unwrap();
        sc.assets[0].image = Some(crate::assets::Image {
            width: 2,
            height: 2,
            components: 4,
            pixels: Pixels::U8(vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128,
            ]),
        });
        sc
    }

    /// The runs of `r` in the order of the layers: each one's members and
    /// its mode.
    fn runs_of(r: &Renderer) -> Vec<(Vec<String>, Mode)> {
        let mut out: Vec<(Vec<String>, Mode)> = r
            .runs
            .iter()
            .map(|run| {
                let names = run.members.iter().map(|m| m.name.clone()).collect();
                (names, run.mode())
            })
            .collect();
        out.sort_by_key(|(names, _)| r.places().scene[&names[0]]);
        out
    }

    /// A rect at `x`, `y` of `w` by `h` pixels, half opaque.
    fn rect(name: &str, at: [f64; 4], color: &str) -> Value {
        json!({"name": name, "shape": "rect", "x": at[0], "y": at[1], "width": at[2],
               "height": at[3], "color": color, "opacity": 0.6})
    }

    /// Give text layer `layer` of `r` the words `words` through its live
    /// `text`.
    fn set_words(r: &Renderer, layer: &str, words: &str) {
        let at = Address {
            layer,
            effect: None,
            field: "text",
        };
        r.set_layer(&at, Given::Text(words)).unwrap();
    }

    /// Words in the pixel font at `x`, `y` and `px` pixels per em.
    fn words(name: &str, words: &str, x: f64, y: f64, px: f64) -> Value {
        json!({"name": name, "text": words, "font_family": "pixel", "font_size": px,
               "x": x, "y": y})
    }

    /// `layers` with `layer` at `at`, or in place of the layer of its name.
    fn with(layers: &[Value], at: Option<usize>, layer: &Value) -> Vec<Value> {
        let mut out = layers.to_vec();
        match at {
            Some(k) => out.insert(k, layer.clone()),
            None => {
                let k = out.iter().position(|l| l["name"] == layer["name"]).unwrap();
                out[k] = layer.clone();
            }
        }
        out
    }

    /// Overlapping shapes and slug texts that draw as one run: the pixel
    /// font at 8 and 16 px and fitted, and DejaVu at 13 px in a gradient.
    fn members() -> Vec<Value> {
        let mut t4 = json!({"name": "t4", "text": "Wq", "render": "slug", "font_size": 13,
                            "x": 70.5, "y": 30, "color": {"from": "red", "to": "#40a0ff",
                            "angle": 60}});
        t4["opacity"] = json!(0.8);
        vec![
            rect("a", [2.0, 2.0, 30.0, 12.0], "red"),
            words("t1", "Ab", 10.0, 4.0, 8.0),
            json!({"name": "b", "shape": "line", "points": [[4, 8], [40, 20], [52, 6]],
                   "stroke_width": 2, "color": "lime"}),
            json!({"name": "t2", "text": "cd", "font_family": "pixel", "font_size": 16,
                   "x": 20, "y": 6, "color": "orange", "opacity": 0.6}),
            rect("c", [30.0, 0.0, 20.0, 20.0], "blue"),
            json!({"name": "t3", "text": "fit", "font_family": "pixel", "x": 96}),
            t4,
        ]
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_run_changed_in_place_draws_and_groups_as_a_scene_set_up_with_its_layers() {
        // Each row changes a running scene of shapes and slug texts in one
        // way, and the frame and the runs match those of the scene set up
        // with the layers that the change leaves: a layer added into a run,
        // or splitting one, a member built again, replaced or leaving its
        // run, a layer between runs that leaves them to join, and the live
        // fields of members. An image ahead of a run leaves its picture on
        // the texture unit that a run's texts read their curves from.
        enum Op {
            Add(usize, Value),
            Replace(Value),
            Update(Value),
            Set(&'static str, &'static str, Value),
            Words(&'static str, &'static str),
        }
        let base = members();
        let shapes: Vec<Value> = base
            .iter()
            .filter(|l| l.get("shape").is_some())
            .cloned()
            .collect();
        let mut t2_blink = base[3].clone();
        t2_blink["blink"] = json!({});
        let mut t2_small = base[3].clone();
        t2_small["font_size"] = json!(8);
        let t2_other = json!({"name": "t2", "text": "xy", "font_family": "pixel", "font_size": 16,
                              "x": 22, "y": 8, "color": "lime"});
        let mut wave = words("w", "w", 24.0, 30.0, 8.0);
        wave["wave"] = json!({"height": 0.2});
        let calm = words("w", "w", 24.0, 30.0, 8.0);
        let mut wave_b = wave.clone();
        wave_b["name"] = json!("b");
        let between = with(&base, Some(3), &wave);
        let mut a_add = base[0].clone();
        a_add["blend"] = json!("add");
        let mut t3_fits = base[5].clone();
        t3_fits["text"] = json!("fits");
        let set = |name: &str, field: &str, value: Value| {
            let mut out = base.clone();
            let k = out.iter().position(|l| l["name"] == name).unwrap();
            out[k][field] = value;
            out
        };
        let gradient = json!({"from": "yellow", "to": "purple", "angle": 30});
        let s = rect("s", [12.0, 10.0, 6.0, 6.0], "yellow");
        let image = json!({"name": "img", "image": "pic", "x": 20, "y": 4, "width": 8});
        let add = {
            let mut l = rect("plus", [18.0, 4.0, 10.0, 10.0], "white");
            l["blend"] = json!("add");
            l
        };
        let rows: Vec<(&str, Vec<Value>, Op, Vec<Value>)> = vec![
            (
                "a shape joins",
                base.clone(),
                Op::Add(2, s.clone()),
                with(&base, Some(2), &s),
            ),
            (
                "a shape joins a text alone",
                vec![image.clone(), base[1].clone()],
                Op::Add(2, s.clone()),
                vec![image.clone(), base[1].clone(), s.clone()],
            ),
            (
                "a text is left alone",
                vec![image.clone(), base[1].clone(), base[2].clone()],
                Op::Replace(wave_b.clone()),
                vec![image.clone(), base[1].clone(), wave_b.clone()],
            ),
            (
                "a text joins shapes",
                shapes.clone(),
                Op::Add(1, base[1].clone()),
                with(&shapes, Some(1), &base[1]),
            ),
            (
                "an image splits",
                base.clone(),
                Op::Add(3, image.clone()),
                with(&base, Some(3), &image),
            ),
            (
                "an effect splits",
                base.clone(),
                Op::Add(3, wave.clone()),
                between.clone(),
            ),
            (
                "a blend splits",
                base.clone(),
                Op::Add(3, add.clone()),
                with(&base, Some(3), &add),
            ),
            (
                "a member takes an effect",
                base.clone(),
                Op::Update(t2_blink.clone()),
                with(&base, None, &t2_blink),
            ),
            (
                "a member is replaced",
                base.clone(),
                Op::Replace(t2_other.clone()),
                with(&base, None, &t2_other),
            ),
            (
                "a member takes a size",
                base.clone(),
                Op::Update(t2_small.clone()),
                with(&base, None, &t2_small),
            ),
            (
                "the first member takes a blend",
                base.clone(),
                Op::Update(a_add.clone()),
                with(&base, None, &a_add),
            ),
            (
                "an effect leaves",
                between.clone(),
                Op::Replace(calm.clone()),
                with(&between, None, &calm),
            ),
            (
                "a shape becomes a text",
                base.clone(),
                Op::Replace(words("c", "C", 30.0, 0.0, 8.0)),
                with(&base, None, &words("c", "C", 30.0, 0.0, 8.0)),
            ),
            (
                "a text moves",
                base.clone(),
                Op::Set("t2", "x", json!(30)),
                set("t2", "x", json!(30)),
            ),
            (
                "a text hides",
                base.clone(),
                Op::Set("t1", "visible", json!(false)),
                set("t1", "visible", json!(false)),
            ),
            (
                "a text takes a gradient",
                base.clone(),
                Op::Set("t1", "color", gradient.clone()),
                set("t1", "color", gradient.clone()),
            ),
            (
                "a line grows",
                base.clone(),
                Op::Set("b", "stroke_width", json!(3)),
                set("b", "stroke_width", json!(3)),
            ),
            (
                "a text takes words",
                base.clone(),
                Op::Words("t3", "fits"),
                with(&base, None, &t3_fits),
            ),
        ];
        let (_g, mut r) = gpu_at(128, 64);
        for (what, from, op, want) in rows {
            let built = first_frames(&mut r, &run_scene(&want), 1);
            let grouped = runs_of(&r);
            first_frames(&mut r, &run_scene(&from), 1);
            match op {
                Op::Add(k, l) => r.add_layer("main", k, &l).unwrap(),
                Op::Replace(l) => r.replace_layer(&l).unwrap(),
                Op::Update(l) => assert!(r.update_layer(&l).unwrap(), "{what}"),
                Op::Set(layer, field, value) => {
                    let at = Address {
                        layer,
                        effect: None,
                        field,
                    };
                    r.set_layer(&at, Given::Json(&value)).unwrap();
                }
                Op::Words(layer, words) => set_words(&r, layer, words),
            }
            r.set_time(Some(0.0));
            assert!(still(&mut r) == built, "{what}");
            assert_eq!(runs_of(&r), grouped, "{what}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_run_draws_as_set_up_after_a_text_outside_it_grows_the_texture_of_curves() {
        // The texts of a run read the texture of curves at its height, and
        // the new words of a text with an effect grow it.
        let mut wave = words("w", "w", 24.0, 30.0, 8.0);
        wave["wave"] = json!({"height": 0.2});
        let from = with(&members(), Some(3), &wave);
        let many = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let mut grown = wave.clone();
        grown["text"] = json!(many);
        let want = with(&from, None, &grown);
        let (_g, mut r) = gpu_at(128, 64);
        first_frames(&mut r, &run_scene(&from), 1);
        let rows = r.glyphs.borrow().rows();
        assert!(r.set_text("w", many).unwrap());
        assert!(r.glyphs.borrow().rows() > rows);
        r.set_time(Some(0.0));
        let changed = still(&mut r);
        assert!(changed == first_frames(&mut r, &run_scene(&want), 1));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_run_draws_each_member_as_its_own_program_does_and_a_glyph_edge_within_a_level() {
        // The members draw as one run, and apart, each between texts with an
        // effect and no words. A run reads a glyph's font size from its
        // vertex, where the text's own program holds a constant, which moves
        // an edge pixel by a level at most.
        let together = members();
        let mut apart = Vec::new();
        for (k, l) in together.iter().enumerate() {
            apart.push(l.clone());
            apart.push(
                json!({"name": format!("x{k}"), "text": "", "font_family": "pixel",
                              "blink": {}}),
            );
        }
        let (_g, mut r) = gpu_at(128, 64);
        let run = first_frames(&mut r, &run_scene(&together), 1);
        let names: Vec<String> = together
            .iter()
            .map(|l| l["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(runs_of(&r), [(names, Mode::Mixed)]);
        let each = first_frames(&mut r, &run_scene(&apart), 1);
        assert!(runs_of(&r).iter().all(|(m, _)| m.len() == 1));
        assert!(each != first_frames(&mut r, &run_scene(&[]), 1));
        assert!(worst(&run, &each) <= 1, "{}", worst(&run, &each));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_run_of_one_member_or_of_shapes_draws_through_its_own_program() {
        // A text alone draws through its own program and shapes through
        // theirs, and a scene with no run of a text and another member
        // compiles none.
        let layers = vec![
            words("t1", "Ab", 2.0, 2.0, 8.0),
            json!({"name": "img", "image": "pic", "x": 20, "y": 4, "width": 8}),
            rect("a", [2.0, 20.0, 10.0, 10.0], "red"),
            rect("b", [8.0, 24.0, 10.0, 10.0], "blue"),
            {
                let mut t = words("t2", "cd", 30.0, 20.0, 16.0);
                t["blend"] = json!("add");
                t
            },
        ];
        let (_g, mut r) = gpu_at(128, 64);
        first_frames(&mut r, &run_scene(&layers), 1);
        let s = |n: &[&str]| n.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            runs_of(&r),
            [
                (s(&["t1"]), Mode::Alone),
                (s(&["a", "b"]), Mode::Shapes),
                (s(&["t2"]), Mode::Alone)
            ]
        );
        assert!(!r.mixed_kept.get());
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_layer_past_its_cutoff_stays_out_of_a_run_with_a_text() {
        // A rect past SHAPE_PIXELS between texts draws in a run of shapes,
        // and a text past GLYPH_PIXELS between shapes draws alone, so every
        // layer draws through its own program, byte for byte as apart. A
        // live set that brings the rect under its cutoff joins it to the
        // texts at the next frame, and a text that leaves the run compiles
        // nothing.
        let side = (crate::run::SHAPE_PIXELS.sqrt() * 1.2).ceil();
        let big = rect("big", [4.0, 4.0, side, side], "navy");
        let mut long = String::from("M");
        let px = 32.0;
        loop {
            let t = TextLayer::from_dict("t", &words("t", &long, 0.0, 0.0, px)).unwrap();
            let b = t.build(&long).unwrap().built.unwrap();
            if crate::run::glyph_pixels(&b.quads.floats, px) > crate::run::GLYPH_PIXELS {
                break;
            }
            long.push('M');
        }
        let tall = words("tall", &long, 2.0, 30.0, px);
        let rows = [
            vec![
                words("t1", "Ab", 2.0, 2.0, 8.0),
                big.clone(),
                words("t2", "cd", 6.0, 8.0, 8.0),
            ],
            vec![
                rect("a", [0.0, 28.0, 6.0, 6.0], "red"),
                tall,
                rect("b", [10.0, 40.0, 6.0, 6.0], "lime"),
            ],
        ];
        let (_g, mut r) = gpu_at(256, 128);
        for layers in rows {
            let together = first_frames(&mut r, &run_scene(&layers), 1);
            assert!(runs_of(&r).iter().all(|(m, _)| m.len() == 1), "{layers:?}");
            let mut apart = Vec::new();
            for (k, l) in layers.iter().enumerate() {
                apart.push(l.clone());
                apart.push(
                    json!({"name": format!("x{k}"), "text": "", "font_family": "pixel",
                                  "blink": {}}),
                );
            }
            assert!(
                first_frames(&mut r, &run_scene(&apart), 1) == together,
                "{layers:?}"
            );
        }
        let around = |l: &Value| {
            [
                words("t1", "Ab", 2.0, 2.0, 8.0),
                l.clone(),
                words("t2", "cd", 6.0, 8.0, 8.0),
            ]
        };
        let small = rect("big", [4.0, 4.0, 4.0, 4.0], "navy");
        for (from, to, side) in [(&big, &small, 4.0), (&small, &big, side)] {
            first_frames(&mut r, &run_scene(&around(from)), 1);
            let compiled = r.compiled();
            for field in ["width", "height"] {
                let at = Address {
                    layer: "big",
                    effect: None,
                    field,
                };
                r.set_layer(&at, Given::Json(&json!(side))).unwrap();
            }
            r.set_time(Some(0.0));
            let live = still(&mut r);
            let grouped = runs_of(&r);
            if side > 4.0 {
                assert_eq!(
                    r.compiled(),
                    compiled,
                    "the texts that leave compile nothing"
                );
            }
            assert!(
                first_frames(&mut r, &run_scene(&around(to)), 1) == live,
                "{side}"
            );
            assert_eq!(runs_of(&r), grouped, "{side}");
            assert_eq!(grouped.len(), if side > 4.0 { 3 } else { 1 }, "{side}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_text_whose_cover_a_pass_reads_draws_alone() {
        // A pass reads t2's words through `t2_cover`, which lays them out
        // into a texture of their own, so t2 draws alone between runs of
        // the members beside it, as it draws apart.
        let layers = members();
        let doc = |layers: &[Value]| {
            let mut sc = run_scene(layers);
            sc.passes[0].fragment = "void main(){ float c = t2_cover(uv * t2_size, 16.0); \
                                 gl_FragColor = vec4(0.5, 0.25 + 0.5 * c, 0.75, 1.0); }"
                .to_string();
            sc
        };
        let (_g, mut r) = gpu_at(128, 64);
        let run = first_frames(&mut r, &doc(&layers), 1);
        let s = |n: &[&str]| n.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            runs_of(&r),
            [
                (s(&["a", "t1", "b"]), Mode::Mixed),
                (s(&["t2"]), Mode::Alone),
                (s(&["c", "t3", "t4"]), Mode::Mixed)
            ]
        );
        let mut apart = Vec::new();
        for (k, l) in layers.iter().enumerate() {
            apart.push(l.clone());
            apart.push(
                json!({"name": format!("x{k}"), "text": "", "font_family": "pixel",
                              "blink": {}}),
            );
        }
        assert!(worst(&run, &first_frames(&mut r, &doc(&apart), 1)) <= 1);
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_changed_member_of_a_run_of_a_text_hands_the_engine_its_own_vertices() {
        // A run of a rect, two glyphs of t1 and a rect. A change to one
        // member hands the engine that member's vertices alone, changes to
        // the two rects one span from the first to the last, and new words
        // of another count every vertex of the run.
        use crate::draw::{Arg, Call, CallKind};
        let call = |name: &str, x: f64| {
            let mut args = vec![Arg::None; CallKind::Rect.keys().len()];
            args[0] = Arg::Num(x);
            args[1] = Arg::Num(2.0);
            args[2] = Arg::Num(6.0);
            args[3] = Arg::Num(6.0);
            Call {
                kind: CallKind::Rect,
                layer: name.to_string(),
                args,
            }
        };
        let layers = [
            json!({"name": "a", "shape": "rect", "x": 2, "y": 2, "width": 6, "height": 6}),
            words("t1", "Ab", 10.0, 2.0, 8.0),
            json!({"name": "b", "shape": "rect", "x": 30, "y": 2, "width": 6, "height": 6}),
        ];
        let quad = 4 * 4 * crate::run::ATTRIBUTES;
        let (_g, mut r) = gpu_at(64, 32);
        first_frames(&mut r, &run_scene(&layers), 1);
        assert_eq!(runs_of(&r).len(), 1);
        let x = Address {
            layer: "t1",
            effect: None,
            field: "x",
        };
        type Change = Box<dyn Fn(&mut Renderer)>;
        let rows: [(&str, Change, usize); 5] = [
            (
                "t1 x",
                Box::new(move |r| r.set_layer(&x, Given::Json(&json!(12))).unwrap()),
                2 * quad,
            ),
            (
                "b",
                Box::new(move |r| {
                    assert_eq!(r.update_calls(&[call("b", 31.0)]).unwrap().changed, [0])
                }),
                quad,
            ),
            (
                "a and b",
                Box::new(move |r| {
                    let taken = r.update_calls(&[call("a", 3.0), call("b", 32.0)]).unwrap();
                    assert_eq!(taken.changed, [0, 1]);
                }),
                4 * quad,
            ),
            ("t1 words", Box::new(|r| set_words(r, "t1", "Cd")), 2 * quad),
            (
                "t1 more words",
                Box::new(|r| set_words(r, "t1", "Cde")),
                5 * quad,
            ),
        ];
        for (what, change, floats) in rows {
            let before = r.run_floats.get();
            change(&mut r);
            assert_eq!(r.run_floats.get() - before, floats, "{what}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_layer_builds_its_colour_drawing_at_the_first_words_that_hold_a_colour_glyph() {
        // The fixture beside the scene, as a colour font asset, and as the
        // emoji font that the built-in font falls back to.
        let dir = crate::testing::TempDir::new("engine-colour");
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/fixtures/fonts/NotoColorEmoji-Smiles.ttf"
        );
        std::fs::copy(fixture, dir.join("smiles.ttf")).unwrap();
        std::fs::copy(fixture, dir.join(crate::emoji::FILE)).unwrap();
        let load = |font: Option<&str>, words: &str| {
            let mut layer = json!({"name": "label", "text": words, "x": 0, "y": 0,
                                   "font_size": 20});
            if let Some(f) = font {
                layer["font_family"] = json!(f);
            }
            let mut sc = Scene::from_dict(
                &json!({
                    "version": 1,
                    "assets": {"emoji": {"file": "smiles.ttf"}},
                    "passes": [{"name": "main",
                        "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                        "layers": [layer]}],
                }),
                None,
                None,
            )
            .unwrap();
            sc.load_assets(&dir).unwrap();
            sc
        };
        let (_g, mut r) = gpu();
        r.atlas_cache = Some(dir.join("atlas"));
        // Whether a program reads the colour atlas, whether the atlas exists,
        // and whether the frame shows colour, once a frame with the time
        // pinned waits for the words.
        let drawing = |r: &mut Renderer| {
            r.set_time(Some(0.0));
            let frame = r.render(None).unwrap();
            let shown = frame.chunks_exact(4).any(|p| p[0] != p[2]);
            let size = CString::new(text::internal(GLSL_NAME, "_emoji_size")).unwrap();
            let read = unsafe { ffi::ghee_mp_reads(r.handle, size.as_ptr()) };
            let held = r
                .scene_assets
                .borrow()
                .iter()
                .any(|n| n == "ghee_label_emoji");
            (read, held, shown)
        };
        // A layer in the colour font, and one in the built-in font.
        for font in [Some("emoji"), None] {
            // Words with no colour glyph build no colour drawing.
            assert_eq!(r.setup(&load(font, " ")).unwrap(), "");
            assert_eq!(drawing(&mut r), (false, false, false), "{font:?}");
            // Live words build it with no new setup.
            set_words(&r, "label", "\u{1f600}");
            assert_eq!(drawing(&mut r), (true, true, true), "{font:?}");
            assert!(r.text_errors().is_empty(), "{:?}", r.text_errors());
            // Words that drop the colour glyph keep the drawing and show none.
            set_words(&r, "label", " ");
            assert_eq!(drawing(&mut r), (true, true, false), "{font:?}");
            // A new setup builds it again only for words that need it.
            assert_eq!(r.setup(&load(font, " ")).unwrap(), "");
            assert_eq!(drawing(&mut r), (false, false, false), "{font:?}");
            assert_eq!(r.setup(&load(font, "\u{1f600}")).unwrap(), "");
            assert_eq!(drawing(&mut r), (true, true, true), "{font:?}");
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_new_colour_glyph_uploads_its_cell_at_every_level_as_a_whole_atlas_does() {
        // Five layers of the words, at 1, 1/2, 1/4, 1/8 and 1/16 of the
        // bitmaps' 109 pixels per em, each read from its own mip level.
        let (w, h) = (400, 224);
        let names: Vec<String> = (0..5).map(|k| format!("l{k}")).collect();
        let layers: Vec<Value> = (0..5)
            .map(|k| {
                let s = 109.0 / f64::from(1 << k);
                let at = [
                    [4.0, 4.0],
                    [4.0, 140.0],
                    [150.0, 140.0],
                    [230.0, 140.0],
                    [280.0, 140.0],
                ][k];
                json!({"name": names[k], "text": " ", "font_family": "emoji", "render": "slug",
                       "x": at[0], "y": at[1], "font_size": s})
            })
            .collect();
        let mut sc = Scene::from_dict(
            &json!({
                "version": 1,
                "assets": {"emoji": {"file": "NotoColorEmoji-Smiles.ttf"}},
                "passes": [{"name": "main",
                    "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
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
            for name in &names {
                let blank = atlas::Field {
                    width: 512,
                    height: 16,
                    texels: vec![0; 512 * 16 * 4],
                    ink: [0.0; 4],
                    pad: 0.0,
                };
                let mut colours = r.colours.borrow_mut();
                let slot = colours.entry(name.clone()).or_default();
                slot.atlas.insert(u32::MAX, &blank).unwrap();
                drop(colours);
                set_words(r, name, words);
            }
        };
        let size = |r: &Renderer| {
            let c = r.colours.borrow();
            (c["l0"].atlas.width, c["l0"].atlas.height)
        };
        start(&mut r, "\u{1f600}");
        let first = size(&r);
        for name in &names {
            set_words(&r, name, "\u{1f600}\u{1f642}");
        }
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
                "passes": [{"name": "main",
                    "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                    "layers": [{"name": "label", "text": words, "font_family": "sans",
                                "x": 2, "y": 1, "font_size": 40}]}],
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
        set_words(&r, "label", "AVZ");
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
        set_words(&r, "label", "AVQ");
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
        set_words(&r, "label", "AVQ");
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
        sc.passes[0].layers[0].as_text_mut().unwrap().render = Render::Slug;
        assert_eq!(r.setup(&sc).unwrap(), "");
        assert!(r.atlases.borrow().is_empty());
        // A hidden layer takes no atlas either.
        let mut sc = mtsdf_scene("AV");
        sc.passes[0].layers[0].as_text_mut().unwrap().visible = false;
        assert_eq!(r.setup(&sc).unwrap(), "");
        assert!(r.atlases.borrow().is_empty());
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn layers_of_near_sizes_share_an_atlas_and_a_far_size_takes_its_own() {
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let near = TextLayer::atlas_px_em(40.0);
        let far = TextLayer::atlas_px_em(100.0);
        assert_eq!(TextLayer::atlas_px_em(44.0), near);
        assert_ne!(far, near);
        let layer = |name: &str, px: f64, visible: bool| {
            json!({"name": name, "text": "AV", "font_family": "sans", "y": 1,
                   "font_size": px, "visible": visible})
        };
        let mut sc = Scene::from_dict(
            &json!({
                "version": 1,
                "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
                "passes": [{"name": "main",
                    "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                    "layers": [layer("a", 40.0, true), layer("b", 44.0, true),
                               layer("c", 100.0, true), layer("d", 70.0, false)]}],
            }),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(fonts).unwrap();
        let cache = crate::testing::TempDir::new("engine-steps");
        let (_g, mut r) = gpu();
        r.atlas_cache = Some(cache.join("atlas"));
        assert_eq!(r.setup(&sc).unwrap(), "");
        let mut slots: Vec<(u32, Vec<String>)> = r
            .atlases
            .borrow()
            .iter()
            .map(|s| (s.px_em, s.textures.iter().map(|t| t.0.clone()).collect()))
            .collect();
        slots.sort();
        let tex = |name: &str| text::internal(name, "_atlas");
        let mut want = vec![(near, vec![tex("a"), tex("b")]), (far, vec![tex("c")])];
        want.sort();
        assert_eq!(slots, want);
    }

    /// Each pixel of the scene holds its column in red and its row from the
    /// top in green, which read back exactly.
    const POSITION: &str = "void main(){ vec2 p = floor(uv * iResolution.xy); \
        gl_FragColor = vec4(p / 255.0, 0.0, 1.0); }";

    /// The frame's pixel (x, y) of a w x h frame shows the scene pixel that a
    /// clockwise turn of `degrees` puts there. The scaled readback takes the
    /// same turn, at two frame pixels to a side.
    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn every_readback_turns_the_frame_clockwise() {
        let (w, h) = (8usize, 4usize);
        for (degrees, scene_size) in [(0, (8, 4)), (90, (4, 8)), (180, (8, 4)), (270, (4, 8))] {
            let (_g, mut r) = gpu_at(w, h);
            r.set_rotate(degrees).unwrap();
            assert_eq!(r.scene_size(), scene_size, "{degrees}");
            let sc = scene_of(json!([{"name": "main", "fragment": POSITION}]));
            assert_eq!(r.setup(&sc).unwrap(), "");
            let frame = r.render(None).unwrap().to_vec();
            let scaled = r.render_scaled(None, (2 * w as i32, 2 * h as i32)).unwrap();
            for (x, y) in (0..h).flat_map(|y| (0..w).map(move |x| (x, y))) {
                let (sx, sy) = match degrees {
                    0 => (x, y),
                    90 => (y, w - 1 - x),
                    180 => (w - 1 - x, h - 1 - y),
                    _ => (h - 1 - y, x),
                };
                let want = (sx as f64, sy as f64);
                let at = |f: &[u8], fw: usize, x: usize, y: usize| {
                    (px(f, fw, x, y, 0), px(f, fw, x, y, 1))
                };
                assert_eq!(at(&frame, w, x, y), want, "{degrees} at ({x}, {y})");
                for (dx, dy) in [(0, 0), (1, 1)] {
                    let got = at(&scaled, 2 * w, 2 * x + dx, 2 * y + dy);
                    assert_eq!(got, want, "{degrees} scaled at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn a_turn_is_a_quarter_turn_clockwise() {
        for (degrees, turns) in [(0, 0), (90, 1), (180, 2), (270, 3)] {
            assert_eq!(quarter_turns(degrees).unwrap(), turns);
        }
        assert_eq!(
            quarter_turns(45).unwrap_err().to_string(),
            "rotate takes 0, 90, 180 or 270, got 45"
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_turn_comes_before_the_first_scene() {
        let (_g, mut r) = gpu();
        assert_eq!(r.setup(&flat_scene()).unwrap(), "");
        assert_eq!(
            r.set_rotate(90).unwrap_err().to_string(),
            "rotate needs a renderer with no scene yet"
        );
        assert_eq!(r.scene_size(), (W as i32, H as i32));
    }
}
