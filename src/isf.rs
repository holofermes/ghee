//! Import an ISF (Interactive Shader Format) shader as a ghee scene or filter.
//!
//! An ISF file is a fragment shader with a JSON header in a leading block
//! comment. The header holds INPUTS, which are typed uniforms plus image and
//! audio samplers. It also holds PASSES, which are the render targets and the
//! size each one draws at. The body reads ISF's own built-ins.
//!
//! A generated shim maps ISF onto the ghee preamble:
//!
//! ```text
//! RENDERSIZE -> resolution         isf_FragNormCoord -> v_texCoord
//! TIME       -> time               an image input / PASSES target -> iChannelN
//! IMG_*      -> texture2D()        an audio input -> a ghee array
//! PASSINDEX  -> a per-pass const int, so the compiler folds the branches
//! ```
//!
//! ISF puts one body in every pass and branches on PASSINDEX. Each ghee pass
//! takes that same body with its own PASSINDEX constant.
//!
//! The body loses two things. ISF targets GLSL 1.20, which allows a file-scope
//! initializer that reads a uniform. ESSL 1.00 refuses one, so the import hoists
//! it into main. IMG_SIZE cannot be a function, because a sampler2D carries no
//! size. IMG_SIZE(x) becomes the constant x_isf_size.
//!
//! ghee refuses three things, by name and reason. A FLOAT (32-bit) pass buffer
//! saturates silently in ghee's 8-bit buffers. A custom ISF vertex shader writes
//! varyings ghee's fixed vertex shader does not emit. IMPORTED loads an external
//! image.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::scene::{
    py_repr, py_str, ArrayInput, Channel, Param, ParamMap, ParamValue, Pass, Scene,
};

// An audio input that names no MAX takes this many bins.
const AUDIO_BINS: i64 = 256;

// VC4 misses 30fps past this many fetches per fragment.
const FETCH_BUDGET: i64 = 200;

// Texture units on VC4, the smallest GPU ghee targets.
const VC4_UNITS: usize = 8;

// The names ghee's own preamble owns. An ISF input or target with one of these
// shadows it, so the import renames it here and in the body.
const RESERVED: [&str; 3] = ["time", "resolution", "v_texCoord"];

/// The render size to_scene and to_filter take when the caller names none.
#[allow(dead_code)]
pub const DEFAULT_SIZE: (i64, i64) = (320, 240);

const GLSL_TYPES: [&str; 15] = [
    "float", "int", "bool", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3",
    "bvec4", "mat2", "mat3", "mat4",
];

const IMG_ACCESSORS: [&str; 5] = [
    "vec4 IMG_PIXEL(vec2 sz, sampler2D s, vec2 p) { return texture2D(s, p / sz); }",
    "vec4 IMG_NORM_PIXEL(sampler2D s, vec2 p) { return texture2D(s, p); }",
    "vec4 IMG_THIS_PIXEL(sampler2D s) { return texture2D(s, v_texCoord); }",
    "vec4 IMG_THIS_NORM_PIXEL(sampler2D s) { return texture2D(s, v_texCoord); }",
    "vec4 IMG_NORM_THIS_PIXEL(sampler2D s) { return texture2D(s, v_texCoord); }",
];

// What an unbound image input reads in scene mode. The pattern is a checkered
// gradient with a disc, so an imported filter has something to act on.
pub const SOURCE_PASS: &str = "isf_source";

const SOURCE_CODE: &str = "void main() {
    vec2 uv = v_texCoord;
    vec2 g = floor(uv * 8.0);
    float checker = mod(g.x + g.y, 2.0);
    float disc = smoothstep(0.30, 0.28, distance(uv, vec2(0.5)));
    vec3 col = vec3(uv.x, uv.y, 1.0 - uv.y) * (0.4 + 0.45 * checker);
    gl_FragColor = vec4(mix(col, vec3(1.0, 0.85, 0.2), disc), 1.0);
}
";

/// A refusal, naming the reason. `convert` puts the file name in front.
fn isf_error(msg: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{msg}")
}

/// A string quoted the way the error strings quote source text.
fn repr(s: &str) -> String {
    py_repr(&Value::String(s.to_string()))
}

/// True when a JSON value counts as empty: null, false, 0, "", [], {}.
fn falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f == 0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// The value at key. A falsy value counts as an absent one.
fn truthy<'a>(d: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    d.get(key).filter(|v| !falsy(v))
}

/// An ISF source, a name for it, and whether a sibling .vs file exists. A read
/// failure carries the bare strerror, which the CLI reports.
pub fn read_isf(path: &str) -> Result<(String, String, bool)> {
    if path.starts_with("http://") || path.starts_with("https://") {
        bail!("{path}: reading an ISF over http(s) needs an HTTP client this build has none of");
    }
    let raw = std::fs::read(path).map_err(|e| {
        let s = e.to_string();
        anyhow!(
            "{}",
            s.split_once(" (os error ").map_or(s.as_str(), |(m, _)| m)
        )
    })?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (stem, _) = splitext(path);
    Ok((
        text,
        basename(path).to_string(),
        std::path::Path::new(&format!("{stem}.vs")).exists(),
    ))
}

/// The leading /*{ ... }*/ JSON header and the GLSL after it.
pub fn split_header(text: &str) -> Result<(Map<String, Value>, String)> {
    let src: Vec<char> = text.chars().collect();
    let found = find_from(&src, "/*", 0).and_then(|start| {
        find_from(&src, "*/", start + 2).map(|end| (slice(&src, start + 2, end), end + 2))
    });
    let Some((header, end)) = found.filter(|(h, _)| h.trim_start().starts_with('{')) else {
        return Err(isf_error(
            "no leading /*{ ... }*/ JSON header, so it is not ISF",
        ));
    };
    let meta: Value = serde_json::from_str(&header)
        .map_err(|e| isf_error(format!("the JSON header does not parse: {e}")))?;
    let Value::Object(meta) = meta else {
        return Err(isf_error(format!(
            "the JSON header needs an object, got {}",
            py_repr(&meta)
        )));
    };
    Ok((meta, slice(&src, end, src.len())))
}

/// An ISF WIDTH or HEIGHT. It is a number, or an equation over $WIDTH, $HEIGHT
/// and $<inputName>. The operators are + - * / and floor, ceil, abs, min, max.
/// The result rounds down to a positive int, which is what a render target takes.
pub fn eval_size(expr: &Value, values: &HashMap<String, f64>) -> Result<i64> {
    let v = match expr {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => Expr::new(s, values).parse()?,
        _ => {
            return Err(isf_error(format!(
                "size {} is neither a number nor an equation",
                py_repr(expr)
            )))
        }
    };
    if !v.is_finite() {
        return Err(isf_error(format!(
            "size equation {} does not evaluate to a number",
            py_repr(expr)
        )));
    }
    Ok(1.max(v as i64))
}

/// Move every non-const file-scope initializer to the top of main. ESSL 1.00
/// takes a constant expression there, and GLSL 1.20 takes any expression.
/// Returns the body and the hoisted names.
pub fn hoist_globals(body: &str) -> Result<(String, Vec<String>)> {
    let src: Vec<char> = body.chars().collect();
    let mask = mask_code(&src, true);
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut assigns: Vec<String> = Vec::new();
    let (mut depth, mut par, mut start) = (0i64, 0i64, 0usize);
    for (i, c) in mask.iter().enumerate() {
        match c {
            '(' | '[' => par += 1,
            ')' | ']' => par -= 1,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    start = i + 1;
                    par = 0;
                }
            }
            ';' if depth == 0 && par == 0 => {
                // A statement slice starts after the last `;` or `}`, so it can
                // carry a comment block the rewrite keeps.
                let seg = &mask[start..i];
                let head = seg
                    .iter()
                    .position(|c| !c.is_whitespace())
                    .unwrap_or(seg.len());
                if let Some((decl, sets)) = split_decl(&slice(&mask, start + head, i)) {
                    edits.push((start + head, i + 1, decl));
                    for (nm, rhs) in sets {
                        assigns.push(format!("    {nm} = {rhs};"));
                        names.push(nm);
                    }
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if assigns.is_empty() {
        return Ok((body.to_string(), Vec::new()));
    }
    let Some(end) = find_main(&mask) else {
        return Err(isf_error(
            "no void main() to hoist the file-scope initializers into",
        ));
    };
    edits.push((end, end, format!("\n{}", assigns.join("\n"))));
    edits.sort_by(|a, b| b.cmp(a));
    let mut out = src;
    for (a, b, text) in edits {
        let mut next: Vec<char> = out[..a].to_vec();
        next.extend(text.chars());
        next.extend_from_slice(&out[b..]);
        out = next;
    }
    Ok((out.into_iter().collect(), names))
}

/// The texture fetches one fragment makes. Each IMG_* or texture2D call site
/// counts once for every trip of a literal-bounded for loop around it. A loop
/// bounded by a uniform counts one trip, so the result is a floor.
pub fn fetch_estimate(body: &str) -> Result<i64> {
    let src: Vec<char> = body.chars().collect();
    let mask = mask_code(&src, true);
    let (mut total, mut mult) = (0i64, 1i64);
    let mut stack: Vec<(i64, i64)> = Vec::new();
    let mut depth = 0i64;
    let mut i = 0usize;
    while i < mask.len() {
        let c = mask[i];
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            while let Some((_, t)) = stack.last().copied().filter(|(d, _)| *d > depth) {
                stack.pop();
                mult = mult.checked_div(t).ok_or_else(|| {
                    isf_error("a for loop bounded by 0 leaves no fetch estimate to divide")
                })?;
            }
        } else if let Some(end) = for_match(&mask, i) {
            let close = match_paren(&mask, end - 1);
            let brace = find_from(&mask, "{", close);
            if brace.is_some_and(|b| mask[close + 1..b].iter().all(|c| c.is_whitespace())) {
                let t = trips(&mask[end..close]);
                stack.push((depth + 1, t));
                mult = mult.saturating_mul(t);
            }
            i = close;
        } else if let Some(end) = fetch_match(&mask, i) {
            total = total.saturating_add(mult);
            i = end - 1;
        }
        i += 1;
    }
    Ok(total)
}

/// Import ISF source as a ghee Scene. Returns the scene and any warnings.
pub fn to_scene(
    text: &str,
    name: &str,
    size: (i64, i64),
    vertex_shader: bool,
) -> Result<(Scene, Vec<String>)> {
    match convert(text, name, size, false, vertex_shader)? {
        (Built::Scene(scene), warnings) => Ok((scene, warnings)),
        (Built::Filter(_), _) => unreachable!("built as a scene"),
    }
}

/// Import a single-pass ISF filter as a ghee filter definition, the object a
/// `<name>.filter.json` holds. Its one image input becomes iChannel0. ghee hands
/// a filter the frame so far there.
pub fn to_filter(
    text: &str,
    name: &str,
    size: (i64, i64),
    vertex_shader: bool,
) -> Result<(Value, Vec<String>)> {
    match convert(text, name, size, true, vertex_shader)? {
        (Built::Filter(flt), warnings) => Ok((flt, warnings)),
        (Built::Scene(_), _) => unreachable!("built as a filter"),
    }
}

/// A file, pass or filter name from an ISF title. 'Radial Spectrogram.fs'
/// becomes 'radial_spectrogram'.
pub fn slug(name: &str) -> String {
    let (stem, _) = splitext(basename(name));
    let mut out = String::new();
    let mut gap = false;
    for c in stem.to_lowercase().chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            if gap {
                out.push('_');
                gap = false;
            }
            out.push(c);
        } else {
            gap = true;
        }
    }
    if gap {
        out.push('_');
    }
    match out.trim_matches('_') {
        "" => "isf".to_string(),
        s => s.to_string(),
    }
}

struct Input {
    name: String,
    ty: Value,
    param: Option<Param>,
    /// The array an audio input becomes.
    size: Option<(i64, i64)>,
}

/// Everything the shim names: the samplers and the size of each one.
#[derive(Default)]
struct Sized {
    channels: Vec<(String, Option<String>)>,
    arrays: Vec<String>,
    sizes: Vec<(String, (i64, i64))>,
}

impl Sized {
    fn set_size(&mut self, name: &str, size: (i64, i64)) {
        match self.sizes.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = size,
            None => self.sizes.push((name.to_string(), size)),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.sizes.iter().any(|(n, _)| n == name)
    }
}

enum Built {
    Scene(Scene),
    Filter(Value),
}

fn convert(
    text: &str,
    src: &str,
    size: (i64, i64),
    as_filter: bool,
    vertex_shader: bool,
) -> Result<(Built, Vec<String>)> {
    build(text, src, size, as_filter, vertex_shader).map_err(|e| anyhow!("{src}: {e}"))
}

fn build(
    text: &str,
    src: &str,
    size: (i64, i64),
    as_filter: bool,
    vertex_shader: bool,
) -> Result<(Built, Vec<String>)> {
    let (meta, body) = split_header(text)?;
    refuse_unrunnable(&meta, &body, vertex_shader)?;
    let mut warn: Vec<String> = Vec::new();

    let mut inputs = inputs_of(&meta)?;
    let mut pass_meta = passes_of(&meta)?;
    let (body, renames) = rename_reserved(&body, &mut inputs, &mut pass_meta);
    warn.extend(
        renames
            .iter()
            .map(|(a, b)| format!("'{a}' is a ghee preamble name, renamed to '{b}'")),
    );

    let count = pass_meta.len().max(1);
    let (names, targets) = pass_names(&pass_meta, count)?;
    let (resolved, pass_sizes) = pass_sizes(&pass_meta, size, &inputs, names.len(), &mut warn)?;

    let images: Vec<&Input> = inputs
        .iter()
        .filter(|i| i.ty.as_str() == Some("image"))
        .collect();
    let arrays: Vec<&Input> = inputs.iter().filter(|i| i.size.is_some()).collect();
    let mut params = ParamMap::new();
    for i in &inputs {
        if let Some(p) = &i.param {
            params.insert(i.name.clone(), p.clone());
        }
    }
    let code = mask_code(&body.chars().collect::<Vec<char>>(), false);
    let refs: Vec<String> = targets
        .iter()
        .flatten()
        .filter(|t| search_word(&code, t).is_some())
        .cloned()
        .collect();

    if as_filter {
        refuse_unfilterable(&pass_meta, &images, &arrays, &refs)?;
    }

    let mut sized = Sized {
        arrays: arrays.iter().map(|a| a.name.clone()).collect(),
        ..Sized::default()
    };
    for i in &images {
        let producer = (!as_filter).then(|| SOURCE_PASS.to_string());
        sized.channels.push((i.name.clone(), producer));
        sized.set_size(&i.name, size);
    }
    for t in &refs {
        sized.channels.push((t.clone(), Some(t.clone())));
        let at = names.iter().position(|n| n == t).unwrap_or(0);
        sized.set_size(t, resolved[at]);
    }
    for a in &arrays {
        sized.set_size(&a.name, a.size.unwrap_or(size));
    }

    let body = rewrite_img(&body, &sized)?;
    let (body, hoisted) = hoist_globals(&body)?;
    if !hoisted.is_empty() {
        warn.push(format!(
            "hoisted file-scope initializer(s) into main: {}",
            hoisted.join(", ")
        ));
    }

    let units = sized.channels.len() + sized.arrays.len();
    if units > VC4_UNITS {
        warn.push(format!(
            "binds {units} textures; VC4-class GPUs (Zero..Pi 4) have {VC4_UNITS}"
        ));
    }
    let fetches = fetch_estimate(&body)?;
    if fetches > FETCH_BUDGET {
        warn.push(format!(
            "about {fetches} texture fetches per fragment; VC4 misses 30fps \
             past roughly 100"
        ));
    }
    if search_word(&body.chars().collect::<Vec<char>>(), "DATE").is_some() {
        warn.push("DATE is a fixed vec4(0, 0, 0, time); ghee has no calendar clock".to_string());
    }

    let title = slug(src);
    if as_filter {
        let mut flt = Map::new();
        flt.insert("name".to_string(), Value::from(title));
        flt.insert(
            "code".to_string(),
            Value::from(format!("{}\n{body}", shim(src, 0, &sized))),
        );
        let mut declared = Map::new();
        for (n, p) in params.iter() {
            declared.insert(n.clone(), p.to_dict());
        }
        flt.insert("params".to_string(), Value::Object(declared));
        let desc = description(&meta);
        if !desc.is_empty() {
            flt.insert("description".to_string(), Value::from(desc));
        }
        return Ok((Built::Filter(Value::Object(flt)), warn));
    }

    let mut passes: Vec<Pass> = Vec::new();
    if !images.is_empty() {
        passes.push(Pass::of(SOURCE_PASS, SOURCE_CODE)?);
        warn.push(format!(
            "image input(s) {} have no producer; bound to a generated \
             '{SOURCE_PASS}' test pattern pass",
            images
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    for (i, nm) in names.iter().enumerate() {
        let mut declared = Vec::new();
        for a in &arrays {
            declared.push(ArrayInput::new(
                &a.name,
                Value::from(a.size.unwrap_or(size).0),
            )?);
        }
        passes.push(Pass::new(
            nm,
            &format!("{}\n{body}", shim(src, i, &sized)),
            declared,
            sized
                .channels
                .iter()
                .map(|(_, b)| Channel::new(b.as_deref()))
                .collect(),
            ParamMap::new(),
            "",
            pass_sizes[i].map(|(w, h)| Value::from(vec![w, h])),
        )?);
    }
    let scene = Scene::new(passes, Value::from(1), Vec::new(), params, None)?;
    Ok((Built::Scene(scene), warn))
}

fn refuse_unrunnable(meta: &Map<String, Value>, body: &str, vertex_shader: bool) -> Result<()> {
    if vertex_shader {
        return Err(isf_error(
            "ships a custom ISF vertex shader (.vs); ghee's vertex shader is \
             fixed and emits v_texCoord only, so this cannot link",
        ));
    }
    if let Some(name) = find_varying(&body.chars().collect::<Vec<char>>()) {
        return Err(isf_error(format!(
            "reads the custom varying '{name}', which only an ISF vertex shader \
             can write; ghee's vertex shader emits v_texCoord only"
        )));
    }
    if let Some(imported) = truthy(meta, "IMPORTED") {
        let which = match imported {
            Value::Object(o) => o.keys().cloned().collect::<Vec<_>>().join(", "),
            v => py_str(v),
        };
        return Err(isf_error(format!(
            "IMPORTED loads external image(s) ({which}); ghee scenes have no \
             image inputs, only pass buffers and host-fed arrays"
        )));
    }
    let raw = match truthy(meta, "PASSES") {
        Some(Value::Array(a)) => a.as_slice(),
        _ => &[],
    };
    let floats: Vec<String> = raw
        .iter()
        .enumerate()
        .filter(|(_, p)| p.get("FLOAT").is_some_and(|f| !falsy(f)))
        .map(|(i, p)| match p.get("TARGET").filter(|t| !falsy(t)) {
            Some(t) => py_str(t),
            None => format!("pass {i}"),
        })
        .collect();
    if !floats.is_empty() {
        return Err(isf_error(format!(
            "buffer(s) {} are FLOAT; ghee's pass buffers are 8-bit RGB, so an \
             accumulator saturates within a few seconds and the scene freezes \
             without ever reporting an error",
            floats.join(", ")
        )));
    }
    Ok(())
}

fn refuse_unfilterable(
    pass_meta: &[Map<String, Value>],
    images: &[&Input],
    arrays: &[&Input],
    refs: &[String],
) -> Result<()> {
    let tail = "Import it as a scene instead";
    if pass_meta.len() > 1 {
        return Err(isf_error(format!(
            "has {} passes; a ghee filter is one pass. {tail}",
            pass_meta.len()
        )));
    }
    if images.len() != 1 {
        let what = if images.is_empty() {
            "no image input".to_string()
        } else {
            format!("{} image inputs", images.len())
        };
        return Err(isf_error(format!(
            "has {what}; a ghee filter reads exactly one, iChannel0 (the frame \
             so far). {tail}"
        )));
    }
    if !refs.is_empty() {
        return Err(isf_error(format!(
            "reads its own buffer ({}) for feedback; a ghee filter binds \
             iChannel0 only. {tail}",
            refs.join(", ")
        )));
    }
    if !arrays.is_empty() {
        return Err(isf_error(format!(
            "reads audio ({}); a ghee filter declares no arrays. {tail}",
            arrays
                .iter()
                .map(|a| a.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(())
}

fn inputs_of(meta: &Map<String, Value>) -> Result<Vec<Input>> {
    let raw = match truthy(meta, "INPUTS") {
        None => &[] as &[Value],
        Some(Value::Array(a)) => a.as_slice(),
        Some(v) => {
            return Err(isf_error(format!(
                "INPUTS needs a list, got {}",
                py_repr(v)
            )))
        }
    };
    let mut out = Vec::new();
    for d in raw {
        let Value::Object(d) = d else {
            return Err(isf_error(format!(
                "an INPUTS entry needs an object, got {}",
                py_repr(d)
            )));
        };
        let name = d.get("NAME").cloned().unwrap_or(Value::Null);
        let ty = d.get("TYPE").cloned().unwrap_or(Value::Null);
        if !name.as_str().is_some_and(is_ident) {
            return Err(isf_error(format!(
                "input name {} is not a GLSL identifier",
                py_repr(&name)
            )));
        }
        let name = py_str(&name);
        let mut inp = Input {
            name: name.clone(),
            ty: ty.clone(),
            param: None,
            size: None,
        };
        match ty.as_str() {
            Some("audio" | "audioFFT") => {
                // MAX true counts as 1, so the shader takes one bin.
                let n = match d.get("MAX") {
                    Some(Value::Bool(true)) => 1,
                    Some(v) if v.is_number() && !falsy(v) => v.as_f64().unwrap_or(0.0) as i64,
                    _ => AUDIO_BINS,
                };
                inp.size = Some((n, 1));
            }
            Some("image") => {}
            _ => inp.param = Some(param_of(&name, &ty, d)?),
        }
        out.push(inp);
    }
    Ok(out)
}

/// One INPUTS entry as a ghee param. An ISF color is RGBA. It becomes a vec4,
/// because ghee's color holds three components. A long becomes an int over the
/// range its VALUES span, because ghee has no labelled int dropdown.
fn param_of(name: &str, ty: &Value, d: &Map<String, Value>) -> Result<Param> {
    let default = d.get("DEFAULT");
    match ty.as_str() {
        Some("float") => Param::new(
            name,
            "float",
            Value::from(num(&[default, d.get("MIN")])),
            d.get("MIN").cloned(),
            d.get("MAX").cloned(),
            None,
            None,
        ),
        Some("bool" | "event") => Param::new(
            name,
            "bool",
            Value::from(default.is_some_and(|v| !falsy(v))),
            None,
            None,
            None,
            None,
        ),
        Some("long") => {
            let values: Vec<&Value> = match truthy(d, "VALUES") {
                Some(Value::Array(a)) => a.iter().filter(|v| v.is_number()).collect(),
                _ => Vec::new(),
            };
            // The pick keeps the first of two equal values, and the JSON number
            // it arrives as. An int range stays an int range.
            let pick = |better: fn(f64, f64) -> bool| -> Option<Value> {
                values
                    .iter()
                    .fold(None::<&&Value>, |best, v| {
                        let beats = best.is_none_or(|b| {
                            better(v.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0))
                        });
                        if beats {
                            Some(v)
                        } else {
                            best
                        }
                    })
                    .map(|v| (*v).clone())
            };
            let lo = pick(|a, b| a < b);
            let hi = pick(|a, b| a > b);
            let step = (!values.is_empty()).then(|| Value::from(1));
            Param::new(
                name,
                "int",
                Value::from(num(&[default, lo.as_ref()]) as i64),
                lo,
                hi,
                step,
                None,
            )
        }
        Some("color") => Param::new(
            name,
            "vec4",
            Value::from(vec_of(default, 4, &[0.0, 0.0, 0.0, 1.0])),
            None,
            None,
            None,
            None,
        ),
        Some("point2D") => Param::new(
            name,
            "vec2",
            Value::from(vec_of(default, 2, &[0.0, 0.0])),
            None,
            None,
            None,
            None,
        ),
        _ => Err(isf_error(format!(
            "input '{name}' has unsupported TYPE {}",
            py_repr(ty)
        ))),
    }
}

/// The first candidate that is a number, else 0.0.
fn num(candidates: &[Option<&Value>]) -> f64 {
    for c in candidates.iter().flatten() {
        if c.is_number() {
            return c.as_f64().unwrap_or(0.0);
        }
    }
    0.0
}

fn vec_of(value: Option<&Value>, n: usize, fallback: &[f64]) -> Vec<f64> {
    let Some(Value::Array(a)) = value else {
        return fallback.to_vec();
    };
    let mut v: Vec<f64> = a
        .iter()
        .map(|x| match x {
            Value::Bool(b) => f64::from(u8::from(*b)),
            x => x.as_f64().unwrap_or(0.0),
        })
        .collect();
    if v.len() < fallback.len() {
        v.extend_from_slice(&fallback[v.len()..]);
    }
    v.truncate(n);
    v
}

fn passes_of(meta: &Map<String, Value>) -> Result<Vec<Map<String, Value>>> {
    match truthy(meta, "PASSES") {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) if a.iter().all(Value::is_object) => Ok(a
            .iter()
            .filter_map(|p| p.as_object().cloned())
            .collect::<Vec<_>>()),
        Some(v) => Err(isf_error(format!(
            "PASSES needs a list of objects, got {}",
            py_repr(v)
        ))),
    }
}

/// An ISF name can also be a ghee preamble name, such as a target called `time`.
/// It shadows the uniform the shim maps TIME onto. Rename it, and every
/// reference the body makes to it, before anything else reads the body.
fn rename_reserved(
    body: &str,
    inputs: &mut [Input],
    pass_meta: &mut [Map<String, Value>],
) -> (String, Vec<(String, String)>) {
    let mut taken: Vec<String> = inputs.iter().map(|i| i.name.clone()).collect();
    taken.extend(
        pass_meta
            .iter()
            .filter_map(|p| p.get("TARGET").and_then(Value::as_str))
            .map(str::to_string),
    );
    let mut chars: Vec<char> = body.chars().collect();
    let mut renames = Vec::new();
    for old in taken {
        if !RESERVED.contains(&old.as_str()) && !is_channel_name(&old) {
            continue;
        }
        let new = format!("isf_{old}");
        chars = replace_word(&chars, &old, &new);
        for i in inputs.iter_mut() {
            if i.name == old {
                i.name = new.clone();
            }
        }
        for p in pass_meta.iter_mut() {
            if p.get("TARGET").and_then(Value::as_str) == Some(old.as_str()) {
                p.insert("TARGET".to_string(), Value::from(new.clone()));
            }
        }
        renames.push((old, new));
    }
    (chars.into_iter().collect(), renames)
}

fn pass_names(
    pass_meta: &[Map<String, Value>],
    count: usize,
) -> Result<(Vec<String>, Vec<Option<String>>)> {
    let (mut names, mut targets, mut used): (Vec<String>, Vec<Option<String>>, Vec<String>) =
        (Vec::new(), Vec::new(), Vec::new());
    for i in 0..count {
        let raw = pass_meta
            .get(i)
            .and_then(|p| p.get("TARGET"))
            .filter(|t| !t.is_null());
        let target = match raw {
            None => None,
            Some(t) if t.as_str().is_some_and(is_ident) => Some(py_str(t)),
            Some(t) => {
                return Err(isf_error(format!(
                    "PASSES[{i}] TARGET {} is not a GLSL identifier",
                    py_repr(t)
                )))
            }
        };
        let mut name = match &target {
            Some(t) => t.clone(),
            None if i == count - 1 => "out".to_string(),
            None => format!("pass{i}"),
        };
        while used.contains(&name) {
            name.push('_');
        }
        used.push(name.clone());
        names.push(name);
        targets.push(target);
    }
    Ok((names, targets))
}

type PassSizes = (Vec<(i64, i64)>, Vec<Option<(i64, i64)>>);

/// Evaluate each pass's WIDTH and HEIGHT equations against the target size.
/// Returns the resolved sizes, which the shim reports as x_isf_size, and the
/// sizes the scene declares. None is the renderer's own size.
fn pass_sizes(
    pass_meta: &[Map<String, Value>],
    size: (i64, i64),
    inputs: &[Input],
    count: usize,
    warn: &mut Vec<String>,
) -> Result<PassSizes> {
    let mut values: HashMap<String, f64> = HashMap::new();
    values.insert("WIDTH".to_string(), size.0 as f64);
    values.insert("HEIGHT".to_string(), size.1 as f64);
    for i in inputs {
        let Some(p) = &i.param else { continue };
        let v = match (p.ty.as_str(), &p.value) {
            ("float", ParamValue::Float(f)) => *f,
            ("int", ParamValue::Int(n)) => *n as f64,
            ("bool", ParamValue::Bool(b)) => f64::from(u8::from(*b)),
            _ => continue,
        };
        values.insert(i.name.clone(), v);
    }
    let (mut resolved, mut declared): PassSizes = (Vec::new(), Vec::new());
    let empty = Map::new();
    for i in 0..count {
        let p = pass_meta.get(i).unwrap_or(&empty);
        let w = match p.get("WIDTH") {
            Some(e) => eval_size(e, &values)?,
            None => size.0,
        };
        let h = match p.get("HEIGHT") {
            Some(e) => eval_size(e, &values)?,
            None => size.1,
        };
        resolved.push((w, h));
        declared.push(((w, h) != size).then_some((w, h)));
    }
    if declared.last().is_some_and(Option::is_some) {
        let last = resolved.len() - 1;
        warn.push(format!(
            "last pass asks for {}x{}; ghee's output pass renders at the render \
             size, so it takes {}x{}",
            resolved[last].0, resolved[last].1, size.0, size.1
        ));
        resolved[last] = size;
        declared[last] = None;
    }
    Ok((resolved, declared))
}

/// IMG_SIZE(x) becomes the constant x_isf_size. IMG_PIXEL(x, p) takes x's size
/// in front, because a sampler2D argument carries no size of its own.
fn rewrite_img(body: &str, sized: &Sized) -> Result<String> {
    let mut unknown: Vec<String> = Vec::new();
    let mut note = |name: &str| {
        if !sized.has(name) && !unknown.iter().any(|u| u == name) {
            unknown.push(name.to_string());
        }
    };
    let body = sub_code(body, "IMG_SIZE", ")", |name| {
        note(name);
        format!("{name}_isf_size")
    });
    let body = sub_code(&body, "IMG_PIXEL", ",", |name| {
        note(name);
        format!("IMG_PIXEL({name}_isf_size, {name},")
    });
    if !unknown.is_empty() {
        unknown.sort();
        return Err(isf_error(format!(
            "IMG_PIXEL/IMG_SIZE reads {}, which is not an image input or a pass \
             target, so its size is unknown",
            unknown.join(", ")
        )));
    }
    Ok(body)
}

/// A rewrite over the code only. The match runs against the body with its
/// comments blanked, so a call inside a comment stays as it is. `close` is what
/// follows the sampler argument.
fn sub_code(body: &str, call: &str, close: &str, mut repl: impl FnMut(&str) -> String) -> String {
    let src: Vec<char> = body.chars().collect();
    let mask = mask_code(&src, false);
    let mut hits: Vec<(usize, usize, String)> = Vec::new();
    let mut i = 0;
    while i < mask.len() {
        match call_match(&mask, i, call, close) {
            Some((end, name)) => {
                hits.push((i, end, name));
                i = end;
            }
            None => i += 1,
        }
    }
    let mut out = src;
    for (a, b, name) in hits.into_iter().rev() {
        let text = repl(&name);
        let mut next: Vec<char> = out[..a].to_vec();
        next.extend(text.chars());
        next.extend_from_slice(&out[b..]);
        out = next;
    }
    out.into_iter().collect()
}

/// `\b<call>\s*\(\s*([A-Za-z_]\w*)\s*<close>`: the end index and the identifier.
fn call_match(mask: &[char], i: usize, call: &str, close: &str) -> Option<(usize, String)> {
    if !boundary(mask, i) || !starts_with(mask, i, call) {
        return None;
    }
    let mut j = skip_ws(mask, i + call.chars().count());
    if mask.get(j) != Some(&'(') {
        return None;
    }
    j = skip_ws(mask, j + 1);
    let start = j;
    if !mask
        .get(j)
        .is_some_and(|c| c.is_ascii_alphabetic() || *c == '_')
    {
        return None;
    }
    while mask.get(j).is_some_and(|c| is_word(*c)) {
        j += 1;
    }
    let name = slice(mask, start, j);
    j = skip_ws(mask, j);
    if !starts_with(mask, j, close) {
        return None;
    }
    Some((j + close.chars().count(), name))
}

fn shim(src: &str, index: usize, sized: &Sized) -> String {
    let mut l = vec![
        format!("// ISF shim for '{src}', pass {index}"),
        "#define RENDERSIZE resolution".to_string(),
        "#define TIME time".to_string(),
        "#define TIMEDELTA DT".to_string(),
        "#define FRAMEINDEX int(time * 60.0)".to_string(),
        "#define DATE vec4(0.0, 0.0, 0.0, time)".to_string(),
        "#define isf_FragNormCoord v_texCoord".to_string(),
        "#define vv_FragNormCoord v_texCoord".to_string(),
        format!("const int PASSINDEX = {index};"),
    ];
    l.extend(IMG_ACCESSORS.iter().map(|s| (*s).to_string()));
    for (i, (name, _)) in sized.channels.iter().enumerate() {
        l.push(format!("#define {name} iChannel{i}"));
    }
    for name in &sized.arrays {
        l.push(format!("#define {name} {name}_tex"));
    }
    // ISF also exposes each image's placement in its texture. ghee binds no
    // atlas and no flipped image, so the rect is the whole buffer.
    for (name, (w, h)) in &sized.sizes {
        l.push(format!("const vec2 {name}_isf_size = vec2({w}.0, {h}.0);"));
        l.push(format!("const vec2 _{name}_imgSize = vec2({w}.0, {h}.0);"));
        l.push(format!(
            "const vec4 _{name}_imgRect = vec4(0.0, 0.0, {w}.0, {h}.0);"
        ));
        l.push(format!("const bool _{name}_flip = false;"));
    }
    format!("{}\n", l.join("\n"))
}

fn description(meta: &Map<String, Value>) -> String {
    ["DESCRIPTION", "CREDIT"]
        .iter()
        .map(|k| meta.get(*k).map_or(String::new(), py_str))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// --- text scanning ----------------------------------------------------------
//
// A rewrite splices the original source around a match. Every index these
// scanners report is a character index into that source. They work on
// `Vec<char>`, because a byte index would not line up with the mask.

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `\b` at index i: a word char on exactly one side.
fn boundary(s: &[char], i: usize) -> bool {
    let before = i > 0 && is_word(s[i - 1]);
    let after = i < s.len() && is_word(s[i]);
    before != after
}

fn starts_with(src: &[char], i: usize, pat: &str) -> bool {
    for (k, c) in pat.chars().enumerate() {
        if src.get(i + k) != Some(&c) {
            return false;
        }
    }
    true
}

fn find_from(src: &[char], pat: &str, from: usize) -> Option<usize> {
    (from..=src.len()).find(|&i| starts_with(src, i, pat))
}

fn skip_ws(src: &[char], mut i: usize) -> usize {
    while src.get(i).is_some_and(|c| c.is_whitespace()) {
        i += 1;
    }
    i
}

fn slice(src: &[char], a: usize, b: usize) -> String {
    src[a..b].iter().collect()
}

/// `\b<needle>\b`. The needle is a GLSL identifier.
fn search_word(hay: &[char], needle: &str) -> Option<usize> {
    let n = needle.chars().count();
    (0..=hay.len().saturating_sub(n))
        .find(|&i| starts_with(hay, i, needle) && boundary(hay, i) && boundary(hay, i + n))
}

/// Replace every `\b<old>\b` in hay with new.
fn replace_word(hay: &[char], old: &str, new: &str) -> Vec<char> {
    let n = old.chars().count();
    let mut out: Vec<char> = Vec::with_capacity(hay.len());
    let mut i = 0;
    while i < hay.len() {
        if starts_with(hay, i, old) && boundary(hay, i) && boundary(hay, i + n) {
            out.extend(new.chars());
            i += n;
        } else {
            out.push(hay[i]);
            i += 1;
        }
    }
    out
}

/// `[A-Za-z_]\w*$` matched from the start. `$` also stands before one trailing
/// newline.
fn is_ident(s: &str) -> bool {
    let s = s.strip_suffix('\n').unwrap_or(s);
    let mut it = s.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => it.all(is_word),
        _ => false,
    }
}

/// `iChannel\d+$` matched from the start.
fn is_channel_name(s: &str) -> bool {
    let s = s.strip_suffix('\n').unwrap_or(s);
    match s.strip_prefix("iChannel") {
        Some(digits) => !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()),
        None => false,
    }
}

/// src with every comment blanked, and every preprocessor directive too when
/// `directives` is true. A scan then sees code only, and every index still lines
/// up with the original.
fn mask_code(src: &[char], directives: bool) -> Vec<char> {
    let mut out = src.to_vec();
    let n = src.len();
    let mut i = 0;
    while i < n {
        if starts_with(src, i, "//") {
            while i < n && src[i] != '\n' {
                out[i] = ' ';
                i += 1;
            }
        } else if starts_with(src, i, "/*") {
            let end = find_from(src, "*/", i + 2).map_or(n, |e| e + 2);
            for (k, o) in out.iter_mut().enumerate().take(end).skip(i) {
                if src[k] != '\n' {
                    *o = ' ';
                }
            }
            i = end;
        } else if directives && src[i] == '#' && line_start(src, i) {
            let mut more = true;
            while more && i < n {
                let eol = src[i..]
                    .iter()
                    .position(|c| *c == '\n')
                    .map_or(n, |p| i + p);
                more = src[i..eol]
                    .iter()
                    .rposition(|c| !c.is_whitespace())
                    .is_some_and(|p| src[i + p] == '\\');
                for o in out.iter_mut().take(eol).skip(i) {
                    *o = ' ';
                }
                i = eol + 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

fn line_start(src: &[char], i: usize) -> bool {
    let begin = src[..i]
        .iter()
        .rposition(|c| *c == '\n')
        .map_or(0, |p| p + 1);
    src[begin..i].iter().all(|c| c.is_whitespace())
}

/// `^[ \t]*(?:varying|in)[ \t]+\w+[ \t]+(\w+)` over every line: the name.
fn find_varying(body: &[char]) -> Option<String> {
    let mut line = 0;
    loop {
        let start = {
            let mut i = line;
            while matches!(body.get(i), Some(' ' | '\t')) {
                i += 1;
            }
            i
        };
        let hit = ["varying", "in"].iter().find_map(|kw| {
            let mut j = start + kw.chars().count();
            if !starts_with(body, start, kw) || !matches!(body.get(j), Some(' ' | '\t')) {
                return None;
            }
            let word = |body: &[char], mut j: usize| {
                let at = j;
                while body.get(j).is_some_and(|c| is_word(*c)) {
                    j += 1;
                }
                (at, j)
            };
            while matches!(body.get(j), Some(' ' | '\t')) {
                j += 1;
            }
            let (at, end) = word(body, j);
            if end == at || !matches!(body.get(end), Some(' ' | '\t')) {
                return None;
            }
            j = end;
            while matches!(body.get(j), Some(' ' | '\t')) {
                j += 1;
            }
            let (at, end) = word(body, j);
            (end > at).then(|| slice(body, at, end))
        });
        if hit.is_some() {
            return hit;
        }
        line += body[line..].iter().position(|c| *c == '\n')? + 1;
    }
}

/// `\bvoid\s+main\s*\([^)]*\)\s*\{`: the index just past the brace.
fn find_main(mask: &[char]) -> Option<usize> {
    (0..mask.len()).find_map(|i| {
        if !boundary(mask, i) || !starts_with(mask, i, "void") {
            return None;
        }
        let mut j = i + 4;
        if !mask.get(j).is_some_and(|c| c.is_whitespace()) {
            return None;
        }
        j = skip_ws(mask, j);
        if !starts_with(mask, j, "main") {
            return None;
        }
        j = skip_ws(mask, j + 4);
        if mask.get(j) != Some(&'(') {
            return None;
        }
        j += mask[j..].iter().position(|c| *c == ')')?;
        j = skip_ws(mask, j + 1);
        (mask.get(j) == Some(&'{')).then_some(j + 1)
    })
}

/// `\bfor\s*\(` matched at i: the index just past the paren.
fn for_match(mask: &[char], i: usize) -> Option<usize> {
    if !boundary(mask, i) || !starts_with(mask, i, "for") {
        return None;
    }
    let j = skip_ws(mask, i + 3);
    (mask.get(j) == Some(&'(')).then_some(j + 1)
}

/// `\b(?:IMG_(?:NORM_)?(?:THIS_)?(?:NORM_)?PIXEL|texture2D)\s*\(` matched at i.
fn fetch_match(mask: &[char], i: usize) -> Option<usize> {
    if !boundary(mask, i) {
        return None;
    }
    let mut j = i;
    if starts_with(mask, j, "IMG_") {
        j += 4;
        for part in ["NORM_", "THIS_", "NORM_"] {
            if starts_with(mask, j, part) {
                j += part.len();
            }
        }
        if !starts_with(mask, j, "PIXEL") {
            return None;
        }
        j += 5;
    } else if starts_with(mask, j, "texture2D") {
        j += 9;
    } else {
        return None;
    }
    j = skip_ws(mask, j);
    (mask.get(j) == Some(&'(')).then_some(j + 1)
}

fn match_paren(src: &[char], i: usize) -> usize {
    let mut depth = 0i64;
    for (k, c) in src.iter().enumerate().skip(i) {
        if *c == '(' {
            depth += 1;
        } else if *c == ')' {
            depth -= 1;
            if depth == 0 {
                return k;
            }
        }
    }
    src.len() - 1
}

/// The literal bound of a for header (`<=?\s*(\d+)`). One trip otherwise.
fn trips(header: &[char]) -> i64 {
    for i in 0..header.len() {
        if header[i] != '<' {
            continue;
        }
        for eq in [true, false] {
            let mut j = i + 1;
            if eq {
                if header.get(j) != Some(&'=') {
                    continue;
                }
                j += 1;
            }
            j = skip_ws(header, j);
            let at = j;
            while header.get(j).is_some_and(char::is_ascii_digit) {
                j += 1;
            }
            if j > at {
                return slice(header, at, j).parse().unwrap_or(i64::MAX);
            }
        }
    }
    1
}

/// A file-scope `<type> a = x, b;` as (declaration, [(name, initializer)]).
/// None when the statement initializes no variable: a prototype, a const, a
/// uniform, a varying, an interface block.
fn split_decl(stmt: &str) -> Option<(String, Vec<(String, String)>)> {
    let s = stmt.trim();
    let (prefix, rest) = ["lowp", "mediump", "highp"]
        .iter()
        .find_map(|q| {
            let tail = s.strip_prefix(*q)?;
            let cut = tail.len() - tail.trim_start().len();
            (cut > 0).then(|| (format!("{q}{}", &tail[..cut]), &tail[cut..]))
        })
        .filter(|(_, rest)| decl_head(rest).is_some())
        .unwrap_or((String::new(), s));
    let (ty, body) = decl_head(rest)?;
    if !GLSL_TYPES.contains(&ty.as_str()) {
        return None;
    }
    let (mut decls, mut sets): (Vec<String>, Vec<(String, String)>) = (Vec::new(), Vec::new());
    for part in split_commas(&body) {
        let Some((lhs, rhs)) = split_assign(&part) else {
            decls.push(part.trim().to_string());
            continue;
        };
        if lhs.contains('[') {
            // ESSL 1.00 has no array initializer.
            return None;
        }
        decls.push(lhs.trim().to_string());
        sets.push((lhs.trim().to_string(), rhs.trim().to_string()));
    }
    if sets.is_empty() {
        return None;
    }
    Some((format!("{prefix}{ty} {};", decls.join(", ")), sets))
}

/// `([A-Za-z_]\w*)\s+(.*)$`: the type name and everything after it.
fn decl_head(s: &str) -> Option<(String, String)> {
    let c: Vec<char> = s.chars().collect();
    if !c
        .first()
        .is_some_and(|c| c.is_ascii_alphabetic() || *c == '_')
    {
        return None;
    }
    let mut j = 0;
    while c.get(j).is_some_and(|c| is_word(*c)) {
        j += 1;
    }
    if !c.get(j).is_some_and(|c| c.is_whitespace()) {
        return None;
    }
    Some((slice(&c, 0, j), slice(&c, skip_ws(&c, j), c.len())))
}

fn split_commas(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let (mut parts, mut depth, mut start) = (Vec::new(), 0i64, 0usize);
    for i in 0..c.len() {
        match c[i] {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(slice(&c, start, i));
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(slice(&c, start, c.len()));
    parts
}

fn split_assign(part: &str) -> Option<(String, String)> {
    let c: Vec<char> = part.chars().collect();
    let mut depth = 0i64;
    for i in 0..c.len() {
        match c[i] {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '=' if depth == 0 && c.get(i + 1) != Some(&'=') => {
                return Some((slice(&c, 0, i), slice(&c, i + 1, c.len())))
            }
            _ => {}
        }
    }
    None
}

fn basename(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

/// Split off the extension. A leading dot belongs to the name.
fn splitext(path: &str) -> (&str, &str) {
    let sep = path.rfind('/').map_or(0, |i| i + 1);
    let Some(dot) = path.rfind('.').filter(|d| *d >= sep) else {
        return (path, "");
    };
    if path[sep..dot].chars().all(|c| c == '.') {
        return (path, "");
    }
    (&path[..dot], &path[dot..])
}

/// Recursive descent over an ISF size equation. The vocabulary is $names,
/// numbers, + - * /, and floor, ceil, abs, min, max.
struct Expr<'a> {
    text: &'a str,
    values: &'a HashMap<String, f64>,
    tok: Vec<String>,
    i: usize,
}

impl<'a> Expr<'a> {
    fn new(text: &'a str, values: &'a HashMap<String, f64>) -> Expr<'a> {
        Expr {
            text,
            values,
            tok: tokenize(text),
            i: 0,
        }
    }

    fn parse(&mut self) -> Result<f64> {
        let v = self.sum()?;
        if self.i < self.tok.len() {
            return Err(isf_error(format!(
                "size equation {}: unexpected {}",
                repr(self.text),
                repr(&self.tok[self.i])
            )));
        }
        Ok(v)
    }

    fn peek(&self) -> Option<&str> {
        self.tok.get(self.i).map(String::as_str)
    }

    fn take(&mut self, what: Option<&str>) -> Result<String> {
        match self.peek() {
            Some(t) if what.is_none_or(|w| w == t) => {
                let t = t.to_string();
                self.i += 1;
                Ok(t)
            }
            _ => Err(isf_error(format!(
                "size equation {}: expected {}",
                repr(self.text),
                what.unwrap_or("a value")
            ))),
        }
    }

    fn sum(&mut self) -> Result<f64> {
        let mut v = self.product()?;
        while matches!(self.peek(), Some("+" | "-")) {
            let plus = self.take(None)? == "+";
            let rhs = self.product()?;
            v = if plus { v + rhs } else { v - rhs };
        }
        Ok(v)
    }

    fn product(&mut self) -> Result<f64> {
        let mut v = self.unary()?;
        while matches!(self.peek(), Some("*" | "/")) {
            if self.take(None)? == "*" {
                v *= self.unary()?;
            } else {
                let d = self.unary()?;
                if d == 0.0 {
                    return Err(isf_error(format!(
                        "size equation {}: divide by zero",
                        repr(self.text)
                    )));
                }
                v /= d;
            }
        }
        Ok(v)
    }

    fn unary(&mut self) -> Result<f64> {
        if matches!(self.peek(), Some("+" | "-")) {
            let neg = self.take(None)? == "-";
            let v = self.unary()?;
            return Ok(if neg { -v } else { v });
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<f64> {
        let t = self.take(None)?;
        if t == "(" {
            let v = self.sum()?;
            self.take(Some(")"))?;
            return Ok(v);
        }
        if let Some(name) = t.strip_prefix('$') {
            return self.values.get(name).copied().ok_or_else(|| {
                isf_error(format!(
                    "size equation {} reads ${name}, which is not WIDTH, HEIGHT \
                     or a numeric input",
                    repr(self.text)
                ))
            });
        }
        if ["floor", "ceil", "abs", "min", "max"].contains(&t.as_str()) {
            self.take(Some("("))?;
            let mut args = vec![self.sum()?];
            while self.peek() == Some(",") {
                self.take(Some(","))?;
                args.push(self.sum()?);
            }
            self.take(Some(")"))?;
            return call_fn(&t, &args).ok_or_else(|| {
                isf_error(format!(
                    "size equation {}: {t}() got {} value(s)",
                    repr(self.text),
                    args.len()
                ))
            });
        }
        t.parse::<f64>().map_err(|_| {
            isf_error(format!(
                "size equation {}: {} is not a value",
                repr(self.text),
                repr(&t)
            ))
        })
    }
}

/// floor, ceil, abs, and min or max over two or more values. None when the
/// argument count is wrong for the function.
fn call_fn(name: &str, args: &[f64]) -> Option<f64> {
    match (name, args) {
        ("floor", [a]) => Some(a.floor()),
        ("ceil", [a]) => Some(a.ceil()),
        ("abs", [a]) => Some(a.abs()),
        ("min", [a, rest @ ..]) if !rest.is_empty() => Some(rest.iter().fold(*a, |x, y| x.min(*y))),
        ("max", [a, rest @ ..]) if !rest.is_empty() => Some(rest.iter().fold(*a, |x, y| x.max(*y))),
        _ => None,
    }
}

/// `\$?[A-Za-z_]\w*|\d*\.\d+|\d+\.?|[-+*/(),]|\S`, whitespace dropped.
fn tokenize(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let mut j = i + usize::from(c[i] == '$');
        if c.get(j)
            .is_some_and(|x| x.is_ascii_alphabetic() || *x == '_')
        {
            while c.get(j).is_some_and(|x| is_word(*x)) {
                j += 1;
            }
            out.push(slice(&c, i, j));
            i = j;
            continue;
        }
        j = i;
        while c.get(j).is_some_and(char::is_ascii_digit) {
            j += 1;
        }
        if c.get(j) == Some(&'.') && c.get(j + 1).is_some_and(char::is_ascii_digit) {
            j += 1;
            while c.get(j).is_some_and(char::is_ascii_digit) {
                j += 1;
            }
        } else if j > i {
            j += usize::from(c.get(j) == Some(&'.'));
        } else {
            if !c[i].is_whitespace() {
                out.push(c[i].to_string());
            }
            i += 1;
            continue;
        }
        out.push(slice(&c, i, j));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ArraySize;
    use serde_json::json;

    const FILTER: &str = r#"/*{
    "DESCRIPTION": "half tint",
    "CREDIT": "tests",
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        { "NAME": "amount", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0, "MAX": 1.0 },
        { "NAME": "invert", "TYPE": "bool", "DEFAULT": true },
        { "NAME": "mode", "TYPE": "long", "VALUES": [0, 1, 2], "DEFAULT": 1 },
        { "NAME": "tint", "TYPE": "color", "DEFAULT": [1.0, 0.2, 0.0, 1.0] },
        { "NAME": "centre", "TYPE": "point2D", "DEFAULT": [0.5, 0.5] }
    ]
}*/

void main() {
    vec4 src = IMG_THIS_PIXEL(inputImage);
    vec4 far = IMG_PIXEL(inputImage, isf_FragNormCoord * RENDERSIZE);
    float d = distance(isf_FragNormCoord, centre) * float(mode) + TIME * 0.0;
    vec3 c = mix(src.rgb, tint.rgb, amount * d);
    gl_FragColor = vec4(invert ? 1.0 - c : c, far.a);
}
"#;

    const TWO_PASS: &str = r#"/*{
    "INPUTS": [ { "NAME": "gain", "TYPE": "float", "DEFAULT": 2.0 } ],
    "PASSES": [
        { "TARGET": "acc", "WIDTH": "floor($WIDTH/2.0)", "HEIGHT": "max($HEIGHT/4.0, 1.0)", "PERSISTENT": true },
        { }
    ]
}*/

void main() {
    if (PASSINDEX == 0) {
        vec4 prev = IMG_THIS_PIXEL(acc);
        gl_FragColor = vec4(prev.rgb * 0.9 + isf_FragNormCoord.x * 0.1, 1.0);
        return;
    }
    gl_FragColor = IMG_NORM_PIXEL(acc, isf_FragNormCoord) * gain;
}
"#;

    const AUDIO: &str = r#"/*{
    "INPUTS": [ { "NAME": "fft", "TYPE": "audioFFT", "MAX": 32 } ]
}*/

void main() {
    vec4 v = IMG_NORM_PIXEL(fft, vec2(isf_FragNormCoord.x, 0.5));
    gl_FragColor = vec4(vec3((v.r + v.g + v.b) / 3.0), 1.0);
}
"#;

    const BODY: &str = "void main(){ gl_FragColor = vec4(1.0); }";

    fn isf(meta: Value) -> String {
        isf_body(meta, BODY)
    }

    fn isf_body(meta: Value, body: &str) -> String {
        format!("/*{}*/\n{body}", serde_json::to_string(&meta).unwrap())
    }

    fn scene(src: &str, name: &str, size: (i64, i64)) -> Scene {
        to_scene(src, name, size, false).unwrap().0
    }

    fn names(sc: &Scene) -> Vec<String> {
        sc.passes.iter().map(|p| p.name.clone()).collect()
    }

    fn buffers(p: &Pass) -> Vec<Option<String>> {
        p.channels.iter().map(|c| c.buffer.clone()).collect()
    }

    fn err<T>(r: Result<T>) -> String {
        format!("{}", r.err().expect("expected a refusal"))
    }

    fn dims(width: f64, height: f64) -> HashMap<String, f64> {
        HashMap::from([("WIDTH".to_string(), width), ("HEIGHT".to_string(), height)])
    }

    #[test]
    fn header_split() {
        let (meta, body) = split_header(FILTER).unwrap();
        assert_eq!(meta["DESCRIPTION"], json!("half tint"));
        assert!(body.trim_start().starts_with("void main()"));
    }

    #[test]
    fn header_missing_is_refused() {
        assert!(err(split_header("void main(){}")).contains("not ISF"));
    }

    #[test]
    fn header_bad_json_is_refused() {
        assert!(err(split_header("/*{ nope }*/\nvoid main(){}")).contains("does not parse"));
    }

    // --- inputs ------------------------------------------------------------

    #[test]
    fn input_types_map_to_params() {
        let sc = scene(FILTER, "half tint.fs", (64, 48));
        let p = &sc.params;
        let amount = p.get("amount").unwrap();
        assert_eq!(
            (&amount.ty[..], &amount.value),
            ("float", &ParamValue::Float(0.5))
        );
        assert_eq!(
            (&amount.min, &amount.max),
            (&Some(json!(0.0)), &Some(json!(1.0)))
        );
        let invert = p.get("invert").unwrap();
        assert_eq!(
            (&invert.ty[..], &invert.value),
            ("bool", &ParamValue::Bool(true))
        );
        // A long is an int over the range its VALUES span.
        let mode = p.get("mode").unwrap();
        assert_eq!((&mode.ty[..], &mode.value), ("int", &ParamValue::Int(1)));
        assert_eq!(
            (&mode.min, &mode.max, &mode.step),
            (&Some(json!(0)), &Some(json!(2)), &Some(json!(1)))
        );
        // An ISF color is RGBA, so it becomes a vec4.
        let tint = p.get("tint").unwrap();
        assert_eq!(
            (&tint.ty[..], &tint.value),
            ("vec4", &ParamValue::Vec(vec![1.0, 0.2, 0.0, 1.0]))
        );
        let centre = p.get("centre").unwrap();
        assert_eq!(
            (&centre.ty[..], &centre.value),
            ("vec2", &ParamValue::Vec(vec![0.5, 0.5]))
        );
        assert!(!p.contains_key("inputImage")); // An image input is a channel.
    }

    #[test]
    fn event_is_a_bool() {
        let sc = scene(
            &isf(json!({"INPUTS": [{"NAME": "go", "TYPE": "event"}]})),
            "e.fs",
            (8, 8),
        );
        let go = sc.params.get("go").unwrap();
        assert_eq!((&go.ty[..], &go.value), ("bool", &ParamValue::Bool(false)));
    }

    #[test]
    fn defaults_when_the_input_names_none() {
        let sc = scene(
            &isf(json!({"INPUTS": [
                {"NAME": "f", "TYPE": "float", "MIN": 3.0},
                {"NAME": "c", "TYPE": "color"},
                {"NAME": "p", "TYPE": "point2D"},
            ]})),
            "d.fs",
            (8, 8),
        );
        // A float with no DEFAULT falls back to MIN, then to 0.0.
        assert_eq!(sc.params.get("f").unwrap().value, ParamValue::Float(3.0));
        assert_eq!(
            sc.params.get("c").unwrap().value,
            ParamValue::Vec(vec![0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            sc.params.get("p").unwrap().value,
            ParamValue::Vec(vec![0.0, 0.0])
        );
    }

    #[test]
    fn unknown_input_type_is_refused() {
        let src = isf(json!({"INPUTS": [{"NAME": "x", "TYPE": "audioFFTBins"}]}));
        let e = err(to_scene(&src, "u.fs", DEFAULT_SIZE, false));
        assert!(e.contains("unsupported TYPE 'audioFFTBins'"), "{e}");
    }

    #[test]
    fn audio_input_becomes_an_array() {
        let sc = scene(AUDIO, "audio.fs", (64, 48));
        assert_eq!(sc.passes[0].arrays.len(), 1);
        let a = &sc.passes[0].arrays[0];
        assert_eq!((&a.name[..], a.size), ("fft", ArraySize::Linear(32)));
        // ghee binds the array as sampler2D <name>_tex in GL_LUMINANCE. The ISF
        // (r+g+b)/3 then reads the array value.
        assert!(sc.passes[0].code.contains("#define fft fft_tex"));
        assert!(sc.passes[0].channels.is_empty());
    }

    #[test]
    fn audio_input_defaults_to_256_bins() {
        let sc = scene(
            &isf(json!({"INPUTS": [{"NAME": "w", "TYPE": "audio"}]})),
            "a.fs",
            (8, 8),
        );
        assert_eq!(sc.passes[0].arrays[0].size, ArraySize::Linear(256));
    }

    // --- passes, channels, sizes -------------------------------------------

    #[test]
    fn image_input_gets_a_generated_source_pass() {
        let (sc, warn) = to_scene(FILTER, "half tint.fs", (64, 48), false).unwrap();
        assert_eq!(names(&sc), ["isf_source", "out"]);
        assert_eq!(buffers(&sc.passes[1]), [Some("isf_source".to_string())]);
        assert!(sc.passes[1].code.contains("#define inputImage iChannel0"));
        assert!(warn.iter().any(|w| w.contains("no producer")));
    }

    #[test]
    fn multipass_shares_one_body_with_its_own_passindex() {
        let sc = scene(TWO_PASS, "two.fs", (320, 240));
        assert_eq!(names(&sc), ["acc", "out"]);
        assert!(sc.passes[0].code.contains("const int PASSINDEX = 0;"));
        assert!(sc.passes[1].code.contains("const int PASSINDEX = 1;"));
        // ISF puts one body in every pass. Only the constant differs.
        let body = "void main() {";
        assert_eq!(
            sc.passes[0].code.split_once(body).unwrap().1,
            sc.passes[1].code.split_once(body).unwrap().1
        );
    }

    #[test]
    fn persistent_target_is_a_self_channel() {
        let sc = scene(TWO_PASS, "two.fs", (320, 240));
        assert_eq!(buffers(&sc.passes[0]), [Some("acc".to_string())]);
        assert_eq!(buffers(&sc.passes[1]), [Some("acc".to_string())]);
        assert!(sc.passes[0].code.contains("#define acc iChannel0"));
    }

    #[test]
    fn pass_sizes_come_from_the_equations() {
        let sc = scene(TWO_PASS, "two.fs", (320, 240));
        assert_eq!(sc.passes[0].size, Some((160, 60)));
        assert_eq!(sc.passes[1].size, None); // The output pass takes the render size.
        assert!(sc.passes[0]
            .code
            .contains("const vec2 acc_isf_size = vec2(160.0, 60.0);"));
    }

    #[test]
    fn a_sized_last_pass_is_dropped_with_a_warning() {
        let src = isf(json!({
            "PASSES": [{"TARGET": "a"}, {"WIDTH": "$WIDTH/2.0", "HEIGHT": "$HEIGHT"}]
        }));
        let (sc, warn) = to_scene(&src, "s.fs", (64, 48), false).unwrap();
        assert_eq!(sc.passes.last().unwrap().size, None);
        assert!(warn
            .iter()
            .any(|w| w.contains("output pass renders at the render size")));
    }

    #[test]
    fn reserved_name_is_renamed_in_the_body() {
        let src = isf_body(
            json!({"PASSES": [{"TARGET": "time"}, {}]}),
            "void main(){ gl_FragColor = IMG_THIS_PIXEL(time) * TIME; }",
        );
        let (sc, warn) = to_scene(&src, "r.fs", (8, 8), false).unwrap();
        assert_eq!(names(&sc), ["isf_time", "out"]);
        assert!(sc.passes[0].code.contains("IMG_THIS_PIXEL(isf_time)"));
        assert!(warn.iter().any(|w| w.contains("renamed to 'isf_time'")));
    }

    #[test]
    fn img_size_becomes_a_constant() {
        let src = isf_body(
            json!({"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]}),
            "void main(){ gl_FragColor = vec4(IMG_SIZE(inputImage), 0.0, 1.0); }",
        );
        let sc = scene(&src, "z.fs", (64, 48));
        let code = &sc.passes.last().unwrap().code;
        assert!(code.contains("inputImage_isf_size") && !code.contains("IMG_SIZE("));
        assert!(code.contains("const vec2 inputImage_isf_size = vec2(64.0, 48.0);"));
    }

    #[test]
    fn img_pixel_is_handed_the_sampler_size() {
        let sc = scene(FILTER, "f.fs", (64, 48));
        assert!(sc
            .passes
            .last()
            .unwrap()
            .code
            .contains("IMG_PIXEL(inputImage_isf_size, inputImage,"));
    }

    #[test]
    fn commented_out_img_call_is_left_alone() {
        let src = isf_body(
            json!({}),
            "void main(){\n// gl_FragColor = IMG_PIXEL(nosuchimage, vec2(0.0));\n\
             gl_FragColor = vec4(1.0); }",
        );
        let sc = scene(&src, "c.fs", (8, 8));
        assert!(sc
            .passes
            .last()
            .unwrap()
            .code
            .contains("IMG_PIXEL(nosuchimage, vec2(0.0))"));
    }

    #[test]
    fn img_pixel_on_an_unknown_sampler_is_refused() {
        let src = isf_body(
            json!({}),
            "void main(){ gl_FragColor = IMG_PIXEL(mystery, vec2(0.0)); }",
        );
        assert!(err(to_scene(&src, "m.fs", (8, 8), false)).contains("mystery"));
    }

    // --- refusals ----------------------------------------------------------

    #[test]
    fn float_buffer_is_refused() {
        let src = isf(json!({"PASSES": [{"TARGET": "acc", "FLOAT": true}, {}]}));
        assert!(err(to_scene(&src, "acc.fs", DEFAULT_SIZE, false)).contains("FLOAT"));
    }

    #[test]
    fn float_refusal_names_the_file_and_the_buffer() {
        let src = isf(json!({"PASSES": [{"TARGET": "acc", "FLOAT": true}, {}]}));
        let e = err(to_scene(&src, "acc.fs", DEFAULT_SIZE, false));
        assert!(e.contains("acc.fs") && e.contains("acc"));
        assert!(e.contains("8-bit"), "{e}");
    }

    #[test]
    fn sibling_vertex_shader_is_refused() {
        assert!(err(to_scene(FILTER, "f.fs", (8, 8), true)).contains("vertex shader"));
    }

    #[test]
    fn custom_varying_is_refused() {
        let src = isf_body(
            json!({}),
            "varying vec2 texOffsets;\nvoid main(){ gl_FragColor = vec4(1.0); }",
        );
        assert!(err(to_scene(&src, "v.fs", DEFAULT_SIZE, false)).contains("texOffsets"));
    }

    #[test]
    fn imported_image_is_refused() {
        let src = isf(json!({"IMPORTED": {"noiseTex": {"PATH": "noise.png"}}}));
        assert!(err(to_scene(&src, "i.fs", DEFAULT_SIZE, false)).contains("noiseTex"));
    }

    #[test]
    fn no_main_to_hoist_into_is_refused() {
        let src = isf_body(json!({}), "float ratio = RENDERSIZE.x;\n");
        assert!(err(to_scene(&src, "n.fs", DEFAULT_SIZE, false)).contains("no void main"));
    }

    // --- the two body rewrites ---------------------------------------------

    #[test]
    fn hoist_moves_a_uniform_initializer_into_main() {
        let body = "float ratio = RENDERSIZE.x / RENDERSIZE.y;\nvoid main(){ f(ratio); }";
        let (out, names) = hoist_globals(body).unwrap();
        assert_eq!(names, ["ratio"]);
        assert!(out.starts_with("float ratio;"));
        assert!(out.contains("void main(){\n    ratio = RENDERSIZE.x / RENDERSIZE.y;"));
    }

    #[test]
    fn hoist_handles_several_declarators() {
        let body = "vec2 a = vec2(1.0), b, c = a * 2.0;\nvoid main(){}";
        let (out, names) = hoist_globals(body).unwrap();
        assert_eq!(names, ["a", "c"]);
        assert!(out.starts_with("vec2 a, b, c;"));
        assert!(out.find("a = vec2(1.0);") < out.find("c = a * 2.0;"));
    }

    #[test]
    fn hoist_leaves_const_and_declarations_alone() {
        let body = "const float PI = 3.14;\nuniform float u;\nvec4 f(vec2 p);\n\
                    struct S { float a; };\nvoid main(){}";
        let (out, names) = hoist_globals(body).unwrap();
        assert!(names.is_empty());
        assert_eq!(out, body);
    }

    #[test]
    fn hoist_keeps_a_leading_comment() {
        let body = "// keep me\nfloat r = RENDERSIZE.x;\nvoid main(){}";
        let (out, _) = hoist_globals(body).unwrap();
        assert!(out.starts_with("// keep me\nfloat r;"));
    }

    #[test]
    fn hoist_ignores_locals() {
        let body = "void main(){ float x = RENDERSIZE.x; }";
        let (out, names) = hoist_globals(body).unwrap();
        assert_eq!((out.as_str(), names.len()), (body, 0));
    }

    // --- size equations ----------------------------------------------------

    #[test]
    fn size_equations() {
        let mut values = dims(320.0, 240.0);
        values.insert("quality".to_string(), 0.1);
        let cases: [(Value, i64); 12] = [
            (json!(512), 512),
            (json!(1.0), 1),
            (json!("$WIDTH"), 320),
            (json!("$HEIGHT"), 240),
            (json!("floor($WIDTH/4.0)"), 80),
            (json!("$WIDTH / 100.0"), 3),
            (json!("max(floor($WIDTH*0.02),1.0)"), 6),
            (json!("max($HEIGHT / 243.0, 1.0)"), 1),
            (json!("floor($WIDTH*min((0.3),1.0))"), 96),
            (json!("max(8.0,floor($HEIGHT*$quality))"), 24),
            (json!("$WIDTH - $HEIGHT"), 80),
            (json!("-$WIDTH"), 1), // A target is at least one pixel.
        ];
        for (expr, want) in cases {
            assert_eq!(eval_size(&expr, &values).unwrap(), want, "{expr}");
        }
    }

    #[test]
    fn size_equation_unknown_name_is_refused() {
        let e = err(eval_size(&json!("$nope * 2.0"), &dims(320.0, 240.0)));
        assert!(e.contains("$nope"), "{e}");
    }

    #[test]
    fn size_equation_garbage_is_refused() {
        assert!(eval_size(&json!("$WIDTH >> 2"), &dims(320.0, 240.0)).is_err());
    }

    #[test]
    fn size_equation_reads_a_numeric_input() {
        let src = isf(json!({
            "INPUTS": [{"NAME": "trailCount", "TYPE": "float", "DEFAULT": 8}],
            "PASSES": [{"TARGET": "a", "WIDTH": "$trailCount", "HEIGHT": "1"}, {}],
        }));
        assert_eq!(scene(&src, "t.fs", (64, 48)).passes[0].size, Some((8, 1)));
    }

    // --- filter mode -------------------------------------------------------

    #[test]
    fn filter_mode_writes_a_filter_definition() {
        let (fd, _) = to_filter(FILTER, "half tint.fs", (64, 48), false).unwrap();
        assert_eq!(fd["name"], json!("half_tint"));
        assert_eq!(fd["description"], json!("half tint tests"));
        assert!(fd["code"]
            .as_str()
            .unwrap()
            .contains("#define inputImage iChannel0"));
        let mut keys: Vec<&String> = fd["params"].as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["amount", "centre", "invert", "mode", "tint"]);
        assert_eq!(
            fd["params"]["invert"],
            json!({"type": "bool", "value": true})
        );
    }

    #[test]
    fn filter_mode_loads_through_the_filter_registry() {
        let _g = crate::filters::env_guard();
        let dir = std::env::temp_dir().join(format!("ghee-isf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (fd, _) = to_filter(FILTER, "half tint.fs", (64, 48), false).unwrap();
        std::fs::write(dir.join("half_tint.filter.json"), fd.to_string()).unwrap();
        let prev = std::env::var("GHEE_FILTERS").ok();
        std::env::set_var("GHEE_FILTERS", &dir);
        let registry = crate::filters::load_filters(&[]);
        match prev {
            Some(v) => std::env::set_var("GHEE_FILTERS", v),
            None => std::env::remove_var("GHEE_FILTERS"),
        }
        std::fs::remove_dir_all(&dir).unwrap();

        let mut sc = Scene::of(vec![Pass::of(
            "main",
            "void main(){ gl_FragColor=vec4(1.0); }",
        )
        .unwrap()])
        .unwrap();
        crate::filters::apply_filters(&mut sc, &["half_tint".to_string()], Some(&registry))
            .unwrap();
        assert_eq!(names(&sc), ["main", "half_tint"]);
        assert_eq!(
            buffers(sc.passes.last().unwrap()),
            [Some("main".to_string())]
        );
    }

    #[test]
    fn filter_mode_refusals() {
        let cases = [
            (json!({"PASSES": [{"TARGET": "a"}, {}]}), "2 passes"),
            (
                json!({"INPUTS": [
                    {"NAME": "a", "TYPE": "image"},
                    {"NAME": "b", "TYPE": "image"},
                ]}),
                "2 image inputs",
            ),
            (json!({}), "no image input"),
            (
                json!({"INPUTS": [
                    {"NAME": "a", "TYPE": "image"},
                    {"NAME": "f", "TYPE": "audio"},
                ]}),
                "reads audio",
            ),
        ];
        for (meta, want) in cases {
            let e = err(to_filter(&isf(meta), "x.fs", (8, 8), false));
            assert!(e.contains(want), "{want} not in {e}");
        }
    }

    #[test]
    fn filter_mode_refuses_feedback() {
        let src = isf_body(
            json!({
                "INPUTS": [{"NAME": "inputImage", "TYPE": "image"}],
                "PASSES": [{"TARGET": "last", "PERSISTENT": true}],
            }),
            "void main(){ gl_FragColor = IMG_THIS_PIXEL(last); }",
        );
        assert!(err(to_filter(&src, "fb.fs", (8, 8), false)).contains("reads its own buffer"));
    }

    // --- warnings ----------------------------------------------------------

    #[test]
    fn fetch_estimate_multiplies_by_literal_loop_bounds() {
        assert_eq!(
            fetch_estimate("void main(){ vec4 a = IMG_THIS_PIXEL(x); }").unwrap(),
            1
        );
        let body = "void main(){ for (int i = 0; i < 16; i++) { c += IMG_THIS_PIXEL(x); } }";
        assert_eq!(fetch_estimate(body).unwrap(), 16);
        let nested = "void main(){ for (int i = 0; i < 4; i++) { \
                      for (int j = 0; j < 8; j++) { c += IMG_THIS_PIXEL(x); } } \
                      c += IMG_THIS_PIXEL(x); }";
        assert_eq!(fetch_estimate(nested).unwrap(), 33);
    }

    #[test]
    fn heavy_shader_warns_but_still_imports() {
        let src = isf_body(
            json!({"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]}),
            "void main(){ vec4 c = vec4(0.0);\n\
             for (int i = 0; i < 300; i++) { c += IMG_THIS_PIXEL(inputImage); }\n\
             gl_FragColor = c; }",
        );
        let (sc, warn) = to_scene(&src, "heavy.fs", (8, 8), false).unwrap();
        assert!(!sc.passes.is_empty());
        assert!(warn
            .iter()
            .any(|w| w.contains("texture fetches per fragment")));
    }

    #[test]
    fn slug_names() {
        assert_eq!(slug("Radial Spectrogram.fs"), "radial_spectrogram");
        assert_eq!(slug("/a/b/RGB Trails 3.0.fs"), "rgb_trails_3_0");
    }

    // --- on a GPU ----------------------------------------------------------

    /// The renderer these tests share. None when there is no GL device.
    fn renderer() -> Option<crate::engine::Renderer> {
        let mut errs = Vec::new();
        for dev in crate::engine::probe_dri_devices() {
            match crate::engine::Renderer::new(64, 48, 4, Some(&dev), false, false) {
                Ok(r) => return Some(r),
                Err(e) => errs.push(format!("{dev}: {e}")),
            }
        }
        eprintln!("no GL device for rendering: {}", errs.join("; "));
        None
    }

    #[test]
    fn imported_scenes_compile_and_render() {
        let Some(mut r) = renderer() else { return };
        let (w, h, comp) = (r.width as usize, r.height as usize, r.components as usize);
        for (src, name) in [(FILTER, "f.fs"), (TWO_PASS, "t.fs"), (AUDIO, "a.fs")] {
            let (sc, _) = to_scene(src, name, (w as i64, h as i64), false).unwrap();
            assert_eq!(r.setup(&sc).unwrap(), "", "{name}");
            let mut feed = |_: &str, size: ArraySize| {
                let n = match size {
                    ArraySize::Linear(n) => n as usize,
                    ArraySize::Grid(gw, gh) => (gw * gh) as usize,
                };
                let last = (n - 1).max(1) as f32;
                crate::engine::Feed::Data((0..n).map(|i| i as f32 / last).collect())
            };
            let frame = r.render(Some(&mut feed)).unwrap();
            assert_eq!(frame.len(), h * w * comp);
            let rgb: Vec<f64> = (0..frame.len())
                .filter(|i| i % comp < 3)
                .map(|i| f64::from(frame[i]))
                .collect();
            let mean = rgb.iter().sum::<f64>() / rgb.len() as f64;
            let var = rgb.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / rgb.len() as f64;
            assert!(var.sqrt() > 0.5, "{name} rendered flat");
        }
    }
}
