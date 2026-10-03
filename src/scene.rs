//! The scene model, and the GLSL that it compiles to.
//!
//! A [`Scene`] holds an ordered list of passes, and the params, assets and
//! texts that the passes share. A [`Pass`] holds its GLSL, its channels,
//! arrays and params, and its layers. [`Scene::load`] reads a scene JSON
//! file: it refuses a key that it does not take, checks every name and size,
//! and reads the files that the scene names, from the scene's directory or
//! from `$GHEE_ASSETS`. A [`Reach`] sets whether a file may lie outside that
//! directory. [`Scene::of`] and [`Pass::of`] build a scene in code.
//!
//! [`Pass::glsl`] writes the fragment shader of a pass in three parts: the
//! preamble, the scene's common block, then the pass's own GLSL. The preamble
//! declares the uniforms that every pass reads, `time`, `DT`, `frame` and
//! `resolution`, and `iChannel0` and up, then one uniform for each param
//! under the param's name, and the samplers and the reader functions of each
//! array, asset and text. The C engine sets each uniform by looking up that
//! same name in the compiled program, so a name that the preamble writes is
//! the contract between the two. A layer's vertex and fragment shaders take
//! the same preamble, with ghee's code for the text's fill around the
//! layer's own GLSL.

use std::fs;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::engine::Blend;
use crate::text::{Fill, Text};

const GLSL_TYPE: [(&str, &str); 3] = [("float", "float"), ("int", "int"), ("bool", "bool")];

const VEC_ARITY: [(&str, usize); 3] = [("vec2", 2), ("vec3", 3), ("vec4", 4)];

/// Every name that ghee writes into GLSL and an author never writes starts
/// with this prefix: a text's own uniforms, textures and functions, a layer's
/// attributes, varyings and locals, and the vertex flip.
pub const PREFIX: &str = "ghee_";

/// The uniforms and the varying that ghee declares in every pass for its
/// author to read, beside iChannel0 and up.
const BUILT_INS: [&str; 5] = ["time", "DT", "frame", "resolution", "v_texCoord"];

/// True for a name that ghee declares in every pass for its author to read.
pub(crate) fn built_in(name: &str) -> bool {
    BUILT_INS.contains(&name)
        || name
            .strip_prefix("iChannel")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

const PARAM_TYPES: [&str; 8] = [
    "float", "int", "bool", "color", "string", "vec2", "vec3", "vec4",
];

// An array and a pass both become a GL texture, so a texture size bounds them.
// One dimension caps at 16384, above any GLES2 GL_MAX_TEXTURE_SIZE. A 2D input
// or a sized pass caps at 16M texels.
const MAX_DIM: i64 = 16384;
const MAX_TEXELS: i64 = 1 << 24;

fn vec_arity(ty: &str) -> Option<usize> {
    VEC_ARITY.iter().find(|(t, _)| *t == ty).map(|(_, n)| *n)
}

fn glsl_type_of(ty: &str) -> Option<&'static str> {
    GLSL_TYPE.iter().find(|(t, _)| *t == ty).map(|(_, g)| *g)
}

fn is_num(v: &Value) -> bool {
    v.is_number()
}

fn is_dim(v: &Value) -> bool {
    v.as_i64().is_some_and(|n| 0 < n && n <= MAX_DIM)
}

/// `[A-Za-z_]\w*$` matched from the start.
pub(crate) fn is_ident(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {
            it.all(|c| c.is_alphanumeric() || c == '_')
        }
        _ => false,
    }
}

/// A JSON size is an int or a whole float. Any other value passes through
/// unchanged, so the size check reports what the scene declares.
fn to_int(v: &Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 9.0e18 => Value::from(f as i64),
            _ => v.clone(),
        },
        _ => v.clone(),
    }
}

fn list_of<'a>(d: &'a Map<String, Value>, key: &str) -> Result<&'a Vec<Value>> {
    static EMPTY: Vec<Value> = Vec::new();
    match d.get(key) {
        None => Ok(&EMPTY),
        Some(Value::Array(a)) => Ok(a),
        Some(v) => bail!("'{key}' needs a list, got {}", py_repr(v)),
    }
}

fn obj_of<'a>(d: &'a Map<String, Value>, key: &str) -> Result<&'a Map<String, Value>> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    match d.get(key) {
        None => Ok(EMPTY.get_or_init(Map::new)),
        Some(Value::Object(o)) => Ok(o),
        Some(v) => bail!(
            "'{key}' needs an object of name -> definition, got {}",
            py_repr(v)
        ),
    }
}

/// The keys of the scene's top level.
const SCENE_KEYS: [&str; 7] = [
    "version", "passes", "filters", "params", "common", "assets", "texts",
];

/// The keys of a pass.
const PASS_KEYS: [&str; 10] = [
    "name", "code", "shader", "arrays", "channels", "params", "filter", "size", "mipmaps", "layers",
];

/// The keys of a layer.
pub(crate) const LAYER_KEYS: [&str; 8] = [
    "name",
    "text",
    "vertex",
    "px_per_em",
    "fragment",
    "blend",
    "pad",
    "emoji",
];

/// The keys of an asset. Only a .raw file takes the four after `sampling`.
pub(crate) const ASSET_KEYS: [&str; 6] = [
    "file",
    "sampling",
    "width",
    "height",
    "format",
    "components",
];

/// The keys of a text.
pub(crate) const TEXT_KEYS: [&str; 5] = ["value", "font", "align", "fill", "atlas"];

/// The keys of a param.
const PARAM_KEYS: [&str; 6] = ["type", "value", "min", "max", "step", "options"];

/// The keys of a channel, an array and the common block.
const CHANNEL_KEYS: [&str; 1] = ["buffer"];
const ARRAY_KEYS: [&str; 2] = ["name", "size"];
const COMMON_KEYS: [&str; 2] = ["code", "shader"];

/// The edits that turn `a` into `b`, one at a time: a character inserted,
/// removed or replaced, or two neighbours swapped.
fn edits(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    // d[i][j]: the edits from the first i characters of a to the first j of b.
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let replace = d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]);
            let mut best = replace.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = best;
        }
    }
    d[a.len()][b.len()]
}

/// The edits from `typed` to `key` when `key` is a likely meaning of it: a
/// third of the longer name or fewer, or the first four characters shared.
fn near(typed: &str, key: &str) -> Option<usize> {
    let d = edits(typed, key);
    let longer = typed.chars().count().max(key.chars().count());
    let shared = typed
        .chars()
        .zip(key.chars())
        .take_while(|(a, b)| a == b)
        .count();
    (d <= (longer / 3).max(1) || shared >= 4).then_some(d)
}

/// Refuse the first key of `o` that `what` does not take. The error names
/// the key and the closest key that `what` takes, or every key that it takes
/// when none comes close. `elsewhere` holds the keys of other objects, each
/// with the object that takes it, and a key that sits there, or comes close
/// to one, names that object.
pub(crate) fn known_keys(
    o: &Map<String, Value>,
    what: &str,
    takes: &[&str],
    elsewhere: &[(&str, String)],
) -> Result<()> {
    let Some(key) = o.keys().find(|k| !takes.contains(&k.as_str())) else {
        return Ok(());
    };
    if let Some((_, on)) = elsewhere.iter().find(|(k, _)| k == key) {
        bail!("{what} takes no '{key}', which sits on {on}");
    }
    let own = takes.iter().map(|k| (*k, None));
    let other = elsewhere.iter().map(|(k, on)| (*k, Some(on)));
    let closest = own
        .chain(other)
        .filter_map(|(k, on)| near(key, k).map(|d| (d, k, on)))
        .min_by_key(|(d, _, _)| *d);
    match closest {
        Some((_, k, None)) => bail!("{what} takes no '{key}', the closest key is '{k}'"),
        Some((_, k, Some(on))) => {
            bail!("{what} takes no '{key}', the closest key is '{k}', which sits on {on}")
        }
        None => {
            let all = match takes.split_last() {
                Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
                _ => takes.join(""),
            };
            bail!("{what} takes no '{key}', only {all}")
        }
    }
}

/// The keys in `of` that `takes` lacks, each with `on`, the object that
/// takes it, for known_keys().
pub(crate) fn keys_on<'a>(takes: &[&str], of: &[&'a str], on: &str) -> Vec<(&'a str, String)> {
    of.iter()
        .filter(|k| !takes.contains(k))
        .map(|k| (*k, on.to_string()))
        .collect()
}

/// A JSON value as Python's repr() writes it: `None`, `True`, `False`, a
/// quoted string, a shortest round-trip float. The errors quote a scene's
/// values back in this form.
pub(crate) fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => i.to_string(),
            (_, Some(u), _) => u.to_string(),
            (_, _, Some(f)) => py_float(f),
            _ => n.to_string(),
        },
        Value::String(s) => py_str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// A JSON value as Python's str() writes it: as py_repr() does, but a string
/// stays unquoted.
pub(crate) fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => py_repr(v),
    }
}

fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A float as Python's repr() writes it: the shortest round-trip. A decimal
/// exponent in -4..15 prints fixed. Any other prints with an exponent of two
/// digits or more.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sci = format!("{f:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exponent.parse().unwrap_or(0);
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exp) {
        if exp < 0 {
            return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
        }
        let lead = (exp + 1) as usize;
        if digits.len() > lead {
            format!("{sign}{}.{}", &digits[..lead], &digits[lead..])
        } else {
            format!("{sign}{digits}{}.0", "0".repeat(lead - digits.len()))
        }
    } else {
        let frac = match &digits[1..] {
            "" => String::new(),
            tail => format!(".{tail}"),
        };
        let s = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}{frac}e{s}{:02}", &digits[..1], exp.abs())
    }
}

/// The OS message alone. io::Error prints an errno after it.
pub(crate) fn strerror(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// A lexical path cleanup. It does not resolve a symlink.
pub(crate) fn normpath(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut out: Vec<&str> = Vec::new();
    for part in path.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part != ".." || (!absolute && out.is_empty()) || out.last() == Some(&"..") {
            out.push(part);
        } else if !out.is_empty() {
            out.pop();
        }
    }
    let joined = out.join("/");
    if absolute {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

fn abspath(path: &str) -> String {
    if path.starts_with('/') {
        return normpath(path);
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    normpath(&format!("{}/{path}", cwd.to_string_lossy()))
}

pub(crate) fn join(root: &str, path: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("{root}/{path}")
    }
}

/// Whether a normalized path is the directory root or lies inside it. The
/// comparison takes whole components, so "/" holds every absolute path, and
/// "/a/b" holds no "/a/bc".
pub(crate) fn inside(path: &str, root: &str) -> bool {
    std::path::Path::new(path).starts_with(root)
}

/// Which files a scene may name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    /// Any file: a path may be absolute or leave the scene's directory. A
    /// scene that its user runs reads this way.
    Any,
    /// The files inside the scene's directory alone. The console reads every
    /// scene this way, since anyone who reaches it posts and saves scenes.
    Inside,
}

/// The directory where a scene's files resolve, and the files it may name.
#[derive(Clone, Copy, Debug)]
struct Files<'a> {
    dir: &'a str,
    reach: Reach,
}

/// `base_dir` with the reach of a scene that its user runs.
fn any_file(base_dir: Option<&str>) -> Option<Files<'_>> {
    base_dir.map(|dir| Files {
        dir,
        reach: Reach::Any,
    })
}

/// Read a `shader` file from the scene's own directory.
///
/// A scene with no directory names no file. A console post and an in-memory
/// scene have no directory. A scene read with Reach::Inside reads inside its
/// directory only. `what` names the holder in an error: a pass, or the
/// scene's common block.
fn read_shader(path: &Value, files: Option<Files>, what: &str) -> Result<String> {
    let Value::String(name) = path else {
        bail!("{what} shader needs a filename, got {}", py_repr(path));
    };
    let Some(files) = files.filter(|f| !f.dir.is_empty()) else {
        bail!(
            "{what} names a shader file ({}) but the scene has no directory to \
             resolve it against; inline the GLSL in 'code'",
            py_repr(path)
        );
    };
    let root = normpath(&abspath(files.dir));
    let full = normpath(&join(&root, name));
    if files.reach == Reach::Inside && !inside(&full, &root) {
        bail!(
            "{what} shader {} is outside the scene directory",
            py_repr(path)
        );
    }
    match fs::read_to_string(&full) {
        Ok(code) => {
            crate::sources::record(&full);
            Ok(code)
        }
        Err(e) => bail!("{what} shader {}: {}", py_repr(path), strerror(&e)),
    }
}

/// The scene's shared GLSL. It arrives as a pass's own GLSL does:
/// 'code' inline, or a 'shader' file next to the scene. None is no common block.
/// "" is an empty block the console creates.
fn common_block(d: &Map<String, Value>, files: Option<Files>) -> Result<Option<String>> {
    let c = match d.get("common") {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let Value::Object(c) = c else {
        bail!(
            "'common' needs an object with 'code' (inline GLSL) or 'shader' \
             (a file next to the scene), got {}",
            py_repr(c)
        );
    };
    known_keys(c, "'common'", &COMMON_KEYS, &[])?;
    let code = c.get("code").filter(|v| !v.is_null());
    let shader = c.get("shader").filter(|v| !v.is_null());
    if code.is_none() == shader.is_none() {
        bail!(
            "'common' needs exactly one of 'code' (inline GLSL) or 'shader' \
             (a file next to the scene)"
        );
    }
    if let Some(s) = shader {
        return read_shader(s, files, "common").map(Some);
    }
    match code {
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => bail!("'common' needs GLSL text in 'code', got {}", py_repr(v)),
        None => Ok(None),
    }
}

/// The preamble uniform declaration of one param. A color is a vec3. A
/// string is host-side, and it declares no uniform.
fn param_uniform_lines(p: &Param) -> Vec<String> {
    if p.ty == "string" {
        return Vec::new();
    }
    let ty = match p.glsl_type() {
        Ok(t) => t,
        Err(_) => p.ty.as_str(),
    };
    vec![format!("uniform {ty} {};", p.name)]
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Color([f64; 3]),
    Vec(Vec<f64>),
    Str(String),
}

impl ParamValue {
    pub fn to_json(&self) -> Value {
        match self {
            ParamValue::Float(f) => Value::from(*f),
            ParamValue::Int(i) => Value::from(*i),
            ParamValue::Bool(b) => Value::from(*b),
            ParamValue::Color(c) => Value::from(c.to_vec()),
            ParamValue::Vec(v) => Value::from(v.clone()),
            ParamValue::Str(s) => Value::from(s.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: String,
    pub value: ParamValue,
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub step: Option<Value>,
    /// A string param offers these choices. The console renders them as a
    /// dropdown. Without them it renders a text input.
    pub options: Option<Vec<String>>,
    /// ghee declares this string param for the text of its name, and the
    /// param holds the text's words. A save writes them into the text's
    /// `value`.
    pub(crate) of_text: bool,
}

impl Param {
    pub fn new(
        name: &str,
        ty: &str,
        value: Value,
        min: Option<Value>,
        max: Option<Value>,
        step: Option<Value>,
        options: Option<Value>,
    ) -> Result<Param> {
        if !PARAM_TYPES.contains(&ty) {
            bail!(
                "param '{name}' has unknown type {}; known types: {}",
                py_str_repr(ty),
                PARAM_TYPES.join(", ")
            );
        }
        // A color is [r, g, b], each 0..1.
        let stored = if ty == "color" {
            let Some(nums) = number_list(&value, 3) else {
                bail!(
                    "color param '{name}' needs three numbers [r, g, b], got {}",
                    py_repr(&value)
                );
            };
            ParamValue::Color([nums[0], nums[1], nums[2]])
        } else if let Some(n) = vec_arity(ty) {
            let Some(nums) = number_list(&value, n) else {
                bail!(
                    "{ty} param '{name}' needs {n} numbers, got {}",
                    py_repr(&value)
                );
            };
            ParamValue::Vec(nums)
        } else if ty == "string" {
            match &value {
                Value::String(s) => ParamValue::Str(s.clone()),
                v => bail!("string param '{name}' needs a string, got {}", py_repr(v)),
            }
        // A bool is `uniform bool` in the shader and an int over the C ABI. The
        // engine has no bool type. glUniform1i sets a bool uniform.
        } else if ty == "bool" {
            match &value {
                Value::Bool(b) => ParamValue::Bool(*b),
                v => bail!(
                    "bool param '{name}' needs true or false, got {}",
                    py_repr(v)
                ),
            }
        } else {
            if !is_num(&value) {
                bail!(
                    "{ty} param '{name}' needs a number, got {}",
                    py_repr(&value)
                );
            }
            let f = value.as_f64().unwrap_or_default();
            if ty == "int" {
                ParamValue::Int(value.as_i64().unwrap_or(f as i64))
            } else {
                ParamValue::Float(f)
            }
        };
        for (k, r) in [("min", &min), ("max", &max), ("step", &step)] {
            if let Some(r) = r.as_ref().filter(|v| !v.is_null()) {
                if !is_num(r) {
                    bail!("param '{name}' {k} needs a number, got {}", py_repr(r));
                }
            }
        }
        let options = match options.filter(|o| !o.is_null()) {
            None => None,
            Some(o) => {
                if ty != "string" {
                    bail!("param '{name}': options belong to string params only");
                }
                match &o {
                    Value::Array(a) if a.iter().all(Value::is_string) => {
                        Some(a.iter().map(py_str).collect::<Vec<_>>())
                    }
                    _ => bail!("string param '{name}' options need a list of strings"),
                }
            }
        };
        Ok(Param {
            name: name.to_string(),
            ty: ty.to_string(),
            value: stored,
            min: min.filter(|v| !v.is_null()),
            max: max.filter(|v| !v.is_null()),
            step: step.filter(|v| !v.is_null()),
            options,
            of_text: false,
        })
    }

    /// A param with no min, max or step.
    fn plain(name: &str, ty: &str, value: ParamValue) -> Param {
        Param {
            name: name.to_string(),
            ty: ty.to_string(),
            value,
            min: None,
            max: None,
            step: None,
            options: None,
            of_text: false,
        }
    }

    pub fn float(name: &str, value: f64) -> Param {
        Param::plain(name, "float", ParamValue::Float(value))
    }

    pub fn int(name: &str, value: i64) -> Param {
        Param::plain(name, "int", ParamValue::Int(value))
    }

    pub fn bool(name: &str, value: bool) -> Param {
        Param::plain(name, "bool", ParamValue::Bool(value))
    }

    /// [r, g, b], each 0..1. It reaches a pass as a vec3.
    pub fn color(name: &str, rgb: [f64; 3]) -> Param {
        Param::plain(name, "color", ParamValue::Color(rgb))
    }

    /// The value count sets the type. 2 numbers are a vec2, 3 a vec3, 4 a vec4.
    pub fn vec(name: &str, values: &[f64]) -> Result<Param> {
        let Some((ty, _)) = VEC_ARITY.iter().find(|(_, n)| *n == values.len()) else {
            bail!(
                "vec param '{name}' needs 2, 3 or 4 numbers, got {}",
                values.len()
            );
        };
        Ok(Param::plain(name, ty, ParamValue::Vec(values.to_vec())))
    }

    /// A string param is host-side, and it has no uniform. Use `new` with
    /// `options` for the choices in the console's dropdown.
    pub fn string(name: &str, value: &str) -> Param {
        Param::plain(name, "string", ParamValue::Str(value.to_string()))
    }

    /// The string param that ghee declares for a text, holding `words`.
    fn words(name: &str, words: &str) -> Param {
        Param {
            of_text: true,
            ..Param::string(name, words)
        }
    }

    /// The GLSL type of the param's uniform. A color is a vec3.
    pub fn glsl_type(&self) -> Result<&'static str> {
        if self.ty == "string" {
            bail!("a string param is host-side, so it has no glsl_type");
        }
        if self.ty == "color" {
            return Ok("vec3");
        }
        if let Some((ty, _)) = VEC_ARITY.iter().find(|(t, _)| *t == self.ty) {
            return Ok(ty);
        }
        glsl_type_of(&self.ty).ok_or_else(|| anyhow!("unknown type {}", self.ty))
    }

    pub fn from_dict(name: &str, d: &Value) -> Result<Param> {
        let Value::Object(o) = d else {
            bail!("param '{name}' needs an object, got {}", py_repr(d));
        };
        known_keys(o, &format!("param '{name}'"), &PARAM_KEYS, &[])?;
        for k in ["type", "value"] {
            if !o.contains_key(k) {
                bail!("param '{name}' has no '{k}'");
            }
        }
        Param::new(
            name,
            &py_str(&o["type"]),
            o["value"].clone(),
            o.get("min").cloned(),
            o.get("max").cloned(),
            o.get("step").cloned(),
            o.get("options").cloned(),
        )
    }

    pub fn to_dict(&self) -> Value {
        let mut d = Map::new();
        d.insert("type".to_string(), Value::from(self.ty.clone()));
        d.insert("value".to_string(), self.value.to_json());
        for (k, v) in [("min", &self.min), ("max", &self.max), ("step", &self.step)] {
            if let Some(v) = v {
                d.insert(k.to_string(), v.clone());
            }
        }
        if let Some(o) = &self.options {
            d.insert("options".to_string(), Value::from(o.clone()));
        }
        Value::Object(d)
    }
}

fn number_list(v: &Value, n: usize) -> Option<Vec<f64>> {
    let Value::Array(a) = v else { return None };
    if a.len() != n || !a.iter().all(is_num) {
        return None;
    }
    Some(a.iter().map(|x| x.as_f64().unwrap_or_default()).collect())
}

/// Insertion-ordered name -> Param. A param's declaration order is its uniform's
/// order in a pass preamble.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamMap(Vec<(String, Param)>);

impl ParamMap {
    pub fn new() -> ParamMap {
        ParamMap(Vec::new())
    }

    pub fn get(&self, name: &str) -> Option<&Param> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Param> {
        self.0.iter_mut().find(|(n, _)| n == name).map(|(_, p)| p)
    }

    pub fn insert(&mut self, name: String, param: Param) {
        match self.0.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = param,
            None => self.0.push((name, param)),
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.0.iter().any(|(n, _)| n == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Param)> {
        self.0.iter().map(|(n, p)| (n, p))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(n, _)| n)
    }

    pub fn values(&self) -> impl Iterator<Item = &Param> {
        self.0.iter().map(|(_, p)| p)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn retain(&mut self, keep: impl Fn(&Param) -> bool) {
        self.0.retain(|(_, p)| keep(p));
    }
}

/// A named float input. The host feeds it, and the engine binds it as a texture
/// sampler.
///
/// Size N is a 1xN array. A read is one fetch at a computed index, so VC4 needs
/// no uniform-array unroll and imposes no size cap. Read it with `<name>_at(i)`
/// or `<name>_sample(t)`.
///
/// Size [w, h] is a 2D w x h mask, such as host-rasterized text or an image.
/// Read it with `<name>_sample2(uv)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayInput {
    pub name: String,
    pub size: ArraySize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArraySize {
    Linear(i64),
    Grid(i64, i64),
}

impl From<i64> for ArraySize {
    fn from(n: i64) -> ArraySize {
        ArraySize::Linear(n)
    }
}

impl From<(i64, i64)> for ArraySize {
    fn from((w, h): (i64, i64)) -> ArraySize {
        ArraySize::Grid(w, h)
    }
}

/// A size as a scene JSON value. The size check reads that form.
fn size_value(size: ArraySize) -> Value {
    match size {
        ArraySize::Linear(n) => Value::from(n),
        ArraySize::Grid(w, h) => Value::from(vec![w, h]),
    }
}

impl ArrayInput {
    pub fn new(name: &str, size: Value) -> Result<ArrayInput> {
        ArrayInput::build(name, size, false)
    }

    /// `pair` marks a [w, h] that from_dict rebuilds. The size check reports it
    /// as a tuple, `(w, h)`.
    fn build(name: &str, size: Value, pair: bool) -> Result<ArrayInput> {
        let stored = if let Value::Array(s) = &size {
            if s.len() != 2 || !s.iter().all(is_dim) {
                let got = if pair && s.len() == 2 {
                    format!("({}, {})", py_repr(&s[0]), py_repr(&s[1]))
                } else {
                    py_repr(&size)
                };
                bail!("array '{name}' size needs two ints in 1..{MAX_DIM} [w, h], got {got}");
            }
            let (w, h) = (s[0].as_i64().unwrap_or(0), s[1].as_i64().unwrap_or(0));
            if w * h > MAX_TEXELS {
                bail!("array '{name}' is {w}x{h}, over the {MAX_TEXELS} element cap");
            }
            ArraySize::Grid(w, h)
        } else if !is_dim(&size) {
            bail!(
                "array '{name}' size needs an int in 1..{MAX_DIM}, got {}",
                py_repr(&size)
            );
        } else {
            ArraySize::Linear(size.as_i64().unwrap_or(0))
        };
        Ok(ArrayInput {
            name: name.to_string(),
            size: stored,
        })
    }

    #[allow(dead_code)]
    pub fn is_2d(&self) -> bool {
        matches!(self.size, ArraySize::Grid(_, _))
    }

    pub fn from_dict(d: &Value) -> Result<ArrayInput> {
        let malformed = || anyhow!("an array needs 'name' and 'size', got {}", py_repr(d));
        let Value::Object(o) = d else {
            return Err(malformed());
        };
        let what = match o.get("name") {
            Some(n) => format!("array '{}'", py_str(n)),
            None => "an array".to_string(),
        };
        known_keys(o, &what, &ARRAY_KEYS, &[])?;
        if !o.contains_key("name") || !o.contains_key("size") {
            return Err(malformed());
        }
        let name = py_str(&o["name"]);
        let size = &o["size"];
        if let Value::Array(a) = size {
            if a.len() != 2 {
                bail!(
                    "array '{name}' size needs two ints [w, h], got {}",
                    py_repr(size)
                );
            }
            return ArrayInput::build(
                &name,
                Value::Array(vec![to_int(&a[0]), to_int(&a[1])]),
                true,
            );
        }
        ArrayInput::new(&name, to_int(size))
    }

    pub fn to_dict(&self) -> Value {
        let mut d = Map::new();
        d.insert("name".to_string(), Value::from(self.name.clone()));
        d.insert("size".to_string(), size_value(self.size));
        Value::Object(d)
    }
}

/// The preamble declarations of one array: its data texture, and the length
/// of a 1xN array.
fn array_lines(a: &ArrayInput) -> Vec<String> {
    let mut l = vec![format!("uniform sampler2D {}_tex;", a.name)];
    if let ArraySize::Linear(n) = a.size {
        l.push(format!("const int {}_size = {n};", a.name));
    }
    l
}

/// The preamble declarations of one image asset: its sampler and its size.
fn asset_lines(a: &Asset) -> Vec<String> {
    vec![
        format!("uniform sampler2D {};", a.name),
        format!("uniform vec2 {}_size;", a.name),
    ]
}

/// The names declared at file scope by the GLSL `lines` that ghee writes: a
/// uniform, a constant, a #define, or a function defined at the start of a
/// line.
fn declared_names(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for l in lines {
        if l.starts_with(char::is_whitespace) {
            continue;
        }
        let w: Vec<&str> = l
            .split(|c: char| c.is_whitespace() || c == '(' || c == ';')
            .filter(|w| !w.is_empty())
            .collect();
        let name = match w.first() {
            Some(&"#define") => w.get(1),
            Some(&"uniform" | &"const") => w.get(2),
            Some(_) if l.contains('(') => w.get(1),
            _ => None,
        };
        if let Some(n) = name.filter(|n| is_ident(n)) {
            out.push(n.to_string());
        }
    }
    out
}

/// The preamble readers for one declared array, bound as a data texture.
///
/// `_at(i)` is one fetch at a computed coord. VC4 needs no uniform-array unroll
/// and imposes no size cap.
///
/// V3D cannot linear-filter a float texture, so a GPU-filtered fetch is nearest
/// there and linear elsewhere. `hw_linear` says which GPU this is. Where the
/// filter works, `_sample` and `_sample2` are one fetch. Where it does not, they
/// read the exact texels and interpolate in the shader. That reproduces GL's
/// LINEAR and CLAMP_TO_EDGE: texel centres and clamped edges.
fn array_reader_lines(a: &ArrayInput, hw_linear: bool) -> Vec<String> {
    let n = &a.name;
    if let ArraySize::Grid(w, h) = a.size {
        if hw_linear {
            return vec![format!(
                "float {n}_sample2(vec2 uv){{ return texture2D({n}_tex, uv).r; }}"
            )];
        }
        return vec![
            format!("float {n}_sample2(vec2 uv){{"),
            format!(
                "  vec2 dim = vec2({}, {});",
                py_float(w as f64),
                py_float(h as f64)
            ),
            "  vec2 x = uv * dim - 0.5;".to_string(),
            "  vec2 i = floor(x);".to_string(),
            "  vec2 lo = (clamp(i, vec2(0.0), dim - 1.0) + 0.5) / dim;".to_string(),
            "  vec2 hi = (clamp(i + 1.0, vec2(0.0), dim - 1.0) + 0.5) / dim;".to_string(),
            format!("  float a = texture2D({n}_tex, vec2(lo.x, lo.y)).r;"),
            format!("  float b = texture2D({n}_tex, vec2(hi.x, lo.y)).r;"),
            format!("  float c = texture2D({n}_tex, vec2(lo.x, hi.y)).r;"),
            format!("  float d = texture2D({n}_tex, vec2(hi.x, hi.y)).r;"),
            "  vec2 f = x - i;".to_string(),
            "  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);".to_string(),
            "}".to_string(),
        ];
    }
    let at = format!(
        "float {n}_at(int i0){{ return texture2D({n}_tex, \
         vec2((float(i0)+0.5)/float({n}_size), 0.5)).r; }}"
    );
    if hw_linear {
        return vec![
            at,
            format!("float {n}_sample(float t){{ return texture2D({n}_tex, vec2(t, 0.5)).r; }}"),
        ];
    }
    vec![
        at,
        format!("float {n}_sample(float t){{"),
        format!("  float x = t * float({n}_size) - 0.5;"),
        "  float i = floor(x);".to_string(),
        format!("  float last = float({n}_size - 1);"),
        format!("  float a = {n}_at(int(clamp(i, 0.0, last)));"),
        format!("  float b = {n}_at(int(clamp(i + 1.0, 0.0, last)));"),
        "  return mix(a, b, x - i);".to_string(),
        "}".to_string(),
    ]
}

/// A named image or font file. Every pass samples an image as
/// `uniform sampler2D <name>` and reads a `vec2 <name>_size` of (width,
/// height). A text names a font, and only the text reads it.
///
/// A .raw file is bare pixels and declares its layout. A .png or .jpg file
/// carries its own. A .ttf or .otf file is a font.
#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    pub name: String,
    pub file: String,
    /// The asset's `sampling`: GL_LINEAR when true, nearest by default.
    pub linear: bool,
    /// The declared layout of a .raw file. None for a decoded format.
    pub raw: Option<RawLayout>,
    /// The decoded pixels. load_assets fills them from the file.
    pub image: Option<crate::assets::Image>,
    /// A font file's bytes, or why they failed to load. load_assets fills
    /// them.
    pub font: Option<crate::text::Font>,
    /// The filter that declared the asset, empty for the scene's own. The
    /// filter loads it, its name carries the filter's as a prefix, and the
    /// scene's JSON leaves it out.
    pub from_filter: String,
}

/// A .raw file's declared pixel layout. f32 pixels are little-endian on
/// disk and need a GL with float textures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawLayout {
    pub width: i64,
    pub height: i64,
    pub components: i64,
    pub is_f32: bool,
}

impl Asset {
    /// The name the declaring code uses. A filter's asset drops the filter
    /// prefix it runs under.
    pub fn local_name(&self) -> &str {
        if self.from_filter.is_empty() {
            return &self.name;
        }
        self.name
            .strip_prefix(&self.from_filter)
            .and_then(|r| r.strip_prefix('_'))
            .unwrap_or(&self.name)
    }

    /// A .ttf or .otf file.
    pub fn is_font(&self) -> bool {
        crate::text::is_font_file(&self.file)
    }

    /// Decode `bytes` as this asset's declared format. `what` names the source
    /// in the error. A font that fails to parse keeps the error for its texts,
    /// which draw error boxes.
    pub fn decode(&mut self, bytes: &[u8], what: &str) -> Result<()> {
        if self.is_font() {
            self.font = Some(
                crate::text::font(bytes)
                    .map_err(|e| format!("asset '{}' ({what}): unreadable font: {e}", self.name)),
            );
            return Ok(());
        }
        let decoded = match &self.raw {
            Some(r) => crate::assets::decode_raw(
                bytes,
                r.width as u32,
                r.height as u32,
                r.components as u32,
                r.is_f32,
            ),
            None if self.file.to_ascii_lowercase().ends_with(".png") => {
                crate::assets::decode_png(bytes)
            }
            None => crate::assets::decode_jpeg(bytes),
        }
        .map_err(|e| anyhow!("asset '{}' ({what}): {e}", self.name))?;
        self.image = Some(decoded);
        Ok(())
    }

    pub fn from_dict(name: &str, d: &Value) -> Result<Asset> {
        let Value::Object(o) = d else {
            bail!("asset '{name}' needs an object, got {}", py_repr(d));
        };
        if o.contains_key("filter") {
            bail!("asset '{name}' takes 'nearest' or 'linear' in 'sampling', got 'filter'");
        }
        if o.contains_key("from_filter") {
            bail!("asset '{name}' takes no 'from_filter': a filter declares its own assets");
        }
        known_keys(o, &format!("asset '{name}'"), &ASSET_KEYS, &[])?;
        let file = match o.get("file") {
            None => bail!("asset '{name}' has no 'file'"),
            Some(Value::String(s)) => s.clone(),
            Some(v) => bail!("asset '{name}' 'file' needs a filename, got {}", py_repr(v)),
        };
        let linear = match o.get("sampling").filter(|v| !v.is_null()) {
            None => false,
            Some(Value::String(s)) if s == "nearest" => false,
            Some(Value::String(s)) if s == "linear" => true,
            Some(v) => bail!(
                "asset '{name}' sampling needs 'nearest' or 'linear', got {}",
                py_repr(v)
            ),
        };
        let ext = file
            .rsplit_once('.')
            .map_or(String::new(), |(_, e)| e.to_ascii_lowercase());
        let raw = match ext.as_str() {
            "raw" => {
                for k in ["width", "height", "format"] {
                    if !o.contains_key(k) {
                        bail!("raw asset '{name}' has no '{k}'");
                    }
                }
                let mut dims = [0i64; 2];
                for (slot, k) in dims.iter_mut().zip(["width", "height"]) {
                    let v = to_int(&o[k]);
                    if !is_dim(&v) {
                        bail!(
                            "asset '{name}' {k} needs an int in 1..{MAX_DIM}, got {}",
                            py_repr(&o[k])
                        );
                    }
                    *slot = v.as_i64().unwrap_or(0);
                }
                let (w, h) = (dims[0], dims[1]);
                if w * h > MAX_TEXELS {
                    bail!("asset '{name}' is {w}x{h}, over the {MAX_TEXELS} pixel cap");
                }
                let is_f32 = match &o["format"] {
                    Value::String(s) if s == "u8" => false,
                    Value::String(s) if s == "f32" => true,
                    v => bail!(
                        "asset '{name}' format needs 'u8' or 'f32', got {}",
                        py_repr(v)
                    ),
                };
                let components = match o.get("components").filter(|v| !v.is_null()) {
                    None => 1,
                    Some(v) => match to_int(v).as_i64() {
                        Some(c @ (1 | 3 | 4)) => c,
                        _ => bail!(
                            "asset '{name}' components needs 1, 3 or 4, got {}",
                            py_repr(v)
                        ),
                    },
                };
                Some(RawLayout {
                    width: w,
                    height: h,
                    components,
                    is_f32,
                })
            }
            "png" | "jpg" | "jpeg" => {
                for k in ["width", "height", "format", "components"] {
                    if o.contains_key(k) {
                        bail!("asset '{name}' takes no '{k}', a .{ext} file carries its own");
                    }
                }
                None
            }
            "ttf" | "otf" => {
                for k in ["width", "height", "format", "components", "sampling"] {
                    if o.contains_key(k) {
                        bail!("asset '{name}' takes no '{k}', a .{ext} file is a font");
                    }
                }
                None
            }
            _ => bail!(
                "asset '{name}' file {} needs a .raw, .png, .jpg, .jpeg, .ttf or .otf extension",
                py_str_repr(&file)
            ),
        };
        Ok(Asset {
            name: name.to_string(),
            file,
            linear,
            raw,
            image: None,
            font: None,
            from_filter: String::new(),
        })
    }

    pub fn to_dict(&self) -> Value {
        let mut d = Map::new();
        d.insert("file".to_string(), Value::from(self.file.clone()));
        if let Some(r) = &self.raw {
            d.insert("width".to_string(), Value::from(r.width));
            d.insert("height".to_string(), Value::from(r.height));
            d.insert(
                "format".to_string(),
                Value::from(if r.is_f32 { "f32" } else { "u8" }),
            );
            if r.components != 1 {
                d.insert("components".to_string(), Value::from(r.components));
            }
        }
        if self.linear {
            d.insert("sampling".to_string(), Value::from("linear"));
        }
        Value::Object(d)
    }
}

/// A numbered sampler -> iChannel<index>, bound to a pass's output buffer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Channel {
    pub buffer: Option<String>,
}

impl Channel {
    pub fn new(buffer: Option<&str>) -> Channel {
        Channel {
            buffer: buffer.map(str::to_string),
        }
    }

    /// `what` names the channel in an error, as "channel 0".
    pub fn from_dict(what: &str, d: &Value) -> Result<Channel> {
        let Value::Object(o) = d else {
            bail!(
                "a channel needs an object like {{'buffer': 'pass'}}, got {}",
                py_repr(d)
            );
        };
        known_keys(o, what, &CHANNEL_KEYS, &[])?;
        Ok(Channel {
            buffer: o.get("buffer").filter(|v| !v.is_null()).map(py_str),
        })
    }

    pub fn to_dict(&self) -> Value {
        let mut d = Map::new();
        d.insert(
            "buffer".to_string(),
            match &self.buffer {
                Some(b) => Value::from(b.clone()),
                None => Value::Null,
            },
        );
        Value::Object(d)
    }
}

/// A text drawn as one quad a glyph into its pass's buffer, after the pass's
/// own quad.
///
/// The vertex GLSL defines `vec2 glyph_place(vec2 em)`. It takes a point of
/// the block, in em from the block's top left with y down, and returns the
/// pass pixel where that point lands, in the space of
/// `v_texCoord * resolution`. The fragment GLSL colours a glyph's pixels with
/// its own `void main()`, premultiplied by alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// The layer's own name, unique among the layers of its pass, as
    /// is_layer_name() takes it. None goes by its text, as layer_labels() says.
    pub name: Option<String>,
    /// A name in the scene's texts.
    pub text: String,
    pub vertex: Vertex,
    /// None draws the letters white at full coverage, premultiplied.
    pub fragment: Option<String>,
    pub blend: Blend,
    /// How far the quads reach past the ink, in em, for an effect around the
    /// outline.
    pub pad: f64,
    /// The layer paints the text's colour glyphs. None paints them from the
    /// last layer of the text in its pass alone, as Pass::paints_emoji says.
    pub emoji: Option<bool>,
}

/// Where a layer puts its glyphs.
#[derive(Clone, Debug, PartialEq)]
pub enum Vertex {
    /// The layer's `vertex`: GLSL that defines `vec2 glyph_place(vec2 em)`.
    Glsl(String),
    /// The layer's `px_per_em`: the block centred in the pass at this many
    /// pixels per em.
    Centred(f64),
}

/// The fragment GLSL of a layer with none of its own.
const WHITE: &str = "void main() { gl_FragColor = vec4(glyph_cover()); }";

/// A layer's `name`: ASCII letters, digits, '_' and '-'. A name holds no space,
/// so it never reads as the label of a layer without one, such as "t 2".
pub fn is_layer_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// What each layer of a pass goes by, from its name and its text, in order:
/// its name, else its text, numbered from the text's second layer without a
/// name, as "t", "t 2", "t 3". None for a layer with neither.
pub fn layer_labels<'a>(
    layers: impl IntoIterator<Item = (Option<&'a str>, Option<&'a str>)>,
) -> Vec<Option<String>> {
    let mut seen: Vec<&str> = Vec::new();
    layers
        .into_iter()
        .map(|(name, text)| match (name, text) {
            (Some(n), _) => Some(n.to_string()),
            (None, Some(t)) => {
                seen.push(t);
                Some(match seen.iter().filter(|s| **s == t).count() {
                    1 => t.to_string(),
                    n => format!("{t} {n}"),
                })
            }
            (None, None) => None,
        })
        .collect()
}

/// Layer `k` of pass `pass` in an error: "pass 'main' layer 'glow'" by its
/// label, else by its place, "the 2nd layer of pass 'main'".
fn layer_what(pass: &str, k: usize, label: Option<&str>) -> String {
    match label {
        Some(l) => format!("pass '{pass}' layer '{l}'"),
        None => {
            let n = k + 1;
            let th = match (n % 10, n % 100) {
                (_, 11..=13) => "th",
                (1, _) => "st",
                (2, _) => "nd",
                (3, _) => "rd",
                _ => "th",
            };
            format!("the {n}{th} layer of pass '{pass}'")
        }
    }
}

impl Layer {
    /// The GLSL that defines glyph_place for the layer.
    pub fn vertex_glsl(&self) -> String {
        match &self.vertex {
            Vertex::Glsl(v) => v.clone(),
            Vertex::Centred(px) => {
                let (name, px) = (&self.text, py_float(*px));
                format!(
                    "vec2 glyph_place(vec2 em) {{ return (resolution - {name}_size * {px}) * 0.5 \
                     + em * {px}; }}"
                )
            }
        }
    }

    /// The layer's fragment GLSL, its own or the white letters.
    pub fn fragment_glsl(&self) -> &str {
        self.fragment.as_deref().unwrap_or(WHITE)
    }

    /// `what` names the layer in an error, as "pass 'main' layer 'glow'".
    pub fn from_dict(what: &str, d: &Value) -> Result<Layer> {
        let Value::Object(o) = d else {
            bail!("{what} needs an object, got {}", py_repr(d));
        };
        for (old, key) in [("place", "vertex"), ("code", "fragment")] {
            if o.contains_key(old) {
                bail!("{what} takes its {key} GLSL in '{key}', got '{old}'");
            }
        }
        if o.contains_key("alpha") {
            bail!("{what} takes no 'alpha': its alpha blends by 'blend', as its colour does");
        }
        let text = match o.get("text") {
            Some(Value::String(t)) if !t.trim().is_empty() => format!("text '{t}'"),
            _ => "its text".to_string(),
        };
        let elsewhere = [
            keys_on(&LAYER_KEYS, &TEXT_KEYS, &text),
            keys_on(&LAYER_KEYS, &PASS_KEYS, "its pass"),
        ]
        .concat();
        known_keys(o, what, &LAYER_KEYS, &elsewhere)?;
        let glsl = |key: &str, holds: &str| match o.get(key) {
            Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.clone()),
            Some(v) => bail!("{what} '{key}' needs {holds}, got {}", py_repr(v)),
            None => bail!("{what} has no '{key}', {holds}"),
        };
        let text = glsl("text", "the name of a scene text")?;
        let name = match o.get("name").filter(|v| !v.is_null()) {
            None => None,
            Some(Value::String(n)) if is_layer_name(n) => Some(n.clone()),
            Some(v) => bail!(
                "{what} name needs one or more ASCII letters, digits, '_' or '-', got {}",
                py_repr(v)
            ),
        };
        let given = |key: &str| o.get(key).filter(|v| !v.is_null());
        let vertex = match (given("vertex"), given("px_per_em")) {
            (Some(_), Some(_)) => bail!("{what} takes 'vertex' or 'px_per_em', got both"),
            (Some(_), None) => Vertex::Glsl(glsl(
                "vertex",
                "vertex GLSL that defines vec2 glyph_place(vec2 em)",
            )?),
            (None, Some(v)) => match v.as_f64() {
                Some(f) if f.is_finite() && f > 0.0 => Vertex::Centred(f),
                _ => bail!(
                    "{what} px_per_em needs a number of pixels above 0, got {}",
                    py_repr(v)
                ),
            },
            (None, None) => bail!(
                "{what} needs 'px_per_em', the pixels per em of the words centred in \
                 the pass, or 'vertex', GLSL that defines vec2 glyph_place(vec2 em)"
            ),
        };
        let fragment = match given("fragment") {
            None => None,
            Some(_) => Some(glsl("fragment", "fragment GLSL with its own void main()")?),
        };
        let blend = match o.get("blend").filter(|v| !v.is_null()) {
            None => Blend::Over,
            Some(Value::String(b)) if b == "over" => Blend::Over,
            Some(Value::String(b)) if b == "add" => Blend::Add,
            Some(Value::String(b)) if b == "replace" => Blend::Replace,
            Some(v) => bail!(
                "{what} blend needs 'over', 'add' or 'replace', got {}",
                py_repr(v)
            ),
        };
        let pad = match o.get("pad").filter(|v| !v.is_null()) {
            None => 0.0,
            Some(v) => match v.as_f64() {
                Some(f) if f.is_finite() && f >= 0.0 => f,
                _ => bail!(
                    "{what} pad needs a number of em, 0 or more, got {}",
                    py_repr(v)
                ),
            },
        };
        let emoji = match o.get("emoji").filter(|v| !v.is_null()) {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(v) => bail!("{what} emoji needs true or false, got {}", py_repr(v)),
        };
        Ok(Layer {
            name,
            text,
            vertex,
            fragment,
            blend,
            pad,
            emoji,
        })
    }

    /// `last` is the default of `emoji` at the layer's place in its pass, and
    /// the dict holds `emoji` only when the layer sets it otherwise.
    pub fn to_dict(&self, last: bool) -> Value {
        let mut d = Map::new();
        if let Some(n) = &self.name {
            d.insert("name".to_string(), Value::from(n.clone()));
        }
        d.insert("text".to_string(), Value::from(self.text.clone()));
        match &self.vertex {
            Vertex::Glsl(v) => d.insert("vertex".to_string(), Value::from(v.clone())),
            Vertex::Centred(px) => d.insert("px_per_em".to_string(), Value::from(*px)),
        };
        if let Some(f) = &self.fragment {
            d.insert("fragment".to_string(), Value::from(f.clone()));
        }
        let blend = match self.blend {
            Blend::Over => None,
            Blend::Add => Some("add"),
            Blend::Replace => Some("replace"),
        };
        if let Some(b) = blend {
            d.insert("blend".to_string(), Value::from(b));
        }
        if self.pad != 0.0 {
            d.insert("pad".to_string(), Value::from(self.pad));
        }
        if let Some(e) = self.emoji.filter(|e| *e != last) {
            d.insert("emoji".to_string(), Value::from(e));
        }
        Value::Object(d)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    pub name: String,
    /// The pass's fragment shader. A pass with layers may leave it empty: the
    /// pass then starts each frame transparent and draws its layers alone.
    pub code: String,
    pub arrays: Vec<ArrayInput>,
    pub channels: Vec<Channel>,
    pub params: ParamMap,
    /// A non-empty name marks a pass as a named filter's output. The console
    /// renders such a pass read-only.
    pub filter: String,
    /// The [w, h] of this pass's render. None is the renderer's size. A
    /// consumer samples a sized pass through normalized UVs, so it ignores
    /// the size. The output pass leaves this None.
    pub size: Option<(i64, i64)>,
    /// The pass's buffer holds a mip chain, built after the pass renders each
    /// frame, and every channel that reads it samples trilinearly.
    pub mipmaps: bool,
    /// Texts drawn into this pass's buffer after its quad, in order.
    pub layers: Vec<Layer>,
}

impl Pass {
    pub fn new(
        name: &str,
        code: &str,
        arrays: Vec<ArrayInput>,
        channels: Vec<Channel>,
        params: ParamMap,
        filter: &str,
        size: Option<Value>,
    ) -> Result<Pass> {
        let size = match size.filter(|v| !v.is_null()) {
            None => None,
            Some(s) => {
                let dims = match &s {
                    Value::Array(a) if a.len() == 2 && a.iter().all(is_dim) => a,
                    _ => bail!(
                        "pass '{name}' size needs two positive ints [w, h] up to \
                         {MAX_DIM}, got {}",
                        py_repr(&s)
                    ),
                };
                let (w, h) = (dims[0].as_i64().unwrap_or(0), dims[1].as_i64().unwrap_or(0));
                if w * h > MAX_TEXELS {
                    bail!("pass '{name}' is {w}x{h}, over the {MAX_TEXELS} pixel cap");
                }
                Some((w, h))
            }
        };
        Ok(Pass {
            name: name.to_string(),
            code: code.to_string(),
            arrays,
            channels,
            params,
            filter: filter.to_string(),
            size,
            mipmaps: false,
            layers: Vec::new(),
        })
    }

    /// A pass with a name and GLSL only. Every other field takes its default.
    pub fn of(name: &str, code: &str) -> Result<Pass> {
        Pass::new(
            name,
            code,
            Vec::new(),
            Vec::new(),
            ParamMap::new(),
            "",
            None,
        )
    }

    /// False for a pass of layers alone, with no quad of its own.
    pub fn has_code(&self) -> bool {
        !self.code.trim().is_empty()
    }

    /// The error for a pass with no code and no layers, which draws nothing.
    pub(crate) fn empty(&self) -> Option<String> {
        (!self.has_code() && self.layers.is_empty()).then(|| {
            format!(
                "pass '{}' has no 'code' (inline GLSL), no 'shader' (a file next to \
                 the scene) and no 'layers'",
                self.name
            )
        })
    }

    /// True when a channel reads this pass's own buffer.
    #[allow(dead_code)]
    pub fn reads_self(&self) -> bool {
        self.channels
            .iter()
            .any(|c| c.buffer.as_deref() == Some(self.name.as_str()))
    }

    /// A `shader` file resolves in `base_dir`, with the reach of a scene that
    /// its user runs.
    pub fn from_dict(d: &Value, base_dir: Option<&str>) -> Result<Pass> {
        Pass::parse(d, any_file(base_dir))
    }

    fn parse(d: &Value, files: Option<Files>) -> Result<Pass> {
        let Value::Object(o) = d else {
            bail!("a pass needs a JSON object, got {}", py_repr(d));
        };
        let name = o.get("name").map_or("main".to_string(), py_str);
        let elsewhere = [
            keys_on(&PASS_KEYS, &LAYER_KEYS, "a layer"),
            keys_on(&PASS_KEYS, &TEXT_KEYS, "a text"),
            keys_on(&PASS_KEYS, &ASSET_KEYS, "an asset"),
            keys_on(&PASS_KEYS, &CHANNEL_KEYS, "a channel"),
            keys_on(&PASS_KEYS, &PARAM_KEYS, "a param"),
            keys_on(&PASS_KEYS, &SCENE_KEYS, "the scene"),
        ]
        .concat();
        known_keys(o, &format!("pass '{name}'"), &PASS_KEYS, &elsewhere)?;
        let code = match (o.get("code").filter(|v| !v.is_null()), o.get("shader")) {
            (Some(v), _) => py_str(v),
            (None, Some(s)) => read_shader(s, files, &format!("pass '{name}'"))?,
            (None, None) => String::new(),
        };
        let (arrays, channels, params) =
            (|| -> Result<(Vec<ArrayInput>, Vec<Channel>, ParamMap)> {
                let arrays = list_of(o, "arrays")?
                    .iter()
                    .map(ArrayInput::from_dict)
                    .collect::<Result<Vec<_>>>()?;
                let channels = list_of(o, "channels")?
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Channel::from_dict(&format!("channel {i}"), c))
                    .collect::<Result<Vec<_>>>()?;
                let mut params = ParamMap::new();
                for (n, pd) in obj_of(o, "params")? {
                    params.insert(n.clone(), Param::from_dict(n, pd)?);
                }
                Ok((arrays, channels, params))
            })()
            .map_err(|e| anyhow!("pass '{name}': {e}"))?;
        let mut pass = Pass::new(
            &name,
            &code,
            arrays,
            channels,
            params,
            &o.get("filter")
                .filter(|v| !v.is_null())
                .map_or(String::new(), py_str),
            o.get("size").cloned(),
        )?;
        pass.mipmaps = match o.get("mipmaps").filter(|v| !v.is_null()) {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(v) => bail!(
                "pass '{name}' mipmaps needs true or false, got {}",
                py_repr(v)
            ),
        };
        let layers = list_of(o, "layers").map_err(|e| anyhow!("pass '{name}': {e}"))?;
        // A layer goes by its name and its text in an error about any key,
        // so each comes from the dict before the layer parses.
        let labels = layer_labels(layers.iter().map(|dd| {
            let key = |k: &str| dd.get(k).and_then(Value::as_str);
            (
                key("name").filter(|n| is_layer_name(n)),
                key("text").filter(|t| !t.trim().is_empty()),
            )
        }));
        pass.layers = layers
            .iter()
            .zip(&labels)
            .enumerate()
            .map(|(k, (dd, l))| Layer::from_dict(&layer_what(&name, k, l.as_deref()), dd))
            .collect::<Result<Vec<_>>>()?;
        // A filter is written apart from any scene, and a text belongs to one.
        if !pass.filter.is_empty() && !pass.layers.is_empty() {
            bail!(
                "pass '{name}' is filter '{}', and a filter pass takes no layers",
                pass.filter
            );
        }
        // A filter pass takes its code from its filter, which fills it in later.
        if let Some(e) = pass.empty().filter(|_| pass.filter.is_empty()) {
            bail!("{e}");
        }
        Ok(pass)
    }

    pub fn to_dict(&self) -> Value {
        let mut out = Map::new();
        out.insert("name".to_string(), Value::from(self.name.clone()));
        if self.has_code() || self.layers.is_empty() {
            out.insert("code".to_string(), Value::from(self.code.clone()));
        }
        if let Some((w, h)) = self.size {
            out.insert("size".to_string(), Value::from(vec![w, h]));
        }
        if self.mipmaps {
            out.insert("mipmaps".to_string(), Value::from(true));
        }
        if !self.arrays.is_empty() {
            out.insert(
                "arrays".to_string(),
                Value::from(
                    self.arrays
                        .iter()
                        .map(ArrayInput::to_dict)
                        .collect::<Vec<_>>(),
                ),
            );
        }
        if !self.channels.is_empty() {
            out.insert(
                "channels".to_string(),
                Value::from(
                    self.channels
                        .iter()
                        .map(Channel::to_dict)
                        .collect::<Vec<_>>(),
                ),
            );
        }
        if !self.params.is_empty() {
            let mut m = Map::new();
            for (n, p) in self.params.iter() {
                m.insert(n.clone(), p.to_dict());
            }
            out.insert("params".to_string(), Value::Object(m));
        }
        if !self.filter.is_empty() {
            out.insert("filter".to_string(), Value::from(self.filter.clone()));
        }
        if !self.layers.is_empty() {
            let layers: Vec<Value> = (0..self.layers.len())
                .map(|k| self.layers[k].to_dict(self.last_of_text(k)))
                .collect();
            out.insert("layers".to_string(), Value::from(layers));
        }
        Value::Object(out)
    }

    /// What each layer of the pass goes by, as layer_labels() gives it.
    pub fn layer_labels(&self) -> Vec<String> {
        layer_labels(
            self.layers
                .iter()
                .map(|d| (d.name.as_deref(), Some(d.text.as_str()))),
        )
        .into_iter()
        .flatten()
        .collect()
    }

    /// Layer `k` in an error, as "pass 'main' layer 'glow'".
    pub fn layer_what(&self, k: usize) -> String {
        layer_what(
            &self.name,
            k,
            self.layer_labels().get(k).map(String::as_str),
        )
    }

    /// The error for two layers of the pass that go by one label.
    fn twin_layers(&self) -> Option<String> {
        let labels = self.layer_labels();
        let (_, l) = labels
            .iter()
            .enumerate()
            .find(|(k, l)| labels[..*k].contains(l))?;
        Some(format!(
            "pass '{}' has two layers that go by '{l}'",
            self.name
        ))
    }

    /// True when layer `k` is the last layer of its text in this pass.
    fn last_of_text(&self, k: usize) -> bool {
        let text = &self.layers[k].text;
        !self.layers[k + 1..].iter().any(|d| d.text == *text)
    }

    /// True when layer `k` paints its text's colour glyphs: its `emoji`, else
    /// when it is the last layer of its text in this pass.
    pub fn paints_emoji(&self, k: usize) -> bool {
        self.layers[k].emoji.unwrap_or_else(|| self.last_of_text(k))
    }

    /// `hw_linear` is the target GPU's Renderer.data_texture_linear. False emits
    /// the array readers that hold on any GPU. `assets` are the scene's assets.
    /// A pass of the scene's own declares the scene's assets. A filter pass
    /// declares only its filter's, each under the name the filter wrote.
    pub fn glsl(
        &self,
        scene_params: Option<&ParamMap>,
        common: Option<&str>,
        assets: &[Asset],
        texts: &[Text],
        hw_linear: bool,
    ) -> String {
        let l = self.preamble(scene_params, assets, texts, hw_linear);
        // An author writes a filter on its own, and ghee appends it to any
        // scene. The scene's common block never reaches it, so a filter name
        // never collides with a scene name.
        let body = match common {
            Some(c) if !c.is_empty() && self.filter.is_empty() => format!("{c}\n\n{}", self.code),
            _ => self.code.clone(),
        };
        format!("{}\n\n{body}", l.join("\n"))
    }

    /// The vertex and the fragment shader of layer `k`. Each holds the pass's
    /// preamble and the scene's common block, then the layer's GLSL for its
    /// stage, as layer_vertex() and the fragment shader below lay them out.
    ///
    /// `scene` is the scene of the pass, and `hw_linear` as glsl() takes it.
    pub fn layer_glsl(&self, k: usize, scene: &Scene, hw_linear: bool) -> (String, String) {
        let d = &self.layers[k];
        let text = scene.texts.iter().find(|t| t.name == d.text);
        let main = match text {
            Some(t) if t.fill == Fill::Mtsdf => crate::text::mtsdf_vertex_main(
                &d.text,
                d.pad,
                t.px_per_em,
                scene.atlas_range(&d.text),
            ),
            _ => crate::text::slug_vertex_main(d.pad),
        };
        let v = self.layer_vertex(d, scene, hw_linear, main);
        let mut f = self.preamble(Some(&scene.params), &scene.assets, &scene.texts, hw_linear);
        f.push(String::new());
        if let Some(c) = scene.common.as_deref().filter(|c| !c.is_empty()) {
            f.push(c.to_string());
            f.push(String::new());
        }
        let (head, tail) = match text.map(|t| t.fill) {
            Some(Fill::Mtsdf) => (
                crate::text::mtsdf_fragment_head(&d.text),
                crate::text::mtsdf_fragment_tail(&d.text),
            ),
            _ => (
                crate::text::slug_fragment_head(&d.text),
                crate::text::slug_fragment_tail(&d.text),
            ),
        };
        f.extend(head);
        f.push(String::new());
        f.push(d.fragment_glsl().to_string());
        f.push(String::new());
        f.extend(tail);
        (v, f.join("\n"))
    }

    /// The vertex and the fragment shader of the error boxes of layer `k`: the
    /// layer's own vertex GLSL with the slug fill's vertex main and no pad, so
    /// each box lands where a glyph would go, and ghee's hazard stripes.
    pub fn layer_box_glsl(&self, k: usize, scene: &Scene, hw_linear: bool) -> (String, String) {
        let main = crate::text::slug_vertex_main(0.0);
        let v = self.layer_vertex(&self.layers[k], scene, hw_linear, main);
        let mut f: Vec<String> = PRECISION.iter().map(|l| l.to_string()).collect();
        f.extend(crate::text::box_fragment());
        (v, f.join("\n"))
    }

    /// The vertex and the fragment shader of the colour batch of layer `k`:
    /// the layer's own vertex GLSL, and ghee's read of the text's colour atlas.
    pub fn layer_colour_glsl(&self, k: usize, scene: &Scene, hw_linear: bool) -> (String, String) {
        let d = &self.layers[k];
        let main = crate::colour::vertex_main(&d.text);
        let v = self.layer_vertex(d, scene, hw_linear, main);
        let mut f: Vec<String> = PRECISION.iter().map(|l| l.to_string()).collect();
        f.extend(crate::colour::fragment(&d.text));
        (v, f.join("\n"))
    }

    /// The vertex shader of layer `d`, with `main` after its vertex GLSL. It
    /// declares what the pass's preamble declares, at the same precision,
    /// since GLSL ES links a uniform declared by both stages only when the two
    /// declarations match, and the scene's common block follows, so the
    /// common block compiles as it does in a pass. A text's coverage reader
    /// is long to compile on VC4, so the stage holds it only for a text whose
    /// `<name>_cover` the common block or the layer's vertex GLSL names.
    fn layer_vertex(&self, d: &Layer, scene: &Scene, hw_linear: bool, main: Vec<String>) -> String {
        let common = scene.common.as_deref().unwrap_or_default();
        let vertex = d.vertex_glsl();
        let mut v = self.preamble_reading(
            Some(&scene.params),
            &scene.assets,
            &scene.texts,
            hw_linear,
            |t| {
                let cover = format!("{}_cover", t.name);
                common.contains(&cover) || vertex.contains(&cover)
            },
        );
        // The fragment stage's int defaults to mediump and the vertex stage's
        // to highp, so an int uniform declared in both needs this to link.
        v.insert(PRECISION.len(), "precision mediump int;".to_string());
        v.push("uniform float ghee_vflip;".to_string());
        v.push(String::new());
        if let Some(c) = scene.common.as_deref().filter(|c| !c.is_empty()) {
            v.push(c.to_string());
            v.push(String::new());
        }
        v.extend(crate::text::layer_vertex_head());
        v.push(String::new());
        v.push(vertex);
        v.push(String::new());
        v.extend(main);
        v.join("\n")
    }

    fn preamble(
        &self,
        scene_params: Option<&ParamMap>,
        assets: &[Asset],
        texts: &[Text],
        hw_linear: bool,
    ) -> Vec<String> {
        self.preamble_reading(scene_params, assets, texts, hw_linear, |_| true)
    }

    /// The preamble, with the coverage reader of each text for which `reads`
    /// holds, and that text's `<name>_size` alone for any other.
    fn preamble_reading(
        &self,
        scene_params: Option<&ParamMap>,
        assets: &[Asset],
        texts: &[Text],
        hw_linear: bool,
        reads: impl Fn(&Text) -> bool,
    ) -> Vec<String> {
        let mut l: Vec<String> = PRECISION.iter().map(|l| l.to_string()).collect();
        l.extend(
            [
                "varying vec2 v_texCoord;",
                "uniform vec2 resolution;",
                "uniform float time;",
                "uniform float DT;",
                "uniform float frame;",
                // VC4 shears gl_FragCoord.xy by about 1/3 px per row when a pass
                // samples its own output. The interpolated varying stays within
                // 0.04 px of the pixel centre. This define maps gl_FragCoord onto
                // the varying, so scene code that reads gl_FragCoord.xy/resolution
                // samples feedback without drift.
                "#define gl_FragCoord vec4(v_texCoord * resolution, 0.0, 1.0)",
            ]
            .map(str::to_string),
        );
        for a in &self.arrays {
            l.extend(array_lines(a));
        }
        for i in 0..self.channels.len() {
            l.push(format!("uniform sampler2D iChannel{i};"));
        }
        let own: Vec<&Asset> = assets
            .iter()
            .filter(|a| a.from_filter == self.filter && !a.is_font())
            .collect();
        for a in &own {
            l.extend(asset_lines(a));
        }
        for a in &self.arrays {
            l.extend(array_reader_lines(a, hw_linear));
        }
        // A text belongs to the scene, so a filter pass never reads one.
        if self.filter.is_empty() {
            for t in texts {
                match reads(t) {
                    true => l.extend(crate::text::reader_lines(&t.name)),
                    false => l.push(format!("uniform vec2 {}_size;", t.name)),
                }
            }
        }
        // A uniform named after a GLSL built-in function, such as mix, hides
        // the function from every line after it, so the params follow the
        // readers above, which call such functions. Every pass takes the same
        // scene param lines, ahead of its own params.
        for p in scene_params.iter().flat_map(|sp| sp.values()) {
            l.extend(param_uniform_lines(p));
        }
        for p in self.params.values() {
            l.extend(param_uniform_lines(p));
        }
        // A filter's asset runs under its filter's prefix, so it never meets a
        // scene name. The defines come last, after every declaration, and
        // give the filter code back the name its author wrote.
        for a in &own {
            let local = a.local_name();
            if local != a.name {
                l.push(format!("#define {local} {}", a.name));
                l.push(format!("#define {local}_size {}_size", a.name));
            }
        }
        l
    }
}

// mediump cannot hold a texel coord like (col+0.5)/w. A feedback pass that
// resamples its own output then drifts a fraction of a pixel per frame. highp
// keeps a ping-pong buffer stable, and VC4 supports it.
const PRECISION: [&str; 6] = [
    "#version 100",
    "#ifdef GL_FRAGMENT_PRECISION_HIGH",
    "precision highp float;",
    "#else",
    "precision mediump float;",
    "#endif",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub passes: Vec<Pass>,
    pub version: Value,
    /// The named filters to append after the scene's own passes, in order. The
    /// last one reaches the display. The filters module resolves a name.
    pub filters: Vec<String>,
    /// The scene params. Every pass preamble declares them, and one set reaches
    /// every pass at once. They take the shape of a pass param. They hold the
    /// string param of each text too, which ghee declares.
    pub params: ParamMap,
    /// The GLSL every scene pass shares, such as a function or a constant. It
    /// follows the preamble. None is no common block. A filter pass never sees
    /// it.
    pub common: Option<String>,
    /// Image assets: set up once by the host, available to every pass as a
    /// sampler uniform plus a `<name>_size` vec2.
    pub assets: Vec<Asset>,
    /// Texts the GPU draws from a font. Each draws the scene string param of
    /// its name, which ghee declares for it, and every scene pass reads it with
    /// `<name>_cover`.
    pub texts: Vec<Text>,
}

impl Scene {
    pub fn new(
        passes: Vec<Pass>,
        version: Value,
        filters: Vec<String>,
        params: ParamMap,
        common: Option<String>,
        assets: Vec<Asset>,
    ) -> Result<Scene> {
        if let Some(e) = passes
            .iter()
            .filter(|p| p.filter.is_empty())
            .find_map(Pass::empty)
        {
            bail!("{e}");
        }
        {
            let mut seen: Vec<&str> = Vec::new();
            for p in &passes {
                if seen.contains(&p.name.as_str()) {
                    bail!(
                        "pass '{}' is declared twice; a pass name is the buffer name \
                         channels bind to, so it must be unique",
                        p.name
                    );
                }
                seen.push(&p.name);
            }
        }
        if let Some(e) = passes.iter().find_map(Pass::twin_layers) {
            bail!("{e}");
        }
        // An asset name lands in every pass preamble as a uniform, so it must
        // be a GLSL identifier.
        {
            let mut seen: Vec<&str> = Vec::new();
            for a in &assets {
                if !is_ident(&a.name) {
                    bail!(
                        "asset name {} is not a GLSL identifier",
                        py_str_repr(&a.name)
                    );
                }
                if seen.contains(&a.name.as_str()) {
                    bail!("asset '{}' is declared twice", a.name);
                }
                seen.push(&a.name);
            }
        }
        let scene = Scene {
            passes,
            version,
            filters,
            params,
            common,
            assets,
            texts: Vec::new(),
        };
        if let Some(e) = scene.name_clash() {
            bail!("{e}");
        }
        Ok(scene)
    }

    /// The error for the first declared name that GLSL cannot take: a param
    /// or an array name that is no GLSL identifier, a name that ghee declares
    /// itself, built in or with its prefix, or one GLSL name that two
    /// declarations make in one pass.
    ///
    /// A scene param reaches every pass, an image asset, a text and a layer's
    /// GLSL every pass of the scene's own, a filter's asset the passes of its
    /// filter, and a pass param or an array its own pass. A layer's GLSL takes
    /// its names in every scene pass, whether or not the pass has layers.
    pub(crate) fn name_clash(&self) -> Option<String> {
        // Each declaration as its kind, its name, where it is declared, and
        // whether its name reaches GLSL as it is.
        let mut kinds: Vec<(&str, &str, String, bool)> = Vec::new();
        for p in self.params.values().filter(|p| !p.of_text) {
            kinds.push(("scene param", &p.name, String::new(), p.ty != "string"));
        }
        for a in &self.assets {
            match a.from_filter.as_str() {
                "" => kinds.push(("asset", &a.name, String::new(), false)),
                f => kinds.push(("asset", a.local_name(), format!(" of filter '{f}'"), false)),
            }
        }
        for t in &self.texts {
            kinds.push(("text", &t.name, String::new(), false));
        }
        for pass in &self.passes {
            let at = format!(" in pass '{}'", pass.name);
            for p in pass.params.values() {
                kinds.push(("param", &p.name, at.clone(), p.ty != "string"));
            }
            for a in &pass.arrays {
                kinds.push(("array", &a.name, at.clone(), true));
            }
        }
        for (kind, name, at, glsl) in &kinds {
            if *glsl && !is_ident(name) {
                return Some(format!(
                    "{kind} name {}{at} is not a GLSL identifier",
                    py_str_repr(name)
                ));
            }
            if built_in(name) {
                return Some(format!("{kind} '{name}'{at}: '{name}' is built in"));
            }
            if name.starts_with(PREFIX) {
                return Some(format!(
                    "{kind} '{name}'{at}: a name that starts with {PREFIX} belongs to ghee"
                ));
            }
        }
        let layers = "the layers".to_string();
        for pass in &self.passes {
            let own = pass.filter.is_empty();
            let mut names: Vec<(String, String)> = Vec::new();
            for p in self.params.values().filter(|p| p.ty != "string") {
                names.push((p.name.clone(), format!("scene param '{}'", p.name)));
            }
            for a in &self.assets {
                if a.from_filter != pass.filter || a.is_font() {
                    continue;
                }
                let owner = match a.from_filter.as_str() {
                    "" => format!("asset '{}'", a.name),
                    f => format!("asset '{}' of filter '{f}'", a.local_name()),
                };
                for n in declared_names(&asset_lines(a)) {
                    names.push((n, owner.clone()));
                }
            }
            if own {
                for t in &self.texts {
                    for n in crate::text::names(&t.name) {
                        names.push((n, format!("text '{}'", t.name)));
                    }
                }
                for n in crate::text::LAYER_NAMES {
                    names.push((n.to_string(), layers.clone()));
                }
            }
            for p in pass.params.values().filter(|p| p.ty != "string") {
                let owner = format!("param '{}' in pass '{}'", p.name, pass.name);
                names.push((p.name.clone(), owner));
            }
            for a in &pass.arrays {
                let owner = format!("array '{}' in pass '{}'", a.name, pass.name);
                let mut lines = array_lines(a);
                lines.extend(array_reader_lines(a, false));
                for n in declared_names(&lines) {
                    names.push((n, owner.clone()));
                }
            }
            for (i, (n, owner)) in names.iter().enumerate() {
                let Some((_, first)) = names[..i].iter().find(|(m, _)| m == n) else {
                    continue;
                };
                if first == owner {
                    return Some(format!("{owner} is declared twice"));
                }
                return Some(format!("{first} and {owner} both declare '{n}'"));
            }
        }
        None
    }

    /// The scene with `texts` in place of its own.
    pub fn with_texts(mut self, mut texts: Vec<Text>) -> Result<Scene> {
        let mut seen: Vec<&str> = Vec::new();
        for t in &texts {
            if !is_ident(&t.name) {
                bail!(
                    "text name {} is not a GLSL identifier",
                    py_str_repr(&t.name)
                );
            }
            if seen.contains(&t.name.as_str()) {
                bail!("text '{}' is declared twice", t.name);
            }
            seen.push(&t.name);
            for font in t.fonts.iter().filter(|f| !crate::text::is_font_file(f)) {
                match self.assets.iter().find(|a| a.name == *font) {
                    Some(a) if a.is_font() => {}
                    Some(a) => bail!(
                        "text '{}' font '{font}' is the image {}, a font is a .ttf or .otf file",
                        t.name,
                        py_str_repr(&a.file)
                    ),
                    None => bail!(
                        "text '{}' font '{font}' is no asset of the scene and no .ttf or .otf file",
                        t.name
                    ),
                }
            }
        }
        // ghee declares the string param of each text, which holds its words,
        // after the scene's own params, and refuses a param declared under the
        // text's name. A param that ghee declared before keeps its words.
        let held: Vec<Param> = self
            .params
            .values()
            .filter(|p| p.of_text)
            .cloned()
            .collect();
        self.params.retain(|p| !p.of_text);
        for t in &mut texts {
            let n = t.name.clone();
            if self.params.contains_key(&n) {
                bail!(
                    "scene param '{n}' has the name of text '{n}', which holds its words in \
                     its 'value'"
                );
            }
            let words = match held.iter().find(|p| p.name == n) {
                Some(Param {
                    value: ParamValue::Str(w),
                    ..
                }) => w.clone(),
                _ => t.value.clone(),
            };
            self.params.insert(n.clone(), Param::words(&n, &words));
            t.value = words;
            if let Some(p) = self.passes.iter().find(|p| p.params.contains_key(&n)) {
                bail!(
                    "param '{n}' in pass '{}' has the name of text '{n}', which holds its \
                     words in its 'value'",
                    p.name
                );
            }
        }
        self.texts = texts;
        if let Some(e) = self.undeclared_layer() {
            bail!("{e}");
        }
        for t in self.texts.iter().filter(|t| t.fill == Fill::Mtsdf) {
            let (pad, layer) = self.largest_pad(&t.name);
            let range = crate::text::atlas_range(t.px_per_em, pad);
            if let (Some(layer), true) = (layer, range > crate::text::ATLAS_MAX) {
                bail!(
                    "{layer} pad {} needs an atlas range of {range} texels at the {} px_per_em \
                     of text '{}', over the cap of {}",
                    py_float(pad),
                    t.px_per_em,
                    t.name,
                    crate::text::ATLAS_MAX
                );
            }
        }
        if let Some(e) = self.name_clash() {
            bail!("{e}");
        }
        Ok(self)
    }

    /// The distance range in texels of the MTSDF atlas of text `name`: the
    /// field reaches the largest `pad` of the text's layers, at the text's
    /// px_per_em.
    pub fn atlas_range(&self, name: &str) -> u32 {
        let px_per_em = self
            .texts
            .iter()
            .find(|t| t.name == name)
            .map_or(crate::atlas::PX_EM, |t| t.px_per_em);
        let range = crate::text::atlas_range(px_per_em, self.largest_pad(name).0);
        u32::try_from(range).unwrap_or(u32::MAX)
    }

    /// The largest `pad` of the layers of text `name`, and the layer that sets
    /// it as "pass 'main' layer 'glow'", None when no layer pads.
    fn largest_pad(&self, name: &str) -> (f64, Option<String>) {
        let mut most = (0.0, None);
        for p in &self.passes {
            for (k, d) in p.layers.iter().enumerate() {
                if d.text == name && d.pad > most.0 {
                    most = (d.pad, Some(p.layer_what(k)));
                }
            }
        }
        most
    }

    /// True when a layer of the scene names text `name`.
    pub fn has_layer_of(&self, name: &str) -> bool {
        self.passes
            .iter()
            .any(|p| p.layers.iter().any(|d| d.text == name))
    }

    /// True when a layer of the scene paints the colour glyphs of text `name`.
    pub fn paints_emoji(&self, name: &str) -> bool {
        self.passes
            .iter()
            .any(|p| (0..p.layers.len()).any(|k| p.layers[k].text == name && p.paints_emoji(k)))
    }

    /// The error for the first layer whose text the scene does not declare.
    pub(crate) fn undeclared_layer(&self) -> Option<String> {
        self.passes.iter().find_map(|p| {
            let (k, d) = p
                .layers
                .iter()
                .enumerate()
                .find(|(_, d)| !self.texts.iter().any(|t| t.name == d.text))?;
            Some(format!(
                "{} names text '{}', which the scene does not declare",
                p.layer_what(k),
                d.text
            ))
        })
    }

    /// A scene with passes only. Every other field takes its default.
    pub fn of(passes: Vec<Pass>) -> Result<Scene> {
        Scene::new(
            passes,
            Value::from(1),
            Vec::new(),
            ParamMap::new(),
            None,
            Vec::new(),
        )
    }

    pub fn output(&self) -> &Pass {
        &self.passes[self.passes.len() - 1]
    }

    /// `path` names the scene in an error message. A `shader` file resolves
    /// in `base_dir`, with the reach of a scene that its user runs.
    pub fn from_dict(d: &Value, base_dir: Option<&str>, path: Option<&str>) -> Result<Scene> {
        Scene::parse(d, any_file(base_dir), path)
    }

    /// from_dict() with a scene's files in `dir`, as far as `reach` allows.
    pub fn from_dict_in(d: &Value, dir: &str, reach: Reach, path: Option<&str>) -> Result<Scene> {
        Scene::parse(d, Some(Files { dir, reach }), path)
    }

    fn parse(d: &Value, files: Option<Files>, path: Option<&str>) -> Result<Scene> {
        let src = path.unwrap_or("scene");
        let Value::Object(o) = d else {
            bail!("{src}: needs a JSON object, got {}", py_repr(d));
        };
        (|| -> Result<Scene> {
            let elsewhere = [
                keys_on(&SCENE_KEYS, &PASS_KEYS, "a pass"),
                keys_on(&SCENE_KEYS, &TEXT_KEYS, "a text"),
                keys_on(&SCENE_KEYS, &LAYER_KEYS, "a layer"),
                keys_on(&SCENE_KEYS, &ASSET_KEYS, "an asset"),
            ]
            .concat();
            known_keys(o, "the scene", &SCENE_KEYS, &elsewhere)?;
            let filters = list_of(o, "filters")?;
            if !filters.iter().all(Value::is_string) {
                bail!(
                    "'filters' needs a list of names, got {}",
                    py_repr(&Value::Array(filters.clone()))
                );
            }
            let version = o.get("version").cloned().unwrap_or_else(|| Value::from(1));
            let passes = list_of(o, "passes")?
                .iter()
                .map(|pd| Pass::parse(pd, files))
                .collect::<Result<Vec<_>>>()?;
            let mut params = ParamMap::new();
            for (n, pd) in obj_of(o, "params")? {
                params.insert(n.clone(), Param::from_dict(n, pd)?);
            }
            let mut assets = Vec::new();
            for (n, ad) in obj_of(o, "assets")? {
                assets.push(Asset::from_dict(n, ad)?);
            }
            let mut texts = Vec::new();
            for (n, td) in obj_of(o, "texts")? {
                texts.push(Text::from_dict(n, td)?);
            }
            let common = common_block(o, files)?;
            Scene::new(
                passes,
                version,
                filters.iter().map(py_str).collect(),
                params,
                common,
                assets,
            )?
            .with_texts(texts)
        })()
        .map_err(|e| anyhow!("{src}: {e}"))
    }

    pub fn to_dict(&self) -> Value {
        let mut out = Map::new();
        out.insert("version".to_string(), self.version.clone());
        out.insert(
            "passes".to_string(),
            Value::from(self.passes.iter().map(Pass::to_dict).collect::<Vec<_>>()),
        );
        // "" is a common block the console can fill.
        if let Some(c) = &self.common {
            let mut m = Map::new();
            m.insert("code".to_string(), Value::from(c.clone()));
            out.insert("common".to_string(), Value::Object(m));
        }
        if !self.filters.is_empty() {
            out.insert("filters".to_string(), Value::from(self.filters.clone()));
        }
        let own: Vec<&Param> = self.params.values().filter(|p| !p.of_text).collect();
        if !own.is_empty() {
            let mut m = Map::new();
            for p in own {
                m.insert(p.name.clone(), p.to_dict());
            }
            out.insert("params".to_string(), Value::Object(m));
        }
        // Only the file references. The console posts them back as they are,
        // and the server resolves them against its store. A filter's asset
        // comes back with its filter.
        let own: Vec<&Asset> = self
            .assets
            .iter()
            .filter(|a| a.from_filter.is_empty())
            .collect();
        if !own.is_empty() {
            let mut m = Map::new();
            for a in own {
                m.insert(a.name.clone(), a.to_dict());
            }
            out.insert("assets".to_string(), Value::Object(m));
        }
        if !self.texts.is_empty() {
            let mut m = Map::new();
            for t in &self.texts {
                let mut d = t.to_dict();
                // The words as they stand, which an edit of the param changes.
                if let Some(ParamValue::Str(w)) = self.params.get(&t.name).map(|p| &p.value) {
                    d["value"] = Value::from(w.clone());
                }
                m.insert(t.name.clone(), d);
            }
            out.insert("texts".to_string(), Value::Object(m));
        }
        Value::Object(out)
    }

    /// Load a scene file. Its files resolve beside it, with the reach of a
    /// scene that its user runs.
    pub fn load(path: &str) -> Result<Scene> {
        Scene::load_in(path, None, Reach::Any)
    }

    /// Load a scene file whose files resolve in `dir`, else beside the file,
    /// as far as `reach` allows.
    pub fn load_in(path: &str, dir: Option<&str>, reach: Reach) -> Result<Scene> {
        let text = fs::read_to_string(path).map_err(|e| {
            anyhow!(
                "[Errno {}] {}: '{path}'",
                e.raw_os_error().unwrap_or(0),
                strerror(&e)
            )
        })?;
        let data: Value =
            serde_json::from_str(&text).map_err(|e| anyhow!("{path}: invalid JSON: {e}"))?;
        crate::sources::record(path);
        let abs = abspath(path);
        let base = match (dir, abs.rfind('/')) {
            (Some(d), _) => d.to_string(),
            (None, Some(0)) => "/".to_string(),
            (None, Some(i)) => abs[..i].to_string(),
            (None, None) => String::new(),
        };
        let mut scene = Scene::from_dict_in(&data, &base, reach, Some(path))?;
        scene
            .load_assets_in(&base, reach)
            .map_err(|e| anyhow!("{path}: {e}"))?;
        Ok(scene)
    }

    /// Read and decode every declared asset file, and hand each text its font,
    /// with the reach of a scene that its user runs, as load_assets_in() says.
    pub fn load_assets(&mut self, base_dir: &str) -> Result<()> {
        self.load_assets_in(base_dir, Reach::Any)
    }

    /// Read and decode every declared asset file, and hand each text its font.
    ///
    /// An absolute path reads as it stands. Any other path resolves in
    /// `base_dir`, then through `$GHEE_ASSETS`, colon-separated directories, in
    /// order, and the error names the first path and the directories of
    /// `$GHEE_ASSETS`. Reach::Inside refuses a path outside `base_dir`. A
    /// filter's asset is the filter's to load.
    pub fn load_assets_in(&mut self, base_dir: &str, reach: Reach) -> Result<()> {
        let font_files = self
            .texts
            .iter()
            .flat_map(|t| &t.fonts)
            .any(|f| crate::text::is_font_file(f));
        if self.assets.is_empty() && !font_files {
            return Ok(());
        }
        let root = normpath(&abspath(base_dir));
        let more = asset_dirs();
        let outside =
            |file: &str| reach == Reach::Inside && !inside(&normpath(&join(&root, file)), &root);
        // The bytes of `file` and the path that held them, or why no path did,
        // with the file named as `what`.
        let read = |file: &str, what: &str| {
            let roots = match file.starts_with('/') {
                true => &[][..],
                false => &more[..],
            };
            let first = normpath(&join(&root, file));
            let found = std::iter::once(first.clone())
                .chain(roots.iter().map(|r| normpath(&join(r, file))))
                .find(|p| std::path::Path::new(p).is_file());
            match found {
                None => {
                    // A font file that shows up later marks the scene stale.
                    crate::sources::record(&first);
                    let elsewhere = match (file.starts_with('/'), more.is_empty()) {
                        (true, _) => String::new(),
                        (false, true) => ", and $GHEE_ASSETS names no directory".to_string(),
                        (false, false) => format!(", nor in $GHEE_ASSETS ({})", more.join(", ")),
                    };
                    Err(format!(
                        "{what} ({first}): No such file or directory{elsewhere}"
                    ))
                }
                Some(path) => {
                    crate::sources::record(&path);
                    match fs::read(&path) {
                        Ok(bytes) => Ok((bytes, path)),
                        Err(e) => Err(format!("{what} ({path}): {}", strerror(&e))),
                    }
                }
            }
        };
        for a in &mut self.assets {
            if !a.from_filter.is_empty() {
                continue; // the filter decoded it
            }
            if outside(&a.file) {
                bail!(
                    "asset '{}' file {} is outside the scene directory",
                    a.name,
                    py_str_repr(&a.file)
                );
            }
            match read(&a.file, &format!("asset '{}'", a.name)) {
                Ok((bytes, path)) => a.decode(&bytes, &path)?,
                // A font that fails to load leaves the scene running, and its
                // texts draw error boxes.
                Err(why) if a.is_font() => a.font = Some(Err(why)),
                Err(why) => bail!("{why}"),
            }
        }
        for t in &mut self.texts {
            let mut data = Vec::new();
            for f in &t.fonts {
                if !crate::text::is_font_file(f) {
                    data.push(
                        self.assets
                            .iter()
                            .find(|a| a.name == *f)
                            .and_then(|a| a.font.clone()),
                    );
                    continue;
                }
                if outside(f) {
                    bail!(
                        "text '{}' font {} is outside the scene directory",
                        t.name,
                        py_str_repr(f)
                    );
                }
                let what = format!("text '{}' font {}", t.name, py_str_repr(f));
                data.push(Some(read(f, &what).and_then(|(bytes, path)| {
                    crate::text::font(&bytes)
                        .map_err(|e| format!("{what} ({path}): unreadable font: {e}"))
                })));
            }
            t.data = data.into_iter().collect();
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_dict())?)?;
        Ok(())
    }
}

/// The directories `$GHEE_ASSETS` names, colon-separated, absolute, in order.
fn asset_dirs() -> Vec<String> {
    std::env::var("GHEE_ASSETS")
        .unwrap_or_default()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(|d| normpath(&abspath(d)))
        .collect()
}

/// Chained construction on a Pass constructor's Result:
/// `Pass::of(name, code).array("level", 8).param(Param::float("gain", 1.0))?`.
/// The first error stops the chain, and `?` returns it.
pub trait PassBuilder {
    /// Declare an array for the host to fill every frame. `8` is a 1x8 linear
    /// array, `(16, 8)` a 16x8 grid.
    fn array<S: Into<ArraySize>>(self, name: &str, size: S) -> Result<Pass>;

    /// Declare a pass param. A second param of the same name replaces the first.
    fn param(self, param: Param) -> Result<Pass>;

    /// Bind the next iChannel to a pass's buffer. `None` leaves it unbound. An
    /// unbound channel reads black.
    fn channel<'a, B: Into<Option<&'a str>>>(self, buffer: B) -> Result<Pass>;

    /// Render this pass at [width, height] instead of the renderer's size. The
    /// output pass has no size of its own.
    fn size(self, width: i64, height: i64) -> Result<Pass>;
}

impl PassBuilder for Result<Pass> {
    fn array<S: Into<ArraySize>>(self, name: &str, size: S) -> Result<Pass> {
        let mut p = self?;
        p.arrays
            .push(ArrayInput::new(name, size_value(size.into()))?);
        Ok(p)
    }

    fn param(self, param: Param) -> Result<Pass> {
        let mut p = self?;
        p.params.insert(param.name.clone(), param);
        Ok(p)
    }

    fn channel<'a, B: Into<Option<&'a str>>>(self, buffer: B) -> Result<Pass> {
        let mut p = self?;
        p.channels.push(Channel::new(buffer.into()));
        Ok(p)
    }

    fn size(self, width: i64, height: i64) -> Result<Pass> {
        let p = self?;
        let mut sized = Pass::new(
            &p.name,
            &p.code,
            p.arrays,
            p.channels,
            p.params,
            &p.filter,
            Some(Value::from(vec![width, height])),
        )?;
        sized.mipmaps = p.mipmaps;
        sized.layers = p.layers;
        Ok(sized)
    }
}

/// Chained construction on a Scene constructor's Result:
/// `Scene::of(passes).param(Param::color("tint", [1.0, 0.0, 0.0]))?`.
pub trait SceneBuilder {
    /// Declare a scene param. It is one uniform in every pass. A pass that
    /// declares the same name fails here.
    fn param(self, param: Param) -> Result<Scene>;

    /// Append named filters, in order. The last one reaches the display.
    fn filters(self, names: &[&str]) -> Result<Scene>;
}

impl SceneBuilder for Result<Scene> {
    fn param(self, param: Param) -> Result<Scene> {
        let mut s = self?;
        s.params.insert(param.name.clone(), param);
        Scene::new(s.passes, s.version, s.filters, s.params, s.common, s.assets)?
            .with_texts(s.texts)
    }

    fn filters(self, names: &[&str]) -> Result<Scene> {
        let mut s = self?;
        s.filters.extend(names.iter().map(|n| (*n).to_string()));
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GOOD: &str = "void main(){ gl_FragColor = vec4(1.0); }";
    const COMMON: &str = "float sat(float x){ return clamp(x, 0.0, 1.0); }";
    const CODE: &str = "void main(){ gl_FragColor = vec4(sat(v_texCoord.x), 0.0, 0.0, 1.0); }";
    const FLAT: &str = "void main(){ gl_FragColor = vec4(vec3(0.5), 1.0); }";

    fn param(name: &str, ty: &str, value: Value) -> Result<Param> {
        Param::new(name, ty, value, None, None, None, None)
    }

    fn tmpdir(tag: &str) -> String {
        let d = std::env::temp_dir().join(format!("ghee-scene-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    fn repo(rel: &str) -> String {
        format!("{}/{rel}", env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn a_sized_pass_keeps_its_layers() {
        let mut p = Pass::of("a", GOOD).unwrap();
        p.layers = vec![Layer::from_dict(
            "pass 'a' layer 't'",
            &json!({"text": "t", "fragment": GOOD, "vertex": "vec2 glyph_place(vec2 em) { return em; }"}),
        )
        .unwrap()];
        let sized = Ok::<Pass, anyhow::Error>(p.clone()).size(8, 4).unwrap();
        assert_eq!(sized.size, Some((8, 4)));
        assert_eq!(sized.layers, p.layers);
    }

    #[test]
    fn preamble_has_resolution_no_orientation() {
        let g = Pass::of("main", "void main(){}")
            .unwrap()
            .glsl(None, None, &[], &[], false);
        assert!(g.contains("uniform vec2 resolution;"));
        assert!(g.contains("uniform float time;"));
        assert!(!g.contains("VERTICAL") && !g.contains("HORIZONTAL"));
    }

    #[test]
    fn preamble_declares_array_and_accessor() {
        let p = Pass::from_dict(
            &json!({"name": "m", "code": "x", "arrays": [{"name": "amps", "size": 8}]}),
            None,
        )
        .unwrap();
        let g = p.glsl(None, None, &[], &[], false);
        assert!(g.contains("uniform sampler2D amps_tex;")); // An array is a 1xN data texture.
        assert!(g.contains("const int amps_size = 8;"));
        assert!(g.contains("amps_at"));
        assert!(g.contains("amps_sample"));
    }

    #[test]
    fn scene_roundtrip() {
        let d = json!({
            "version": 1,
            "passes": [{
                "name": "m",
                "code": "void main(){}",
                "arrays": [{"name": "a", "size": 4}],
                "params": {"k": {"type": "float", "value": 0.5}},
            }],
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn pass_size_roundtrip() {
        let d = json!({
            "version": 1,
            "passes": [
                {"name": "pre", "code": "void main(){}", "size": [32, 18]},
                {"name": "main", "code": "void main(){}"},
            ],
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn pass_mipmaps_roundtrip() {
        let d = json!({
            "version": 1,
            "passes": [
                {"name": "pre", "code": "void main(){}", "size": [64, 32], "mipmaps": true},
                {"name": "main", "code": "void main(){}"},
            ],
        });
        let sc = Scene::from_dict(&d, None, None).unwrap();
        assert!(sc.passes[0].mipmaps && !sc.passes[1].mipmaps);
        assert_eq!(sc.to_dict(), d);
        let sized = Ok::<Pass, anyhow::Error>(sc.passes[0].clone())
            .size(8, 4)
            .unwrap();
        assert!(sized.mipmaps);
        // false is the default, and a save leaves it out.
        let off = json!({"version": 1, "passes": [{"name": "m", "code": "x", "mipmaps": false}]});
        assert_eq!(
            Scene::from_dict(&off, None, None).unwrap().to_dict(),
            json!({"version": 1, "passes": [{"name": "m", "code": "x"}]})
        );
        let bad = json!({"version": 1, "passes": [{"name": "m", "code": "x", "mipmaps": "yes"}]});
        assert_eq!(
            Scene::from_dict(&bad, None, None).unwrap_err().to_string(),
            "scene: pass 'm' mipmaps needs true or false, got 'yes'"
        );
    }

    #[test]
    fn pass_size_validates() {
        let sized =
            |size: Value| Pass::new("ok", "x", vec![], vec![], ParamMap::new(), "", Some(size));
        assert_eq!(sized(json!([8, 4])).unwrap().size, Some((8, 4)));
        for bad in [
            json!([0, 4]),
            json!([4, 0]),
            json!([-1, 4]),
            json!([8]),
            json!([8, 4, 2]),
            json!("8x4"),
            json!([8.5, 4]),
            json!([8, true]),
        ] {
            assert!(sized(bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn color_param_roundtrip() {
        let d = json!({
            "version": 1,
            "passes": [{
                "name": "m",
                "code": "void main(){}",
                "params": {"tint": {"type": "color", "value": [1.0, 0.0, 0.5]}},
            }],
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn color_param_validates_length_and_type() {
        param("ok", "color", json!([0.1, 0.2, 0.3])).unwrap();
        for bad in [
            json!([0.1, 0.2]),
            json!([0.1, 0.2, 0.3, 0.4]),
            json!("red"),
            json!(0.5),
            json!([0.1, 0.2, "x"]),
        ] {
            assert!(param("bad", "color", bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn color_param_is_a_vec3_uniform() {
        let mut params = ParamMap::new();
        params.insert(
            "tint".to_string(),
            param("tint", "color", json!([1.0, 0.0, 0.0])).unwrap(),
        );
        let g = Pass::new(
            "m",
            "void main(){ gl_FragColor = vec4(tint, 1.0); }",
            vec![],
            vec![],
            params,
            "",
            None,
        )
        .unwrap()
        .glsl(None, None, &[], &[], false);
        assert!(g.contains("uniform vec3 tint;"));
        assert!(!g.contains("tint_r") && !g.contains("#define tint"));
        let p = param("tint", "color", json!([1.0, 0.0, 0.0])).unwrap();
        assert_eq!(p.glsl_type().unwrap(), "vec3");
    }

    #[test]
    fn vec_param_roundtrip() {
        let d = json!({
            "version": 1,
            "passes": [{
                "name": "m",
                "code": "void main(){}",
                "params": {
                    "a": {"type": "vec2", "value": [0.1, 0.2]},
                    "b": {"type": "vec3", "value": [0.1, 0.2, 0.3]},
                    "c": {"type": "vec4", "value": [0.1, 0.2, 0.3, 0.4]},
                },
            }],
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn vec_param_validates_length_and_type() {
        param("ok2", "vec2", json!([0.1, 0.2])).unwrap();
        param("ok3", "vec3", json!([0.1, 0.2, 0.3])).unwrap();
        param("ok4", "vec4", json!([0.1, 0.2, 0.3, 0.4])).unwrap();
        let bad_by_type = [
            (
                "vec2",
                vec![
                    json!([0.1]),
                    json!([0.1, 0.2, 0.3]),
                    json!("xy"),
                    json!(0.5),
                    json!([0.1, "x"]),
                ],
            ),
            (
                "vec3",
                vec![
                    json!([0.1, 0.2]),
                    json!([0.1, 0.2, 0.3, 0.4]),
                    json!("rgb"),
                    json!(1),
                    json!([0.1, 0.2, "x"]),
                ],
            ),
            (
                "vec4",
                vec![
                    json!([0.1, 0.2, 0.3]),
                    json!([0.1, 0.1, 0.1, 0.1, 0.1]),
                    json!("rgba"),
                    json!(2),
                    json!([0.1, 0.2, 0.3, "x"]),
                ],
            ),
        ];
        for (t, bads) in bad_by_type {
            for bad in bads {
                assert!(param("bad", t, bad.clone()).is_err(), "{t} {bad}");
            }
        }
    }

    #[test]
    fn vec_param_is_a_vector_uniform() {
        for (t, name, n) in [("vec2", "off", 2), ("vec3", "dir", 3), ("vec4", "q", 4)] {
            let mut params = ParamMap::new();
            params.insert(
                name.to_string(),
                param(name, t, json!(vec![0.0; n])).unwrap(),
            );
            let g = Pass::new("m", "void main(){}", vec![], vec![], params, "", None)
                .unwrap()
                .glsl(None, None, &[], &[], false);
            assert!(g.contains(&format!("uniform {t} {name};")), "{g}");
            assert!(!g.contains(&format!("{name}_x")), "{g}");
            assert!(!g.contains(&format!("#define {name}")), "{g}");
            let p = param(name, t, json!(vec![0.0; n])).unwrap();
            assert_eq!(p.glsl_type().unwrap(), t);
        }
    }

    #[test]
    fn scene_params_roundtrip() {
        let d = json!({
            "version": 1,
            "passes": [{
                "name": "m",
                "code": "void main(){}",
                "params": {"k": {"type": "float", "value": 0.5}},
            }],
            "params": {
                "speed": {"type": "float", "value": 0.06, "min": -0.5, "max": 0.5, "step": 0.005},
                "tint": {"type": "color", "value": [1.0, 0.0, 0.5]},
                "off": {"type": "vec2", "value": [0.1, 0.2]},
            },
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn scene_pass_param_name_collision_raises() {
        let mut pass_params = ParamMap::new();
        pass_params.insert(
            "speed".to_string(),
            param("speed", "float", json!(0.1)).unwrap(),
        );
        let mut scene_params = ParamMap::new();
        scene_params.insert(
            "speed".to_string(),
            param("speed", "float", json!(0.2)).unwrap(),
        );
        let p = Pass::new(
            "main",
            "void main(){}",
            vec![],
            vec![],
            pass_params,
            "",
            None,
        )
        .unwrap();
        let e = Scene::new(vec![p], json!(1), vec![], scene_params, None, vec![])
            .unwrap_err()
            .to_string();
        assert!(e.contains("speed") && e.contains("main"), "{e}");
    }

    #[test]
    fn scene_pass_param_collision_via_from_dict_raises() {
        let d = json!({
            "version": 1,
            "passes": [{"name": "p0", "code": "x", "params": {"g": {"type": "float", "value": 0.1}}}],
            "params": {"g": {"type": "float", "value": 0.2}},
        });
        assert!(Scene::from_dict(&d, None, None).is_err());
    }

    #[test]
    fn scene_params_in_every_pass_preamble() {
        let mut params = ParamMap::new();
        params.insert(
            "gain".to_string(),
            param("gain", "float", json!(0.5)).unwrap(),
        );
        params.insert(
            "tint".to_string(),
            param("tint", "color", json!([1.0, 0.0, 0.0])).unwrap(),
        );
        params.insert(
            "off".to_string(),
            param("off", "vec2", json!([0.1, 0.2])).unwrap(),
        );
        let sc = Scene::new(
            vec![
                Pass::of("a", "void main(){}").unwrap(),
                Pass::new(
                    "b",
                    "void main(){}",
                    vec![],
                    vec![Channel::new(Some("a"))],
                    ParamMap::new(),
                    "",
                    None,
                )
                .unwrap(),
            ],
            json!(1),
            vec![],
            params,
            None,
            vec![],
        )
        .unwrap();
        for p in &sc.passes {
            let g = p.glsl(Some(&sc.params), None, &sc.assets, &[], false);
            assert!(g.contains("uniform float gain;"));
            assert!(g.contains("uniform vec3 tint;"));
            assert!(g.contains("uniform vec2 off;"));
        }
    }

    #[test]
    fn no_scene_params_emits_nothing_extra() {
        let mut params = ParamMap::new();
        params.insert("k".to_string(), param("k", "float", json!(0.3)).unwrap());
        let sc = Scene::of(vec![Pass::new(
            "m",
            "void main(){}",
            vec![],
            vec![],
            params,
            "",
            None,
        )
        .unwrap()])
        .unwrap();
        // An empty scene param map adds nothing to the preamble.
        for p in &sc.passes {
            assert_eq!(
                p.glsl(Some(&sc.params), None, &sc.assets, &[], false),
                p.glsl(None, None, &[], &[], false)
            );
        }
    }

    #[test]
    fn string_param_roundtrip_with_options() {
        let d = json!({
            "version": 1,
            "passes": [{"name": "m", "code": "void main(){}"}],
            "params": {
                "text": {"type": "string", "value": "GHEE"},
                "font": {"type": "string", "value": "Verdana", "options": ["Verdana", "Impact"]},
            },
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn string_param_validates() {
        param("ok", "string", json!("hello")).unwrap();
        Param::new(
            "ok",
            "string",
            json!("a"),
            None,
            None,
            None,
            Some(json!(["a", "b"])),
        )
        .unwrap();
        for bad in [json!(1.0), json!(["x"]), json!(null), json!(true)] {
            assert!(param("bad", "string", bad.clone()).is_err(), "{bad}");
        }
        assert!(Param::new(
            "bad",
            "string",
            json!("a"),
            None,
            None,
            None,
            Some(json!([1, 2]))
        )
        .is_err());
        // Only a string param takes options.
        assert!(Param::new(
            "bad",
            "float",
            json!(0.5),
            None,
            None,
            None,
            Some(json!(["a"]))
        )
        .is_err());
    }

    #[test]
    fn string_param_emits_no_uniform() {
        let mut params = ParamMap::new();
        params.insert(
            "text".to_string(),
            param("text", "string", json!("GHEE")).unwrap(),
        );
        let g = Pass::new(
            "m",
            "void main(){ gl_FragColor = vec4(1.0); }",
            vec![],
            vec![],
            params,
            "",
            None,
        )
        .unwrap()
        .glsl(None, None, &[], &[], false);
        assert!(!g.split("void main").next().unwrap().contains("text"));

        let mut scene_params = ParamMap::new();
        scene_params.insert(
            "text".to_string(),
            param("text", "string", json!("GHEE")).unwrap(),
        );
        let s = Scene::new(
            vec![Pass::of("m", "void main(){}").unwrap()],
            json!(1),
            vec![],
            scene_params,
            None,
            vec![],
        )
        .unwrap();
        let g2 = s.passes[0].glsl(Some(&s.params), None, &s.assets, &[], false);
        let head = g2.split("void main").next().unwrap().to_string();
        let tail = head
            .split("uniform float frame;")
            .nth(1)
            .unwrap()
            .to_string();
        assert!(!tail.contains("uniform"));
    }

    #[test]
    fn string_param_glsl_type_raises() {
        let p = param("text", "string", json!("GHEE")).unwrap();
        assert_eq!(
            p.glsl_type().unwrap_err().to_string(),
            "a string param is host-side, so it has no glsl_type"
        );
    }

    // A scene's file reference never escapes the scene directory.

    #[test]
    fn shader_key_without_base_dir_raises() {
        let d = json!({"version": 1, "passes": [{"name": "main", "shader": "/etc/hostname"}]});
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert_eq!(
            e,
            "scene: pass 'main' names a shader file ('/etc/hostname') but the scene \
             has no directory to resolve it against; inline the GLSL in 'code'"
        );
    }

    #[test]
    fn shader_outside_base_dir_raises() {
        let root = tmpdir("outside");
        let base = format!("{root}/scene");
        fs::create_dir_all(&base).unwrap();
        fs::write(format!("{base}/in.glsl"), GOOD).unwrap();
        let secret = format!("{root}/secret.glsl");
        fs::write(&secret, "stolen").unwrap();
        for path in ["../secret.glsl", &secret, "sub/../../secret.glsl"] {
            let d = json!({"version": 1, "passes": [{"name": "main", "shader": path}]});
            let e = Scene::from_dict_in(&d, &base, Reach::Inside, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains("outside the scene directory"), "{path}: {e}");
        }
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn shader_inside_base_dir_still_loads() {
        let base = tmpdir("inside");
        fs::create_dir_all(format!("{base}/sub")).unwrap();
        fs::write(format!("{base}/sub/in.glsl"), GOOD).unwrap();
        let d = json!({"version": 1, "passes": [{"name": "main", "shader": "sub/in.glsl"}]});
        let sc = Scene::from_dict(&d, Some(&base), None).unwrap();
        assert_eq!(sc.passes[0].code, GOOD);
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn serialized_scene_reloads_without_a_directory() {
        let scene = Scene::load(&repo("ghee/examples/bars.scene.json")).unwrap();
        let d = scene.to_dict();
        // The console posts code back inline.
        assert!(!d.to_string().contains("shader"));
        assert_eq!(
            Scene::from_dict(&d, None, None).unwrap().passes[0].code,
            scene.passes[0].code
        );
    }

    #[test]
    fn missing_shader_file_names_scene_and_pass() {
        let dir = tmpdir("missing-shader");
        let p = format!("{dir}/x.scene.json");
        fs::write(
            &p,
            json!({"version": 1, "passes": [{"name": "main", "shader": "gone.glsl"}]}).to_string(),
        )
        .unwrap();
        let msg = Scene::load(&p).unwrap_err().to_string();
        assert!(
            msg.contains(&p) && msg.contains("main") && msg.contains("gone.glsl"),
            "{msg}"
        );
        assert!(msg.contains("No such file or directory"), "{msg}");
        fs::remove_dir_all(&dir).unwrap();
    }

    // A param value is typed.

    #[test]
    fn float_and_int_params_need_numbers() {
        for t in ["float", "int"] {
            for bad in [
                json!("<img src=x onerror=alert(1)>"),
                json!(true),
                json!(null),
                json!([1, 2]),
            ] {
                let e = param("g", t, bad.clone()).unwrap_err().to_string();
                assert!(e.contains("needs a number"), "{t} {bad}: {e}");
            }
        }
        assert_eq!(
            param("g", "float", json!(1)).unwrap().value,
            ParamValue::Float(1.0)
        );
        assert_eq!(
            param("g", "int", json!(3)).unwrap().value,
            ParamValue::Int(3)
        );
    }

    #[test]
    fn param_range_metadata_needs_numbers() {
        Param::new(
            "g",
            "float",
            json!(0.5),
            Some(json!(0)),
            Some(json!(1)),
            Some(json!(0.1)),
            None,
        )
        .unwrap();
        param("g", "float", json!(0.5)).unwrap(); // No min, max or step.
        for k in ["min", "max", "step"] {
            let (mut lo, mut hi, mut st) = (None, None, None);
            match k {
                "min" => lo = Some(json!("lo")),
                "max" => hi = Some(json!("lo")),
                _ => st = Some(json!("lo")),
            }
            let e = Param::new("g", "float", json!(0.5), lo, hi, st, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains(k), "{k}: {e}");
        }
    }

    #[test]
    fn unknown_param_type_raises_at_parse_time() {
        assert!(param("g", "flaot", json!(0.5))
            .unwrap_err()
            .to_string()
            .contains("unknown type"));
        let d = json!({
            "version": 1,
            "passes": [{"name": "m", "code": GOOD, "params": {"g": {"type": "x", "value": 1}}}],
        });
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(e.contains("unknown type"), "{e}");
    }

    // An array size and a pass size are bounded.

    #[test]
    fn array_size_is_bounded() {
        ArrayInput::new("a", json!(16)).unwrap();
        ArrayInput::new("a", json!([96, 16])).unwrap();
        for bad in [
            json!(0),
            json!(-4),
            json!(1e11),
            json!(100000),
            json!("16"),
            json!(4.5),
            json!(null),
        ] {
            assert!(ArrayInput::new("a", bad.clone()).is_err(), "{bad}");
        }
        for bad in [
            json!([0, 4]),
            json!([4, 0]),
            json!([100000, 3]),
            json!([8, 4, 2]),
            json!([8, "4"]),
        ] {
            assert!(ArrayInput::new("a", bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn huge_array_rejected_from_dict() {
        let d = json!({
            "version": 1,
            "passes": [{"name": "m", "code": GOOD, "arrays": [{"name": "a", "size": 1e11}]}],
        });
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(e.contains("size"), "{e}");
        assert!(e.contains("100000000000"), "{e}"); // A whole float reports as an int.
    }

    #[test]
    fn two_d_array_texel_cap() {
        ArrayInput::new("a", json!([4096, 4096])).unwrap(); // Exactly the cap.
        let e = ArrayInput::new("a", json!([8192, 4096]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("cap"), "{e}");
    }

    #[test]
    fn pass_size_is_bounded() {
        let sized =
            |size: Value| Pass::new("p", GOOD, vec![], vec![], ParamMap::new(), "", Some(size));
        sized(json!([32, 18])).unwrap();
        assert!(sized(json!([100000, 100000]))
            .unwrap_err()
            .to_string()
            .contains("16384"));
        assert!(sized(json!([16384, 16384]))
            .unwrap_err()
            .to_string()
            .contains("cap"));
    }

    // A pass name is a buffer name, so it is unique.

    #[test]
    fn duplicate_pass_names_rejected() {
        let e = Scene::of(vec![
            Pass::of("a", GOOD).unwrap(),
            Pass::of("a", GOOD).unwrap(),
        ])
        .unwrap_err()
        .to_string();
        assert!(e.contains("twice"), "{e}");
        let d = json!({
            "version": 1,
            "passes": [{"name": "a", "code": GOOD}, {"name": "a", "code": GOOD}],
        });
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(e.contains("twice"), "{e}");
    }

    // One error carries the scene, the pass and the field.

    #[test]
    fn a_key_that_ghee_does_not_take_is_refused_naming_the_closest_key() {
        // Each row puts one key into a scene that loads without it.
        let pass = |extra: Value| {
            let mut p = json!({"name": "main", "code": GOOD});
            p.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            p
        };
        let layer = |extra: Value| {
            let mut d = json!({"text": "t", "px_per_em": 20});
            d.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            json!({"version": 1, "texts": {"t": {"font": "a.ttf"}},
                   "passes": [pass(json!({"layers": [d]}))]})
        };
        let rows = [
            (
                json!({"version": 1, "paramz": {}, "passes": [pass(json!({}))]}),
                "scene: the scene takes no 'paramz', the closest key is 'params'",
            ),
            (
                json!({"version": 1, "code": GOOD, "passes": [pass(json!({}))]}),
                "scene: the scene takes no 'code', which sits on a pass",
            ),
            (
                json!({"version": 1, "params": {"g": {"type": "float", "value": 1, "mni": 0}},
                       "passes": [pass(json!({}))]}),
                "scene: param 'g' takes no 'mni', the closest key is 'min'",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"params": {"g": {
                    "type": "float", "value": 1, "default": 0}}}))]}),
                "scene: pass 'main': param 'g' takes no 'default', only type, value, min, max, \
                 step and options",
            ),
            (
                json!({"version": 1, "assets": {"a": {"file": "a.png", "sampler": "linear"}},
                       "passes": [pass(json!({}))]}),
                "scene: asset 'a' takes no 'sampler', the closest key is 'sampling'",
            ),
            (
                json!({"version": 1, "texts": {"t": {"font": "a.ttf", "alig": "center"}},
                       "passes": [pass(json!({}))]}),
                "scene: text 't' takes no 'alig', the closest key is 'align'",
            ),
            (
                json!({"version": 1, "texts": {"t": {"font": "a.ttf", "pad": 1}},
                       "passes": [pass(json!({}))]}),
                "scene: text 't' takes no 'pad', which sits on a layer",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"chanels": []}))]}),
                "scene: pass 'main' takes no 'chanels', the closest key is 'channels'",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"font": "a.ttf"}))]}),
                "scene: pass 'main' takes no 'font', which sits on a text",
            ),
            (
                layer(json!({"fill": "slug"})),
                "scene: pass 'main' layer 't' takes no 'fill', which sits on text 't'",
            ),
            (
                layer(json!({"fil": "slug"})),
                "scene: pass 'main' layer 't' takes no 'fil', the closest key is 'fill', which \
                 sits on text 't'",
            ),
            (
                layer(json!({"glow": 1})),
                "scene: pass 'main' layer 't' takes no 'glow', only name, text, vertex, px_per_em, \
                 fragment, blend, pad and emoji",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"channels": [{"bufer": "main"}]}))]}),
                "scene: pass 'main': channel 0 takes no 'bufer', the closest key is 'buffer'",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"channels": [{"pass": "main"}]}))]}),
                "scene: pass 'main': channel 0 takes no 'pass', only buffer",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"arrays": [{"name": "v", "sise": 4}]}))]}),
                "scene: pass 'main': array 'v' takes no 'sise', the closest key is 'size'",
            ),
            (
                json!({"version": 1, "common": {"cod": COMMON}, "passes": [pass(json!({}))]}),
                "scene: 'common' takes no 'cod', the closest key is 'code'",
            ),
        ];
        for (d, want) in rows {
            let got = Scene::from_dict(&d, None, None).map(|_| ()).unwrap_err();
            assert_eq!(got.to_string(), want);
        }
    }

    #[test]
    fn malformed_scene_json_reports_context() {
        let dir = tmpdir("malformed");
        let bad = [
            (
                "param 'g' has no 'type'",
                json!({"version": 1, "passes": [{"name": "m", "code": GOOD, "params": {"g": {"value": 1}}}]}),
            ),
            (
                "'passes' needs a list",
                json!({"version": 1, "passes": "oops"}),
            ),
            (
                "pass 'm' has no 'code'",
                json!({"version": 1, "passes": [{"name": "m"}]}),
            ),
            (
                "needs a list",
                json!({"version": 1, "passes": [{"name": "m", "code": GOOD, "arrays": "oops"}]}),
            ),
            (
                "a channel needs",
                json!({"version": 1, "passes": [{"name": "m", "code": GOOD, "channels": ["m"]}]}),
            ),
        ];
        for (fragment, d) in bad {
            let p = format!("{dir}/x.scene.json");
            fs::write(&p, d.to_string()).unwrap();
            let msg = Scene::load(&p).unwrap_err().to_string();
            assert!(msg.contains(fragment), "{msg}");
            assert!(msg.contains(&p), "{msg}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invalid_json_reports_the_path() {
        let dir = tmpdir("invalid-json");
        let p = format!("{dir}/x.scene.json");
        fs::write(&p, "{oops").unwrap();
        let e = Scene::load(&p).unwrap_err().to_string();
        assert!(e.contains("invalid JSON") && e.contains(&p), "{e}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn valid_scenes_still_load() {
        for name in ["bars", "trail", "twopass", "blur", "echo", "mix"] {
            let sc = Scene::load(&repo(&format!("ghee/examples/{name}.scene.json"))).unwrap();
            assert!(!sc.passes.is_empty());
            assert!(sc.passes.iter().all(|p| !p.code.is_empty()));
        }
    }

    // A scene's common block is one block of GLSL from one source.

    fn common_scene(common: Option<Value>, passes: Option<Vec<Value>>) -> Value {
        let mut d = Map::new();
        d.insert("version".to_string(), json!(1));
        d.insert(
            "passes".to_string(),
            Value::from(passes.unwrap_or_else(|| vec![json!({"name": "main", "code": CODE})])),
        );
        if let Some(c) = common {
            d.insert("common".to_string(), c);
        }
        Value::Object(d)
    }

    #[test]
    fn common_needs_an_object_with_code_or_shader() {
        for bad in [json!(COMMON), json!(["x"]), json!(3), json!(true)] {
            let e = Scene::from_dict(&common_scene(Some(bad.clone()), None), None, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains("'common' needs an object"), "{bad}: {e}");
        }
        let e = Scene::from_dict(&common_scene(Some(json!(["x"])), None), None, None)
            .unwrap_err()
            .to_string();
        assert_eq!(
            e,
            "scene: 'common' needs an object with 'code' (inline GLSL) or 'shader' \
             (a file next to the scene), got ['x']"
        );
    }

    #[test]
    fn common_takes_exactly_one_of_code_and_shader() {
        for bad in [json!({}), json!({"code": COMMON, "shader": "c.glsl"})] {
            let e = Scene::from_dict(&common_scene(Some(bad.clone()), None), None, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains("exactly one"), "{bad}: {e}");
        }
    }

    #[test]
    fn common_code_needs_text() {
        let e = Scene::from_dict(
            &common_scene(Some(json!({"code": ["x"]})), None),
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("'common' needs GLSL text in 'code'"), "{e}");
    }

    #[test]
    fn common_shader_without_base_dir_raises() {
        let e = Scene::from_dict(
            &common_scene(Some(json!({"shader": "/etc/hostname"})), None),
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("common") && e.contains("shader"), "{e}");
    }

    #[test]
    fn common_shader_never_escapes_the_scene_directory() {
        let root = tmpdir("common-escape");
        let base = format!("{root}/scene");
        fs::create_dir_all(&base).unwrap();
        let secret = format!("{root}/secret.glsl");
        fs::write(&secret, "stolen").unwrap();
        for path in ["../secret.glsl", &secret] {
            let e = Scene::from_dict_in(
                &common_scene(Some(json!({"shader": path})), None),
                &base,
                Reach::Inside,
                None,
            )
            .unwrap_err()
            .to_string();
            assert!(e.contains("outside the scene directory"), "{path}: {e}");
        }
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn missing_common_file_names_the_scene_and_the_common() {
        let dir = tmpdir("common-missing");
        let p = format!("{dir}/x.scene.json");
        fs::write(
            &p,
            common_scene(Some(json!({"shader": "gone.glsl"})), None).to_string(),
        )
        .unwrap();
        let msg = Scene::load(&p).unwrap_err().to_string();
        assert!(
            msg.contains(&p) && msg.contains("common") && msg.contains("gone.glsl"),
            "{msg}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn inline_common_roundtrips() {
        let d = common_scene(Some(json!({"code": COMMON})), None);
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn a_common_file_is_inlined_like_a_pass_shader() {
        let base = tmpdir("common-file");
        fs::write(format!("{base}/shared.glsl"), COMMON).unwrap();
        let sc = Scene::from_dict(
            &common_scene(Some(json!({"shader": "shared.glsl"})), None),
            Some(&base),
            None,
        )
        .unwrap();
        assert_eq!(sc.common.as_deref(), Some(COMMON));
        // The console posts code back inline, and a save writes the same form.
        assert_eq!(sc.to_dict()["common"], json!({"code": COMMON}));
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_scene_file_loads_its_common_file() {
        let dir = tmpdir("common-scene-file");
        fs::write(format!("{dir}/shared.glsl"), COMMON).unwrap();
        let p = format!("{dir}/x.scene.json");
        fs::write(
            &p,
            common_scene(Some(json!({"shader": "shared.glsl"})), None).to_string(),
        )
        .unwrap();
        assert_eq!(Scene::load(&p).unwrap().common.as_deref(), Some(COMMON));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_empty_common_is_a_block_that_exists() {
        // The console's "+ common" creates a block before anyone types into it.
        let sc =
            Scene::from_dict(&common_scene(Some(json!({"code": ""})), None), None, None).unwrap();
        assert_eq!(sc.common.as_deref(), Some(""));
        assert_eq!(sc.to_dict()["common"], json!({"code": ""}));
    }

    #[test]
    fn a_scene_without_one_carries_no_common_key() {
        let sc = Scene::from_dict(&common_scene(None, None), None, None).unwrap();
        assert!(sc.to_dict().get("common").is_none());
        assert!(sc.common.is_none());
    }

    #[test]
    fn the_common_reaches_every_scene_pass() {
        let sc = Scene::from_dict(
            &common_scene(
                Some(json!({"code": COMMON})),
                Some(vec![
                    json!({"name": "a", "code": CODE}),
                    json!({"name": "b", "code": FLAT}),
                ]),
            ),
            None,
            None,
        )
        .unwrap();
        for p in &sc.passes {
            let g = p.glsl(
                Some(&sc.params),
                sc.common.as_deref(),
                &sc.assets,
                &[],
                false,
            );
            assert!(g.contains(COMMON));
            // The common block follows the preamble and precedes the pass code.
            let pre = g.find("uniform float time;").unwrap();
            let common_at = g.find(COMMON).unwrap();
            let code_at = g.find(&p.code).unwrap();
            assert!(pre < common_at && common_at < code_at);
        }
    }

    #[test]
    fn no_common_assembles_exactly_as_before() {
        let p = Pass::of("m", CODE).unwrap();
        assert_eq!(
            p.glsl(None, None, &[], &[], false),
            p.glsl(None, None, &[], &[], false)
        );
        assert_eq!(
            p.glsl(None, Some(""), &[], &[], false),
            p.glsl(None, None, &[], &[], false)
        );
    }

    // An asset is a named image file, decoded by the host and fed once.

    fn asset_scene(assets: Value) -> Value {
        json!({"version": 1, "passes": [{"name": "m", "code": GOOD}], "assets": assets})
    }

    #[test]
    fn an_asset_resolves_through_ghee_assets_after_the_scene_directory() {
        let scene_dir = tmpdir("asset-scene");
        let asset_dir = tmpdir("asset-store");
        fs::write(format!("{asset_dir}/ghee-assets-test.raw"), [1u8, 2, 3, 4]).unwrap();
        let doc = asset_scene(json!({
            "n": { "file": "ghee-assets-test.raw", "width": 2, "height": 2, "format": "u8" }
        }));
        let mut scene = Scene::from_dict(&doc, None, None).unwrap();
        let e = scene.load_assets(&scene_dir).unwrap_err().to_string();
        assert!(e.contains("No such file") && e.contains(&scene_dir), "{e}");
        std::env::set_var("GHEE_ASSETS", &asset_dir);
        scene.load_assets(&scene_dir).unwrap();
        std::env::remove_var("GHEE_ASSETS");
        assert!(scene.assets[0].image.is_some());
    }

    #[test]
    fn asset_schema_roundtrips() {
        let d = json!({
            "version": 1,
            "passes": [{"name": "m", "code": GOOD}],
            "assets": {
                "stbn": {"file": "stbn.raw", "width": 64, "height": 32, "format": "f32"},
                "mask": {"file": "mask.raw", "width": 8, "height": 8, "format": "u8",
                         "components": 4},
                "logo": {"file": "logo.png", "sampling": "linear"},
                "photo": {"file": "photo.jpg"},
            },
        });
        assert_eq!(Scene::from_dict(&d, None, None).unwrap().to_dict(), d);
    }

    #[test]
    fn raw_asset_needs_width_height_and_format() {
        for k in ["width", "height", "format"] {
            let mut a = json!({"file": "d.raw", "width": 4, "height": 4, "format": "u8"});
            a.as_object_mut().unwrap().remove(k);
            let e = Scene::from_dict(&asset_scene(json!({ "d": a })), None, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains(&format!("'{k}'")), "{k}: {e}");
        }
    }

    #[test]
    fn a_decoded_asset_declares_no_layout() {
        for k in ["width", "height", "format", "components"] {
            let mut a = json!({"file": "logo.png"});
            a.as_object_mut().unwrap().insert(k.to_string(), json!(4));
            let e = Scene::from_dict(&asset_scene(json!({ "d": a })), None, None)
                .unwrap_err()
                .to_string();
            assert!(e.contains(&format!("'{k}'")), "{k}: {e}");
        }
    }

    #[test]
    fn raw_asset_format_is_u8_or_f32() {
        let d = asset_scene(json!({"d": {"file": "d.raw", "width": 4, "height": 4,
                                         "format": "u16"}}));
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(e.contains("'u8' or 'f32'") && e.contains("u16"), "{e}");
    }

    #[test]
    fn raw_asset_components_are_1_3_or_4() {
        let d = asset_scene(json!({"d": {"file": "d.raw", "width": 4, "height": 4,
                                         "format": "u8", "components": 2}}));
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(e.contains("components") && e.contains("2"), "{e}");
    }

    #[test]
    fn asset_sampling_is_nearest_or_linear() {
        let d = asset_scene(json!({"d": {"file": "logo.png", "sampling": "cubic"}}));
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(
            e.ends_with("asset 'd' sampling needs 'nearest' or 'linear', got 'cubic'"),
            "{e}"
        );
        for (a, want) in [
            (
                json!({"file": "logo.png", "filter": "linear"}),
                "asset 'd' takes 'nearest' or 'linear' in 'sampling', got 'filter'",
            ),
            (
                json!({"file": "logo.png", "from_filter": "frc256"}),
                "asset 'd' takes no 'from_filter': a filter declares its own assets",
            ),
            (
                json!({"file": "sans.ttf", "sampling": "linear"}),
                "asset 'd' takes no 'sampling', a .ttf file is a font",
            ),
        ] {
            let e = Scene::from_dict(&asset_scene(json!({ "d": a })), None, None)
                .unwrap_err()
                .to_string();
            assert!(e.ends_with(want), "{e}");
        }
    }

    #[test]
    fn asset_kind_comes_from_the_extension() {
        let d = asset_scene(json!({"d": {"file": "logo.bmp"}}));
        let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
        assert!(e.contains(".raw, .png, .jpg, .jpeg, .ttf or .otf"), "{e}");
    }

    #[test]
    fn asset_name_is_a_glsl_identifier() {
        for bad in ["2d", "no-dash", "sp ace", ""] {
            let d = asset_scene(json!({bad: {"file": "logo.png"}}));
            let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
            assert!(e.contains("identifier"), "{bad}: {e}");
        }
    }

    #[test]
    fn asset_names_collide_with_nothing() {
        let err = |d: &Value| Scene::from_dict(d, None, None).unwrap_err().to_string();
        let vs_array = json!({
            "version": 1,
            "passes": [{"name": "m", "code": GOOD, "arrays": [{"name": "amps", "size": 8}]}],
            "assets": {"amps": {"file": "amps.png"}},
        });
        assert_eq!(
            err(&vs_array),
            "scene: asset 'amps' and array 'amps' in pass 'm' both declare 'amps_size'"
        );
        let vs_scene_param = json!({
            "version": 1,
            "passes": [{"name": "m", "code": GOOD}],
            "params": {"gain": {"type": "float", "value": 1.0}},
            "assets": {"gain": {"file": "gain.png"}},
        });
        assert_eq!(
            err(&vs_scene_param),
            "scene: scene param 'gain' and asset 'gain' both declare 'gain'"
        );
        let vs_pass_param = json!({
            "version": 1,
            "passes": [{"name": "m", "code": GOOD,
                        "params": {"gain": {"type": "float", "value": 1.0}}}],
            "assets": {"gain": {"file": "gain.png"}},
        });
        assert_eq!(
            err(&vs_pass_param),
            "scene: asset 'gain' and param 'gain' in pass 'm' both declare 'gain'"
        );
        // A JSON map holds a name once, so a duplicate needs the constructor.
        let a = Asset::from_dict("logo", &json!({"file": "logo.png"})).unwrap();
        let e = Scene::new(
            vec![Pass::of("m", GOOD).unwrap()],
            json!(1),
            vec![],
            ParamMap::new(),
            None,
            vec![a.clone(), a],
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("twice"), "{e}");
    }

    /// A scene of passes `a` and `b`, a text `label` and the font asset
    /// `sans`, with `extra` merged in at the top and `pass` merged into `a`.
    fn named(extra: Value, pass: Value) -> Value {
        let mut d = json!({
            "version": 1,
            "assets": {"sans": {"file": "sans.ttf"}},
            "texts": {"label": {"value": "hi", "font": "sans"}},
            "passes": [{"name": "a", "code": GOOD}, {"name": "b", "code": GOOD}],
        });
        for (k, v) in extra.as_object().unwrap() {
            match d.get_mut(k).and_then(Value::as_object_mut) {
                Some(o) => o.extend(v.as_object().unwrap().clone()),
                None => d[k] = v.clone(),
            }
        }
        for (k, v) in pass.as_object().unwrap() {
            d["passes"][0][k] = v.clone();
        }
        d
    }

    fn named_err(extra: Value, pass: Value) -> String {
        match Scene::from_dict(&named(extra, pass), None, None) {
            Ok(_) => "loads".to_string(),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn two_declarations_of_one_glsl_name_in_a_pass_are_refused_naming_both() {
        let f = json!({"type": "float", "value": 1.0});
        let c = json!({"type": "color", "value": [1.0, 0.0, 0.0]});
        for (extra, pass, want) in [
            (
                json!({"params": {"gain": f}}),
                json!({"params": {"gain": c}}),
                "scene param 'gain' and param 'gain' in pass 'a' both declare 'gain'",
            ),
            (
                json!({"params": {"amps_at": f}}),
                json!({"arrays": [{"name": "amps", "size": 4}]}),
                "scene param 'amps_at' and array 'amps' in pass 'a' both declare 'amps_at'",
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "amps", "size": [4, 4]}],
                       "params": {"amps_sample2": f}}),
                "param 'amps_sample2' in pass 'a' and array 'amps' in pass 'a' both declare \
                 'amps_sample2'",
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "amps", "size": 4}, {"name": "amps", "size": 4}]}),
                "array 'amps' in pass 'a' is declared twice",
            ),
            (
                json!({"assets": {"logo_size": {"file": "l.png"}, "logo": {"file": "l.png"}}}),
                json!({}),
                "asset 'logo_size' and asset 'logo' both declare 'logo_size'",
            ),
            (
                json!({"params": {"label_cover": c}}),
                json!({}),
                "scene param 'label_cover' and text 'label' both declare 'label_cover'",
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "label", "size": 4}]}),
                "text 'label' and array 'label' in pass 'a' both declare 'label_size'",
            ),
            (
                json!({"params": {"glyph_cover": f}}),
                json!({}),
                "scene param 'glyph_cover' and the layers both declare 'glyph_cover'",
            ),
            (
                json!({}),
                json!({"params": {"glyph_place": f}}),
                "the layers and param 'glyph_place' in pass 'a' both declare 'glyph_place'",
            ),
        ] {
            assert_eq!(named_err(extra, pass), format!("scene: {want}"));
        }
    }

    #[test]
    fn a_name_beside_another_declares_its_own() {
        let f = json!({"type": "float", "value": 1.0});
        // Near misses of every name above: a name that only starts like one,
        // a 2D array beside a text, a param of each pass, a string param, and
        // a name that only looks like ghee's.
        let extra = json!({
            "params": {"gains": f, "amps_ats": f, "label_covers": f, "glyph_size": f,
                       "glyph": f, "ghee": f, "gheex": f, "Ghee_x": f, "frames": f,
                       "DTs": f, "iChannel": f, "iChannelx": f, "times": f,
                       "words": {"type": "string", "value": "w"}},
            "assets": {"logo": {"file": "l.png"}, "logos": {"file": "l.png"}},
        });
        let pass = json!({"params": {"gain": f, "words": f, "logo_sizes": f},
                          "arrays": [{"name": "label", "size": [4, 4]}]});
        let mut d = named(extra, pass);
        d["passes"][1]["params"] = json!({"gain": f});
        assert!(Scene::from_dict(&d, None, None).is_ok());
    }

    #[test]
    fn a_built_in_name_is_refused_as_any_declared_name() {
        let f = json!({"type": "float", "value": 1.0});
        for name in [
            "time",
            "DT",
            "frame",
            "resolution",
            "v_texCoord",
            "iChannel0",
            "iChannel12",
        ] {
            let why = format!("'{name}' is built in");
            for (extra, pass, owner) in [
                (
                    json!({"params": {name: f}}),
                    json!({}),
                    format!("scene param '{name}'"),
                ),
                (
                    json!({}),
                    json!({"params": {name: f}}),
                    format!("param '{name}' in pass 'a'"),
                ),
                (
                    json!({}),
                    json!({"arrays": [{"name": name, "size": 4}]}),
                    format!("array '{name}' in pass 'a'"),
                ),
                (
                    json!({"assets": {name: {"file": "x.png"}}}),
                    json!({}),
                    format!("asset '{name}'"),
                ),
            ] {
                assert_eq!(named_err(extra, pass), format!("scene: {owner}: {why}"));
            }
            let mut d = named(json!({}), json!({}));
            d["texts"][name] = json!({"font": "sans"});
            let e = Scene::from_dict(&d, None, None).unwrap_err().to_string();
            assert_eq!(e, format!("scene: text '{name}': {why}"));
        }
    }

    #[test]
    fn a_name_with_ghees_prefix_is_refused_as_any_declared_name() {
        let f = json!({"type": "float", "value": 1.0});
        let why = "a name that starts with ghee_ belongs to ghee";
        for (extra, pass, owner) in [
            (
                json!({"params": {"ghee_k": f}}),
                json!({}),
                "scene param 'ghee_k'",
            ),
            (
                json!({}),
                json!({"params": {"ghee_k": f}}),
                "param 'ghee_k' in pass 'a'",
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "ghee_k", "size": 4}]}),
                "array 'ghee_k' in pass 'a'",
            ),
            (
                json!({"assets": {"ghee_k": {"file": "x.png"}}}),
                json!({}),
                "asset 'ghee_k'",
            ),
            (
                json!({"texts": {"ghee_label": {"value": "w", "font": "sans"}}}),
                json!({}),
                "text 'ghee_label'",
            ),
        ] {
            assert_eq!(named_err(extra, pass), format!("scene: {owner}: {why}"));
        }
    }

    #[test]
    fn a_param_or_an_array_name_is_a_glsl_identifier() {
        let f = json!({"type": "float", "value": 1.0});
        for (extra, pass, want) in [
            (
                json!({"params": {"a b": f}}),
                json!({}),
                "scene param name 'a b' is not a GLSL identifier",
            ),
            (
                json!({}),
                json!({"params": {"2x": f}}),
                "param name '2x' in pass 'a' is not a GLSL identifier",
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "a-b", "size": 4}]}),
                "array name 'a-b' in pass 'a' is not a GLSL identifier",
            ),
        ] {
            assert_eq!(named_err(extra, pass), format!("scene: {want}"));
        }
        // A string param reaches no GLSL.
        let words = json!({"params": {"two words": {"type": "string", "value": "w"}}});
        assert_eq!(named_err(words, json!({})), "loads");
    }

    #[test]
    fn a_filter_meets_the_scene_names_that_reach_its_pass() {
        let _g = crate::filters::env_guard();
        let f = json!({"type": "float", "value": 1.0});
        let mut sc = Scene::from_dict(
            &named(json!({"params": {"amount": f}}), json!({})),
            None,
            None,
        )
        .unwrap();
        let e = crate::filters::apply_filters(&mut sc, &["dither".to_string()], None).unwrap_err();
        assert_eq!(
            e.to_string(),
            "scene param 'amount' and param 'amount' in pass 'dither' both declare 'amount'"
        );
        let doc = named(json!({"params": {"frc256_bluenoise": f}}), json!({}));
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        let e = crate::filters::apply_filters(&mut sc, &["frc256".to_string()], None).unwrap_err();
        assert_eq!(
            e.to_string(),
            "filter 'frc256': scene param 'frc256_bluenoise' and asset 'bluenoise' of filter \
             'frc256' both declare 'frc256_bluenoise'"
        );
        // A text's names and a layer's stay out of a filter pass.
        let doc = named(json!({"params": {"bluenoise": f}}), json!({}));
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        crate::filters::apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
    }

    #[test]
    fn preamble_declares_asset_sampler_and_size() {
        let sc = Scene::from_dict(
            &asset_scene(json!({"logo": {"file": "logo.png"}})),
            None,
            None,
        )
        .unwrap();
        let g = sc.passes[0].glsl(Some(&sc.params), None, &sc.assets, &[], false);
        assert!(g.contains("uniform sampler2D logo;"));
        assert!(g.contains("uniform vec2 logo_size;"));
    }

    #[test]
    fn a_raw_length_mismatch_names_the_path_and_the_shape() {
        let dir = tmpdir("asset-mismatch");
        fs::write(format!("{dir}/d.raw"), [0u8; 5]).unwrap();
        let p = format!("{dir}/x.scene.json");
        fs::write(
            &p,
            asset_scene(json!({"d": {"file": "d.raw", "width": 2, "height": 2,
                                     "format": "u8"}}))
            .to_string(),
        )
        .unwrap();
        let e = Scene::load(&p).unwrap_err().to_string();
        assert!(e.contains(&format!("{dir}/d.raw")), "{e}");
        assert!(
            e.contains("5 bytes") && e.contains("2x2x1") && e.contains("needs 4"),
            "{e}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_f32_raw_asset_round_trips_little_endian() {
        let dir = tmpdir("asset-f32");
        let vals = [0.0f32, 0.5, -1.0, 3.25];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(format!("{dir}/d.raw"), &bytes).unwrap();
        let p = format!("{dir}/x.scene.json");
        fs::write(
            &p,
            asset_scene(json!({"d": {"file": "d.raw", "width": 4, "height": 1,
                                     "format": "f32"}}))
            .to_string(),
        )
        .unwrap();
        let sc = Scene::load(&p).unwrap();
        let img = sc.assets[0].image.as_ref().unwrap();
        assert_eq!((img.width, img.height, img.components), (4, 1, 1));
        assert_eq!(img.pixels, crate::assets::Pixels::F32(vals.to_vec()));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_scene_that_its_user_runs_names_a_file_anywhere() {
        // Each file sits outside the scene's directory: a relative path that
        // leaves it, and absolute paths.
        let root = tmpdir("anywhere");
        let base = format!("{root}/scene");
        fs::create_dir_all(&base).unwrap();
        fs::write(format!("{root}/d.raw"), [7u8; 4]).unwrap();
        fs::write(format!("{root}/e.raw"), [9u8; 4]).unwrap();
        fs::write(format!("{root}/in.glsl"), GOOD).unwrap();
        let raw = |file: String| json!({"file": file, "width": 2, "height": 2, "format": "u8"});
        let mut d = asset_scene(json!({"d": raw("../d.raw".to_string()),
                                       "e": raw(format!("{root}/e.raw"))}));
        d["passes"][0] = json!({"name": "main", "shader": format!("{root}/in.glsl")});
        let p = format!("{base}/x.scene.json");
        fs::write(&p, d.to_string()).unwrap();
        let sc = Scene::load(&p).unwrap();
        let px = |i: usize| sc.assets[i].image.as_ref().unwrap().pixels.clone();
        assert_eq!(px(0), crate::assets::Pixels::U8(vec![7u8; 4]));
        assert_eq!(px(1), crate::assets::Pixels::U8(vec![9u8; 4]));
        assert_eq!(sc.passes[0].code, GOOD);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_asset_never_escapes_the_scene_directory() {
        let root = tmpdir("asset-escape");
        let base = format!("{root}/scene");
        fs::create_dir_all(&base).unwrap();
        fs::write(format!("{root}/secret.raw"), [0u8; 4]).unwrap();
        let p = format!("{base}/x.scene.json");
        fs::write(
            &p,
            asset_scene(
                json!({"d": {"file": "../secret.raw", "width": 2, "height": 2,
                                     "format": "u8"}}),
            )
            .to_string(),
        )
        .unwrap();
        let e = Scene::load_in(&p, None, Reach::Inside)
            .unwrap_err()
            .to_string();
        assert!(e.contains("outside the scene directory"), "{e}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_path_inside_a_directory_takes_whole_components() {
        assert!(inside("/", "/"));
        assert!(inside("/a.png", "/"));
        assert!(inside("/a/b", "/a/b"));
        assert!(inside("/a/b/c.png", "/a/b"));
        assert!(!inside("/a/bc.png", "/a/b"));
        assert!(!inside("/a", "/a/b"));
    }

    #[test]
    fn the_root_directory_holds_assets_and_shaders() {
        let dir = tmpdir("in-root");
        fs::write(format!("{dir}/n.raw"), [1u8, 2, 3, 4]).unwrap();
        fs::write(format!("{dir}/in.glsl"), GOOD).unwrap();
        let rel = dir.trim_start_matches('/');
        let mut scene = Scene::from_dict(
            &asset_scene(json!({
                "n": {"file": format!("{rel}/n.raw"), "width": 2, "height": 2, "format": "u8"}
            })),
            None,
            None,
        )
        .unwrap();
        scene.load_assets("/").unwrap();
        let img = scene.assets[0].image.as_ref().unwrap();
        assert_eq!(img.pixels, crate::assets::Pixels::U8(vec![1, 2, 3, 4]));
        let d =
            json!({"version": 1, "passes": [{"name": "main", "shader": format!("{rel}/in.glsl")}]});
        let sc = Scene::from_dict(&d, Some("/"), None).unwrap();
        assert_eq!(sc.passes[0].code, GOOD);
        let mut missing = Scene::from_dict(
            &asset_scene(json!({"m": {"file": "ghee-no-such.png"}})),
            None,
            None,
        )
        .unwrap();
        let e = missing.load_assets("/").unwrap_err().to_string();
        assert!(e.contains("(/ghee-no-such.png)"), "{e}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_asset_file_names_the_path() {
        let dir = tmpdir("asset-missing");
        let p = format!("{dir}/x.scene.json");
        fs::write(
            &p,
            asset_scene(json!({"logo": {"file": "logo.png"}})).to_string(),
        )
        .unwrap();
        let e = Scene::load(&p).unwrap_err().to_string();
        assert!(e.contains(&format!("{dir}/logo.png")), "{e}");
        assert!(e.contains("No such file or directory"), "{e}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn saved_scene_reloads_unchanged() {
        let dir = tmpdir("save");
        let sc = Scene::load(&repo("ghee/examples/echo.scene.json")).unwrap();
        let p = format!("{dir}/echo.scene.json");
        sc.save(&p).unwrap();
        assert_eq!(Scene::load(&p).unwrap().to_dict(), sc.to_dict());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_scene_file_reports_the_os_error() {
        let e = Scene::load("/nope/x.scene.json").unwrap_err().to_string();
        assert_eq!(
            e,
            "[Errno 2] No such file or directory: '/nope/x.scene.json'"
        );
    }

    #[test]
    fn a_pass_knows_when_it_reads_itself() {
        let feedback = Pass::new(
            "main",
            CODE,
            vec![],
            vec![Channel::new(Some("main"))],
            ParamMap::new(),
            "",
            None,
        )
        .unwrap();
        assert!(feedback.reads_self());
        assert!(!Pass::new(
            "main",
            CODE,
            vec![],
            vec![Channel::new(Some("other"))],
            ParamMap::new(),
            "",
            None
        )
        .unwrap()
        .reads_self());
        assert!(!Pass::of("main", CODE).unwrap().reads_self());
    }

    #[test]
    fn the_output_is_the_last_pass() {
        let sc = Scene::of(vec![
            Pass::of("a", CODE).unwrap(),
            Pass::of("b", CODE).unwrap(),
        ])
        .unwrap();
        assert_eq!(sc.output().name, "b");
    }

    #[test]
    fn a_value_quoted_in_an_error_keeps_its_form() {
        assert_eq!(py_repr(&json!(null)), "None");
        assert_eq!(py_repr(&json!(true)), "True");
        assert_eq!(py_repr(&json!(4)), "4");
        assert_eq!(py_repr(&json!(4.5)), "4.5");
        assert_eq!(py_repr(&json!(1e11)), "100000000000.0");
        assert_eq!(py_repr(&json!(1e16)), "1e+16");
        assert_eq!(py_repr(&json!(1e-5)), "1e-05");
        assert_eq!(py_repr(&json!(0.0001)), "0.0001");
        assert_eq!(py_repr(&json!("x")), "'x'");
        assert_eq!(py_repr(&json!([1, "a"])), "[1, 'a']");
        assert_eq!(py_repr(&json!({"k": 1})), "{'k': 1}");
        assert_eq!(py_float(96.0), "96.0");
    }

    /// The scene as to_dict serialises it, with no space after a separator.
    fn canon(s: &Scene) -> String {
        serde_json::to_string(&s.to_dict()).unwrap()
    }

    #[test]
    fn to_dict_matches_the_reference_json() {
        // What to_dict serialises for this input: key order, dropped keys,
        // and int or float per field.
        let d = json!({
            "version": 2,
            "passes": [
                {"name": "a", "code": "x", "size": [8, 8]},
                {"name": "b", "code": "y",
                 "arrays": [{"name": "v", "size": 4}, {"name": "m", "size": [3, 5]}],
                 "channels": [{"buffer": "a"}, {"buffer": null}],
                 "params": {"g": {"type": "float", "value": 1, "min": 0, "max": 2, "step": 0.5},
                            "n": {"type": "int", "value": 3},
                            "b": {"type": "bool", "value": true},
                            "c": {"type": "color", "value": [1, 0, 0.5]},
                            "v2": {"type": "vec2", "value": [1, 2]},
                            "v4": {"type": "vec4", "value": [1, 2, 3, 4]},
                            "s": {"type": "string", "value": "hi", "options": ["hi", "bye"]}},
                 "filter": "dither"}],
            "filters": ["dither", "dither"],
            "params": {"S": {"type": "float", "value": 0.5}},
            "common": {"code": "float f(){ return 1.0; }"},
        });
        assert_eq!(
            canon(&Scene::from_dict(&d, None, None).unwrap()),
            r#"{"version":2,"passes":[{"name":"a","code":"x","size":[8,8]},{"name":"b","code":"y","arrays":[{"name":"v","size":4},{"name":"m","size":[3,5]}],"channels":[{"buffer":"a"},{"buffer":null}],"params":{"g":{"type":"float","value":1.0,"min":0,"max":2,"step":0.5},"n":{"type":"int","value":3},"b":{"type":"bool","value":true},"c":{"type":"color","value":[1.0,0.0,0.5]},"v2":{"type":"vec2","value":[1.0,2.0]},"v4":{"type":"vec4","value":[1.0,2.0,3.0,4.0]},"s":{"type":"string","value":"hi","options":["hi","bye"]}},"filter":"dither"}],"common":{"code":"float f(){ return 1.0; }"},"filters":["dither","dither"],"params":{"S":{"type":"float","value":0.5}}}"#
        );
        // A null range or option list is no range or option list. to_dict drops
        // an empty filter mark. An absent version takes the default.
        let sparse = json!({
            "passes": [{"name": "m", "code": "x", "filter": "",
                        "params": {"g": {"type": "int", "value": 2.0, "min": null,
                                         "max": null, "step": null, "options": null}}}],
        });
        assert_eq!(
            canon(&Scene::from_dict(&sparse, None, None).unwrap()),
            r#"{"version":1,"passes":[{"name":"m","code":"x","params":{"g":{"type":"int","value":2}}}]}"#
        );
        // "" is a common block the console makes. None is no block.
        let empty_common = json!({"passes": [{"name": "m", "code": "z"}], "common": {"code": ""}});
        assert_eq!(
            canon(&Scene::from_dict(&empty_common, None, None).unwrap()),
            r#"{"version":1,"passes":[{"name":"m","code":"z"}],"common":{"code":""}}"#
        );
        // to_dict carries any version through unchanged.
        let odd = json!({"version": "x", "passes": [{"name": "m", "code": "x"}]});
        assert_eq!(
            canon(&Scene::from_dict(&odd, None, None).unwrap()),
            r#"{"version":"x","passes":[{"name":"m","code":"x"}]}"#
        );
    }

    #[test]
    fn to_dict_round_trips_the_shipped_scenes() {
        // A scene reloaded from its own to_dict serialises to the same bytes.
        // A second save never rewrites the file.
        let dir = repo("ghee/examples");
        let mut names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".scene.json"))
            .collect();
        names.sort();
        assert!(!names.is_empty());
        for n in names {
            let one = canon(&Scene::load(&format!("{dir}/{n}")).unwrap());
            let back: Value = serde_json::from_str(&one).unwrap();
            assert_eq!(
                canon(&Scene::from_dict(&back, None, None).unwrap()),
                one,
                "{n}"
            );
        }
    }
}
