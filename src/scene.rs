//! The scene model, and the GLSL that it compiles to.
//!
//! A [`Scene`] holds an ordered list of passes, and the params and assets
//! that the passes share. A [`Pass`] holds its GLSL, its channels, arrays and
//! params, and its layers, which `crate::layer` describes.
//! [`Scene::load`] reads a scene JSON file: it refuses a key that it does not
//! take, checks every name and size, and reads the files that the scene
//! names, from the scene's directory or from `$GHEE_ASSETS`. A [`Reach`] sets
//! whether a file may lie outside that directory. [`Scene::of`] and
//! [`Pass::of`] build a scene in code.
//!
//! [`Pass::glsl`] writes the fragment shader of a pass in three parts: the
//! preamble, the scene's common block, then the pass's own GLSL, its
//! `fragment`. The preamble
//! declares what every pass reads, the varying `uv` and the uniforms `iTime`,
//! `iTimeDelta`, `iFrame`, `iResolution` and `iChannel0` and up, then one
//! uniform for each param
//! under the param's name, and the samplers and the reader functions of each
//! array, asset and text layer. The C engine sets each uniform by looking up
//! that same name in the compiled program, so a name that the preamble writes
//! is the contract between the two.

use std::borrow::Cow;
use std::fs;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::layer::{is_image_file, Address, Form, Layer, Rgb, TextLayer, COLOUR_FORMS};
use crate::shape::Shape;

const GLSL_TYPE: [(&str, &str); 3] = [("float", "float"), ("int", "int"), ("bool", "bool")];

const VEC_ARITY: [(&str, usize); 3] = [("vec2", 2), ("vec3", 3), ("vec4", 4)];

/// Every name that ghee writes into GLSL and an author never writes starts
/// with this prefix: a layer's own uniforms, textures and functions, and the
/// vertex flip.
pub const PREFIX: &str = "ghee_";

/// The uniforms and the varying that ghee declares in every pass for its
/// author to read, beside iChannel0 and up.
const BUILT_INS: [&str; 5] = ["iTime", "iTimeDelta", "iFrame", "iResolution", "uv"];

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

/// The name of the text layer that Scene::measure() lays words out in.
const MEASURED: &str = "measured";

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

/// Why `name` breaks the name rule, None when it keeps it. A name is ASCII
/// letters, digits and '_', with a letter first and a letter or a digit last.
/// GLSL reserves a name that holds "__" or starts with "gl_", and ghee one
/// that starts with "ghee".
pub(crate) fn name_fault(name: &str) -> Option<&'static str> {
    let b = name.as_bytes();
    let ascii = b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_');
    if !ascii || !b.first().is_some_and(u8::is_ascii_alphabetic) {
        return Some("needs ASCII letters, digits and '_', and a letter first");
    }
    if name.ends_with('_') {
        return Some("needs a letter or a digit last");
    }
    if name.contains("__") {
        return Some("holds '__', which GLSL reserves");
    }
    if name.starts_with("gl_") {
        return Some("starts with 'gl_', which GLSL reserves");
    }
    if name.starts_with("ghee") {
        return Some("starts with 'ghee', which ghee reserves");
    }
    None
}

/// The error for the first name of `names` that an earlier one holds, each
/// name with its declaration, and `at` where they sit.
fn first_shared(names: &[(&str, String)], at: &str) -> Option<String> {
    let (i, (n, owner)) = names
        .iter()
        .enumerate()
        .find(|(i, (n, _))| names[..*i].iter().any(|(m, _)| m == n))?;
    let (_, first) = names[..i].iter().find(|(m, _)| m == n)?;
    Some(match first == owner {
        true => format!("{owner}{at} is declared twice"),
        false => format!("{first} and {owner}{at} share a name"),
    })
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

/// An entry of a scene's `filters`, as its name and its eye: a filter's
/// name, or `{"name": <name>, "visible": <bool>}`.
fn filter_entry(v: &Value) -> Option<(String, bool)> {
    match v {
        Value::String(n) => Some((n.clone(), true)),
        Value::Object(o) if o.keys().all(|k| k == "name" || k == "visible") => {
            let visible = match o.get("visible") {
                None => true,
                Some(v) => v.as_bool()?,
            };
            Some((o.get("name")?.as_str()?.to_string(), visible))
        }
        _ => None,
    }
}

/// The keys of the scene's top level.
const SCENE_KEYS: [&str; 6] = ["version", "passes", "filters", "params", "common", "assets"];

/// The keys of a pass.
pub(crate) const PASS_KEYS: [&str; 10] = [
    "name", "fragment", "arrays", "channels", "params", "filter", "size", "mipmaps", "layers",
    "visible",
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

/// The keys of a param.
const PARAM_KEYS: [&str; 6] = ["type", "value", "min", "max", "step", "options"];

/// The keys of a channel, an array and a file of GLSL.
const CHANNEL_KEYS: [&str; 1] = ["buffer"];
const ARRAY_KEYS: [&str; 2] = ["name", "size"];
const GLSL_FILE_KEYS: [&str; 1] = ["file"];

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

/// A key that several keys start with, and the one of them that it most
/// likely means.
const MEANS: [(&str, &str); 1] = [("font", "font_family")];

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
    let meant = MEANS
        .iter()
        .find(|(typed, _)| typed == key)
        .and_then(|(_, k)| match takes.contains(k) {
            true => Some((0, *k, None)),
            false => elsewhere
                .iter()
                .find(|(e, _)| e == k)
                .map(|(e, on)| (0, *e, Some(on))),
        });
    let closest = meant.or_else(|| {
        own.chain(other)
            .filter_map(|(k, on)| near(key, k).map(|d| (d, k, on)))
            .min_by_key(|(d, _, _)| *d)
    });
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
pub(crate) fn py_float(f: f64) -> String {
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
    /// The files inside the scene's directory alone, and the emoji font
    /// wherever crate::emoji finds it. The console reads every scene this
    /// way, since anyone who reaches it posts and saves scenes.
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

/// GLSL as a pass's `fragment` or the scene's `common` holds it: the text
/// itself, or `{"file": <path>}`, a file that resolves in the scene's
/// directory. `what` names the holder in an error.
///
/// A scene with no directory names no file. A console post and an in-memory
/// scene have no directory. A scene read with Reach::Inside reads inside its
/// directory only.
fn glsl_of(v: &Value, files: Option<Files>, what: &str) -> Result<String> {
    let path = match v {
        Value::String(s) => return Ok(s.clone()),
        Value::Object(o) => {
            known_keys(o, what, &GLSL_FILE_KEYS, &[])?;
            o.get("file")
        }
        _ => None,
    };
    let Some(path) = path else {
        bail!(
            "{what} needs GLSL text or {{\"file\": <path>}}, got {}",
            py_repr(v)
        );
    };
    let Value::String(name) = path else {
        bail!("{what} file needs a path, got {}", py_repr(path));
    };
    let Some(files) = files.filter(|f| !f.dir.is_empty()) else {
        bail!(
            "{what} names the file {} but the scene has no directory to resolve \
             it against; give the GLSL text itself",
            py_repr(path)
        );
    };
    let root = normpath(&abspath(files.dir));
    let full = normpath(&join(&root, name));
    if files.reach == Reach::Inside && !inside(&full, &root) {
        bail!(
            "{what} file {} is outside the scene directory",
            py_repr(path)
        );
    }
    match fs::read_to_string(&full) {
        Ok(code) => {
            crate::sources::record(&full);
            Ok(code)
        }
        Err(e) => bail!("{what} file {}: {}", py_repr(path), strerror(&e)),
    }
}

/// The scene's shared GLSL, which arrives as a pass's `fragment` does. None
/// is no common block, and "" is an empty block the console creates.
fn common_block(d: &Map<String, Value>, files: Option<Files>) -> Result<Option<String>> {
    match d.get("common") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => glsl_of(v, files, "'common'").map(Some),
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
    Color(Rgb),
    Vec(Vec<f64>),
    Str(String),
}

impl ParamValue {
    pub fn to_json(&self) -> Value {
        match self {
            ParamValue::Float(f) => Value::from(*f),
            ParamValue::Int(i) => Value::from(*i),
            ParamValue::Bool(b) => Value::from(*b),
            ParamValue::Color(c) => c.to_json(),
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
    /// ghee declares this string param for the text layer of its name, and
    /// the param holds the layer's words. A save writes them into the layer's
    /// `text`.
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
        let stored = if ty == "color" {
            let Some(rgb) = Rgb::from_json(&value) else {
                bail!(
                    "color param '{name}' needs {COLOUR_FORMS}, got {}",
                    py_repr(&value)
                );
            };
            ParamValue::Color(rgb)
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
        Param::plain(name, "color", ParamValue::Color(Rgb::numbers(rgb)))
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

    /// The string param that ghee declares for a text layer, holding
    /// `words`.
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

/// The [r, g, b] of a colour written `#rrggbb`, each channel 0..1.
pub(crate) fn hex_colour(s: &str) -> Option<[f64; 3]> {
    let hex = s
        .strip_prefix('#')
        .filter(|h| h.len() == 6 && h.bytes().all(|b| b.is_ascii_hexdigit()))?;
    let channel = |i: usize| {
        u8::from_str_radix(&hex[i..i + 2], 16)
            .ok()
            .map(|v| f64::from(v) / 255.0)
    };
    Some([channel(0)?, channel(2)?, channel(4)?])
}

pub(crate) fn number_list(v: &Value, n: usize) -> Option<Vec<f64>> {
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

    /// The param at place `k` in the order of the map.
    pub(crate) fn at_mut(&mut self, k: usize) -> Option<&mut Param> {
        self.0.get_mut(k).map(|(_, p)| p)
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
        let stored = if let Value::Array(s) = &size {
            if s.len() != 2 || !s.iter().all(is_dim) {
                bail!(
                    "array '{name}' size needs two ints in 1..{MAX_DIM} [w, h], got {}",
                    py_repr(&size)
                );
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
            return ArrayInput::new(&name, Value::Array(vec![to_int(&a[0]), to_int(&a[1])]));
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
        l.push(format!("const int {}_count = {n};", a.name));
    }
    l
}

/// The preamble declarations of one image asset: its sampler and its size.
fn asset_lines(a: &Asset) -> Vec<String> {
    vec![
        format!("uniform sampler2D {}_tex;", a.name),
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
                "float {n}_sample2(vec2 p){{ return texture2D({n}_tex, p).r; }}"
            )];
        }
        return vec![
            format!("float {n}_sample2(vec2 p){{"),
            format!(
                "  vec2 dim = vec2({}, {});",
                py_float(w as f64),
                py_float(h as f64)
            ),
            "  vec2 x = p * dim - 0.5;".to_string(),
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
         vec2((float(i0)+0.5)/float({n}_count), 0.5)).r; }}"
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
        format!("  float x = t * float({n}_count) - 0.5;"),
        "  float i = floor(x);".to_string(),
        format!("  float last = float({n}_count - 1);"),
        format!("  float a = {n}_at(int(clamp(i, 0.0, last)));"),
        format!("  float b = {n}_at(int(clamp(i + 1.0, 0.0, last)));"),
        "  return mix(a, b, x - i);".to_string(),
        "}".to_string(),
    ]
}

/// A named image or font file. Every pass samples an image as
/// `uniform sampler2D <name>_tex` and reads a `vec2 <name>_size` of (width,
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
                "a channel needs an object like {{\"buffer\": \"pass\"}}, got {}",
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

/// Layer `k` of pass `pass` by its place, as "the 2nd layer of pass 'main'".
fn layer_at(pass: &str, k: usize) -> String {
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

#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    pub name: String,
    /// The pass's fragment shader. A pass with layers may leave it empty: the
    /// pass then starts each frame transparent and draws its layers alone.
    pub fragment: String,
    pub arrays: Vec<ArrayInput>,
    pub channels: Vec<Channel>,
    pub params: ParamMap,
    /// A non-empty name marks a pass as a named filter's output. The console
    /// renders such a pass read-only.
    pub filter: String,
    /// The [w, h] of this pass's render. None is the renderer's size. A
    /// consumer samples a sized pass through normalized UVs, so it ignores
    /// the size. The output pass renders at the renderer's size whatever
    /// this holds.
    pub size: Option<(i64, i64)>,
    /// The pass's buffer holds a mip chain, built after the pass renders each
    /// frame, and every channel that reads it samples trilinearly.
    pub mipmaps: bool,
    /// Layers drawn into this pass's buffer after its quad, in order.
    pub layers: Vec<Layer>,
    /// False hides the pass: it draws a copy of the pass before it, or black
    /// as the first pass, and its layers draw nothing. It keeps its params,
    /// arrays and layers. Scene::drawn_passes() says what the engine draws.
    pub visible: bool,
}

impl Pass {
    pub fn new(
        name: &str,
        fragment: &str,
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
            fragment: fragment.to_string(),
            arrays,
            channels,
            params,
            filter: filter.to_string(),
            size,
            mipmaps: false,
            layers: Vec::new(),
            visible: true,
        })
    }

    /// A pass with a name and GLSL only. Every other field takes its default.
    pub fn of(name: &str, fragment: &str) -> Result<Pass> {
        Pass::new(
            name,
            fragment,
            Vec::new(),
            Vec::new(),
            ParamMap::new(),
            "",
            None,
        )
    }

    /// False for a pass of layers alone, with no quad of its own.
    pub fn has_fragment(&self) -> bool {
        !self.fragment.trim().is_empty()
    }

    /// The error for a pass with no fragment and no layers, which draws nothing.
    pub(crate) fn empty(&self) -> Option<String> {
        (!self.has_fragment() && self.layers.is_empty())
            .then(|| format!("pass '{}' has no 'fragment' and no 'layers'", self.name))
    }

    /// True when a channel reads this pass's own buffer.
    #[allow(dead_code)]
    pub fn reads_self(&self) -> bool {
        self.channels
            .iter()
            .any(|c| c.buffer.as_deref() == Some(self.name.as_str()))
    }

    /// A `fragment` file resolves in `base_dir`, with the reach of a scene
    /// that its user runs.
    pub fn from_dict(d: &Value, base_dir: Option<&str>) -> Result<Pass> {
        Pass::parse(d, any_file(base_dir))
    }

    fn parse(d: &Value, files: Option<Files>) -> Result<Pass> {
        let Value::Object(o) = d else {
            bail!("a pass needs a JSON object, got {}", py_repr(d));
        };
        let name = o.get("name").map_or("main".to_string(), py_str);
        let elsewhere = [
            keys_on(&PASS_KEYS, &crate::layer::layer_keys(), "a layer"),
            keys_on(&PASS_KEYS, &ASSET_KEYS, "an asset"),
            keys_on(&PASS_KEYS, &CHANNEL_KEYS, "a channel"),
            keys_on(&PASS_KEYS, &PARAM_KEYS, "a param"),
            keys_on(&PASS_KEYS, &SCENE_KEYS, "the scene"),
        ]
        .concat();
        known_keys(o, &format!("pass '{name}'"), &PASS_KEYS, &elsewhere)?;
        let fragment = match o.get("fragment").filter(|v| !v.is_null()) {
            Some(v) => glsl_of(v, files, &format!("pass '{name}' fragment"))?,
            None => String::new(),
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
            &fragment,
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
        pass.visible = match o.get("visible").filter(|v| !v.is_null()) {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(v) => bail!(
                "pass '{name}' visible needs true or false, got {}",
                py_repr(v)
            ),
        };
        let layers = list_of(o, "layers").map_err(|e| anyhow!("pass '{name}': {e}"))?;
        pass.layers = layers
            .iter()
            .enumerate()
            .map(|(k, dd)| Layer::from_dict(&layer_at(&name, k), dd))
            .collect::<Result<Vec<_>>>()?;
        // A filter is written apart from any scene, and a layer belongs to one.
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
        self.dict(&|l| l.text.clone())
    }

    /// The pass as a scene holds it, with `words` of each text layer in its
    /// `text`.
    fn dict(&self, words: &dyn Fn(&TextLayer) -> String) -> Value {
        let mut out = Map::new();
        out.insert("name".to_string(), Value::from(self.name.clone()));
        if !self.visible {
            out.insert("visible".to_string(), Value::from(false));
        }
        if self.has_fragment() || self.layers.is_empty() {
            out.insert("fragment".to_string(), Value::from(self.fragment.clone()));
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
            let layers: Vec<Value> = self
                .layers
                .iter()
                .map(|l| match l {
                    Layer::Text(t) => t.to_dict(&words(t)),
                    l => l.to_dict(""),
                })
                .collect();
            out.insert("layers".to_string(), Value::from(layers));
        }
        Value::Object(out)
    }

    /// `hw_linear` is the target GPU's Renderer.data_texture_linear. False emits
    /// the array readers that hold on any GPU. `assets` are the scene's assets.
    /// A pass of the scene's own declares the scene's assets. A filter pass
    /// declares only its filter's, each under the name the filter wrote.
    /// `texts` names the scene's text layers.
    pub fn glsl(
        &self,
        scene_params: Option<&ParamMap>,
        common: Option<&str>,
        assets: &[Asset],
        texts: &[&str],
        hw_linear: bool,
    ) -> String {
        // A hidden pass, as Scene::drawn_passes() leaves it, runs ghee's copy
        // in its code, which reads neither the common block nor a text.
        if !self.visible {
            let l = self.preamble_reading(scene_params, assets, texts, hw_linear, |_| false);
            return format!("{}\n\n{}", l.join("\n"), self.fragment);
        }
        let common_code = common.unwrap_or_default();
        let l = self.preamble_reading(scene_params, assets, texts, hw_linear, |t| {
            calls_cover(common_code, t) || calls_cover(&self.fragment, t)
        });
        // An author writes a filter on its own, and ghee appends it to any
        // scene. The scene's common block never reaches it, so a filter name
        // never collides with a scene name.
        let body = match common {
            Some(c) if !c.is_empty() && self.filter.is_empty() => {
                format!("{c}\n\n{}", self.fragment)
            }
            _ => self.fragment.clone(),
        };
        format!("{}\n\n{body}", l.join("\n"))
    }

    /// The preamble, with the `<name>_size` of each text layer of `texts`,
    /// and the coverage reader of each for which `reads` holds.
    fn preamble_reading(
        &self,
        scene_params: Option<&ParamMap>,
        assets: &[Asset],
        texts: &[&str],
        hw_linear: bool,
        reads: impl Fn(&str) -> bool,
    ) -> Vec<String> {
        let mut l: Vec<String> = PRECISION.iter().map(|l| l.to_string()).collect();
        l.extend(
            [
                "varying vec2 uv;",
                "uniform vec3 iResolution;",
                "uniform float iTime;",
                "uniform float iTimeDelta;",
                "uniform int iFrame;",
                // VC4 shears gl_FragCoord.xy by about 1/3 px per row when a
                // pass samples its own output. The interpolated varying stays
                // within 0.04 px of the pixel centre. This define maps
                // gl_FragCoord onto the varying, so scene code that reads
                // gl_FragCoord.xy/iResolution.xy samples feedback without
                // drift. A function reads the varying, so a local `uv` in the
                // scene's own code leaves gl_FragCoord alone.
                "vec4 ghee_fragcoord() { return vec4(uv * iResolution.xy, 0.0, 1.0); }",
                "#define gl_FragCoord ghee_fragcoord()",
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
        // A layer belongs to the scene, so a filter pass never reads one.
        if self.filter.is_empty() {
            for t in texts {
                l.push(format!("uniform vec2 {t}_size;"));
                if reads(t) {
                    l.extend(crate::text::reader_lines(t));
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
                l.push(format!("#define {local}_tex {}_tex", a.name));
                l.push(format!("#define {local}_size {}_size", a.name));
            }
        }
        l
    }
}

/// True when `glsl` names the coverage reader of text layer `name`. A reader
/// is long to compile on VC4, so a shader holds the reader of such a layer
/// alone.
fn calls_cover(glsl: &str, name: &str) -> bool {
    glsl.contains(&format!("{name}_cover"))
}

/// The code of a hidden pass after the first: a copy of the pass before it.
const COPY: &str = "void main(){ gl_FragColor = texture2D(iChannel0, uv); }";

/// The code of a hidden first pass.
const BLACK: &str = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";

// mediump cannot hold a texel coord like (col+0.5)/w. A feedback pass that
// resamples its own output then drifts a fraction of a pixel per frame. highp
// keeps a ping-pong buffer stable, and VC4 supports it.
pub(crate) const PRECISION: [&str; 6] = [
    "#version 100",
    "#ifdef GL_FRAGMENT_PRECISION_HIGH",
    "precision highp float;",
    "#else",
    "precision mediump float;",
    "#endif",
];

/// The error for a param that a host names as `wire`, `pass.name` or `name`,
/// and that no param of the scene answers to. `wires` are the names that do.
/// A /set and Python's set_param give the same text.
pub fn no_param<'a>(wire: &str, wires: impl IntoIterator<Item = &'a str>) -> String {
    format!(
        "param '{wire}' is not a param this scene declares ({})",
        listed(wires)
    )
}

/// The error for a layer `name` that a host names and the scene lacks.
/// `layers` are the names of the scene's layers. A /set and Python's
/// set_layer give the same text.
pub fn no_layer<'a>(name: &str, layers: impl IntoIterator<Item = &'a str>) -> String {
    format!(
        "layer '{name}' is not a layer this scene declares ({})",
        listed(layers)
    )
}

/// `names` joined by commas, or "none".
fn listed<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let all: Vec<&str> = names.into_iter().collect();
    match all.is_empty() {
        true => "none".to_string(),
        false => all.join(", "),
    }
}

/// The error for `value`, which the live image `what` takes when it names an
/// image asset of the scene's own.
pub(crate) fn not_an_image(what: &str, value: &Value) -> String {
    format!(
        "{what} needs an image asset of the scene or a .png or .jpg file, got {}",
        py_repr(value)
    )
}

/// A new look of a layer of the scene, as Scene::look() finds it.
pub struct Look {
    /// The layer as the scene holds it, and its words.
    pub old: Layer,
    pub old_words: String,
    /// The layer in its new look, and its words. With `rebuild` false, it
    /// holds the fonts and the file of `old`.
    pub new: Layer,
    pub new_words: String,
    /// The new look as a scene declares it, with the effects that it leaves
    /// out bypassed.
    pub look: Value,
    /// True when a field that changes with a rebuild differs.
    pub rebuild: bool,
}

/// The kind of `layer`: text, image, or its shape for a shape layer.
fn kind_of(layer: &Layer) -> &'static str {
    match layer {
        Layer::Text(_) => "text",
        Layer::Image(_) => "image",
        Layer::Shape(l) => match l.shape {
            Shape::Rect { .. } => "rect",
            Shape::Line { .. } => "line",
        },
    }
}

/// `look`, a layer of the kind of `old` as a scene declares it, with each
/// effect of `old` that it leaves out, and a rect's border, bypassed with
/// its values.
fn keep_bypassed(old: &Layer, look: &Value) -> Value {
    let mut look = look.clone();
    let (Some(o), Value::Object(was)) = (look.as_object_mut(), old.to_dict("")) else {
        return look;
    };
    for (key, mut v) in was {
        let effect = v.is_object() && old.live(Some(&key), "visible").is_ok();
        if effect && o.get(&key).is_none_or(Value::is_null) {
            v["visible"] = Value::from(false);
            o.insert(key, v);
        }
    }
    look
}

/// The keys of `d`, layer `layer` as a scene writes it, that change with a
/// rebuild: every key but the name, the words and the live fields of
/// `layer`, its effects' among them. A rect's border is live whole.
fn build_of(layer: &Layer, d: &Value) -> Value {
    let mut d = d.clone();
    if let Some(o) = d.as_object_mut() {
        o.retain(|k, v| {
            if k == "name" || k == "text" || layer.live(None, k).is_ok() {
                return false;
            }
            match v {
                Value::Object(fields) if layer.live(Some(k), "visible").is_ok() => {
                    fields.retain(|f, _| layer.live(Some(k), f).is_err());
                    !(fields.is_empty() && layer.as_shape().is_some())
                }
                _ => true,
            }
        });
    }
    d
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub passes: Vec<Pass>,
    pub version: Value,
    /// The named filters to append after the scene's own passes, in order. The
    /// last one reaches the display. The filters module resolves a name.
    pub filters: Vec<String>,
    /// The names among `filters` whose pass starts hidden. A scene file marks
    /// one as `{"name": <name>, "visible": false}` in `filters`. Appending the
    /// filters puts the eye on each filter's pass and empties this list.
    pub hidden_filters: Vec<String>,
    /// The scene params. Every pass preamble declares them, and one set reaches
    /// every pass at once. They take the shape of a pass param. They hold the
    /// string param of each text layer too, which ghee declares.
    pub params: ParamMap,
    /// The GLSL every scene pass shares, such as a function or a constant. It
    /// follows the preamble. None is no common block. A filter pass never sees
    /// it.
    pub common: Option<String>,
    /// Image assets: set up once by the host, available to every pass as a
    /// sampler uniform `<name>_tex` plus a `<name>_size` vec2.
    pub assets: Vec<Asset>,
    /// Where the scene's files resolve, as its load found them. None for a
    /// scene whose files never loaded.
    pub root: Option<Root>,
}

/// Where a loaded scene's files resolve: its directory, as far as `reach`
/// allows, then the directories of `$GHEE_ASSETS`, and the directory of the
/// system's emoji font.
#[derive(Clone, Debug, PartialEq)]
pub struct Root {
    pub dir: String,
    pub reach: Reach,
    more: Vec<String>,
    emoji_dir: String,
}

impl Root {
    /// The bytes of `file` and the path that held them, or why no path did,
    /// with the file named as `what`. A relative path resolves in `dir`, then
    /// in each of `more`.
    fn read(&self, file: &str, what: &str) -> std::result::Result<(Vec<u8>, String), String> {
        let roots = match file.starts_with('/') {
            true => &[][..],
            false => &self.more[..],
        };
        let first = normpath(&join(&self.dir, file));
        let found = std::iter::once(first.clone())
            .chain(roots.iter().map(|r| normpath(&join(r, file))))
            .find(|p| std::path::Path::new(p).is_file());
        match found {
            None => {
                // A font file that shows up later marks the scene stale.
                crate::sources::record(&first);
                let elsewhere = match (file.starts_with('/'), self.more.is_empty()) {
                    (true, _) => String::new(),
                    (false, true) => ", and $GHEE_ASSETS names no directory".to_string(),
                    (false, false) => format!(", nor in $GHEE_ASSETS ({})", self.more.join(", ")),
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
    }

    /// True when `file` lies outside the directory that the reach allows.
    fn outside(&self, file: &str) -> bool {
        self.reach == Reach::Inside && !inside(&normpath(&join(&self.dir, file)), &self.dir)
    }

    /// The pixels of the image file `file` of layer `layer`, decoded.
    pub fn image(&self, layer: &str, file: &str) -> Result<crate::assets::Image> {
        if self.outside(file) {
            bail!(
                "layer '{layer}' image {} is outside the scene directory",
                py_str_repr(file)
            );
        }
        let what = format!("layer '{layer}' image {}", py_str_repr(file));
        let (bytes, path) = self.read(file, &what).map_err(|e| anyhow!("{e}"))?;
        let decoded = match file.to_ascii_lowercase().ends_with(".png") {
            true => crate::assets::decode_png(&bytes),
            false => crate::assets::decode_jpeg(&bytes),
        };
        decoded.map_err(|e| anyhow!("{what} ({path}): {e}"))
    }
}

impl Scene {
    pub fn new(
        mut passes: Vec<Pass>,
        version: Value,
        filters: Vec<String>,
        mut params: ParamMap,
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
        if !passes.is_empty() && passes.iter().all(|p| !p.visible) {
            bail!("every pass is hidden, and a scene shows one pass at least");
        }
        {
            let mut seen: Vec<&str> = Vec::new();
            for a in &assets {
                if seen.contains(&a.name.as_str()) {
                    bail!("asset '{}' is declared twice", a.name);
                }
                seen.push(&a.name);
            }
        }
        // ghee declares the string param of each text layer, which holds its
        // words, after the scene's own params. A param that ghee declared
        // before keeps its words, and a param of the scene's own under a
        // layer's name stays for name_clash() to refuse.
        let held: Vec<Param> = params.values().filter(|p| p.of_text).cloned().collect();
        params.retain(|p| !p.of_text);
        for l in passes
            .iter_mut()
            .flat_map(|p| &mut p.layers)
            .filter_map(Layer::as_text_mut)
        {
            if params.contains_key(&l.name) {
                continue;
            }
            if let Some(Param {
                value: ParamValue::Str(w),
                ..
            }) = held.iter().find(|p| p.name == l.name)
            {
                l.text = w.clone();
            }
            params.insert(l.name.clone(), Param::words(&l.name, &l.text));
        }
        let scene = Scene {
            passes,
            version,
            filters,
            hidden_filters: Vec::new(),
            params,
            common,
            assets,
            root: None,
        };
        scene.check_layers()?;
        Ok(scene)
    }

    /// The error for the first name that the scene cannot take, as
    /// name_clash() finds it, or for a layer's font or image that names no
    /// asset of the right kind.
    fn check_layers(&self) -> Result<()> {
        let scene = self;
        if let Some(e) = scene.name_clash() {
            bail!("{e}");
        }
        for l in scene.text_layers() {
            if let Some(e) = l.font_family.as_deref().and_then(|f| scene.font_fault(f)) {
                bail!("layer '{}' {e}", l.name);
            }
        }
        for l in scene.layers().filter_map(Layer::as_image) {
            if let Some(e) = scene.image_fault(&l.image) {
                bail!("layer '{}' {e}", l.name);
            }
        }
        Ok(())
    }

    /// Why an image layer of the scene cannot take `image`, or None when it
    /// can: a .png or .jpg file, or an image asset of the scene's own.
    fn image_fault(&self, image: &str) -> Option<String> {
        if is_image_file(image) {
            return None;
        }
        let own = |a: &&Asset| a.name == image && a.from_filter.is_empty();
        match self.assets.iter().find(own) {
            Some(a) if a.is_font() => Some(format!(
                "image '{image}' is the font {}, an image is a .png or .jpg file",
                py_str_repr(&a.file)
            )),
            Some(_) => None,
            None => Some(format!(
                "image '{image}' is no asset of the scene and no .png or .jpg file"
            )),
        }
    }

    /// Put `layer` in the place of the layer of its name, which it differs
    /// from in live fields alone, through the checks of a live set: an image
    /// layer's image. The words param of a text layer takes its words.
    pub(crate) fn swap_live(&mut self, layer: Layer) -> Result<()> {
        let name = layer.name().to_string();
        let Some((at, index)) = self.place_of(&name) else {
            bail!("the scene has no layer '{name}'");
        };
        if let Some(e) = layer.as_image().and_then(|l| self.image_fault(&l.image)) {
            bail!("layer '{name}' {e}");
        }
        if let (Some(l), Some(p)) = (layer.as_text(), self.params.get_mut(&name)) {
            if p.of_text {
                p.value = ParamValue::Str(l.text.clone());
            }
        }
        self.passes[at].layers[index] = layer;
        Ok(())
    }

    /// Put `layer` in the place of layer `i` of pass `p`, which it differs
    /// from in live fields alone and which keeps its image, as swap_live()
    /// does, with `param` the place of the scene param of its name. Returns
    /// the layer that it replaced, and the words of `layer` when they differ
    /// from the words that the layer held, as look() tells them apart.
    pub(crate) fn swap_live_at(
        &mut self,
        p: usize,
        i: usize,
        param: Option<usize>,
        layer: Layer,
    ) -> (Layer, Option<String>) {
        let param = param.and_then(|k| self.params.at_mut(k));
        let old = std::mem::replace(&mut self.passes[p].layers[i], layer);
        let Some(t) = self.passes[p].layers[i].as_text() else {
            return (old, None);
        };
        let held = match param.as_ref().map(|p| &p.value) {
            Some(ParamValue::Str(w)) => w.as_str(),
            _ => old.as_text().map_or("", |o| o.text.as_str()),
        };
        let words = (held != t.text).then(|| t.text.clone());
        if let Some(param) = param.filter(|p| p.of_text) {
            param.value = ParamValue::Str(t.text.clone());
        }
        (old, words)
    }

    /// Why a text layer of the scene cannot take `font` as its
    /// font_family, or None when it can: a font file, a font asset of the
    /// scene, or the built-in pixel font.
    fn font_fault(&self, font: &str) -> Option<String> {
        if crate::text::is_font_file(font) {
            return None;
        }
        match self.assets.iter().find(|a| a.name == font) {
            Some(a) if a.is_font() => None,
            Some(a) => Some(format!(
                "font_family '{font}' is the image {}, a font is a .ttf or .otf file",
                py_str_repr(&a.file)
            )),
            None if font == crate::text::PIXEL => None,
            None => Some(format!(
                "font_family '{font}' is no asset of the scene, no .ttf or .otf file and not \
                 '{}', the built-in pixel font",
                crate::text::PIXEL
            )),
        }
    }

    /// The size in pixels of the block that `words` lay out in, as a text
    /// layer of the scene in a pass of `pass` pixels would lay them out in
    /// `font_family`, the built-in font when None: at `font_size` pixels per
    /// em, or at the size that fits the block to the pass, centred. A font
    /// file resolves as a layer's does.
    pub fn measure(
        &self,
        pass: [f64; 2],
        words: &str,
        font_size: Option<f64>,
        font_family: Option<&str>,
    ) -> Result<[f64; 2]> {
        if let Some(px) = font_size.filter(|px| !(px.is_finite() && *px > 0.0)) {
            bail!(
                "font_size needs a number of pixels above 0, got {}",
                py_float(px)
            );
        }
        if let Some(e) = font_family.and_then(|f| self.font_fault(f)) {
            bail!("{e}");
        }
        let mut d = Map::new();
        d.insert("name".to_string(), Value::from(MEASURED));
        d.insert("text".to_string(), Value::from(words));
        if let Some(px) = font_size {
            d.insert("font_size".to_string(), Value::from(px));
        }
        if let Some(f) = font_family {
            d.insert("font_family".to_string(), Value::from(f));
        }
        let mut layer = Layer::from_dict(MEASURED, &Value::Object(d))?;
        // The layer's name opens each error of its fonts, and the caller
        // named no layer.
        let own = format!("layer '{MEASURED}' ");
        self.load_layer(&mut layer)
            .map_err(|e| anyhow!("{}", e.to_string().replacen(&own, "", 1)))?;
        let l = layer.as_text().expect("a layer of words is a text layer");
        if let Some(Err(why)) = l.data.iter().flatten().find(|f| f.is_err()) {
            bail!("{}", why.replacen(&own, "", 1));
        }
        let laid = l.place(words)?;
        let Some(built) = laid.built else {
            let why = laid.faults.iter().find_map(|f| match f {
                crate::text::Fault::Layout(why) => Some(why.as_str()),
                _ => None,
            });
            bail!("{}", why.unwrap_or("the words lay out in no texture"));
        };
        let px = l.font_px(pass, built.size);
        Ok([built.size[0] * px, built.size[1] * px])
    }

    /// Put `layer` into pass `pass` at `index` among its layers, through the
    /// checks of a scene that declares it there, and declare the words param
    /// of a text layer. A layer that the scene refuses leaves it as it was.
    pub fn insert_layer(&mut self, pass: &str, index: usize, layer: Layer) -> Result<()> {
        let Some(at) = self.passes.iter().position(|p| p.name == pass) else {
            bail!("the scene has no pass '{pass}'");
        };
        let p = &self.passes[at];
        if !p.filter.is_empty() {
            bail!(
                "pass '{pass}' is filter '{}', and a filter pass takes no layers",
                p.filter
            );
        }
        let n = p.layers.len();
        if index > n {
            bail!("pass '{pass}' holds {n} layers, so a layer goes in at 0 to {n}, got {index}");
        }
        let words = layer.as_text().map(|l| Param::words(&l.name, &l.text));
        let name = layer.name().to_string();
        self.passes[at].layers.insert(index, layer);
        let declared = match words {
            Some(w) if !self.params.contains_key(&name) => {
                self.params.insert(name.clone(), w);
                true
            }
            _ => false,
        };
        if let Err(e) = self.check_layers() {
            self.passes[at].layers.remove(index);
            if declared {
                self.params.retain(|p| p.name != name);
            }
            return Err(e);
        }
        Ok(())
    }

    /// Put `layer` in the place of the layer of its name, through the checks
    /// of a scene that declares it there. The words param of a text layer
    /// takes its words, and a layer of another kind leaves no words param. A
    /// layer that the scene refuses leaves it as it was.
    pub fn replace_layer(&mut self, layer: Layer) -> Result<()> {
        let name = layer.name().to_string();
        let Some((at, index)) = self.place_of(&name) else {
            bail!("the scene has no layer '{name}'");
        };
        let held = self.params.get(&name).filter(|p| p.of_text).cloned();
        let words = layer.as_text().map(|l| Param::words(&name, &l.text));
        let old = std::mem::replace(&mut self.passes[at].layers[index], layer);
        match words {
            Some(w) if held.is_some() || !self.params.contains_key(&name) => {
                self.params.insert(name.clone(), w);
            }
            Some(_) => {}
            None => self.params.retain(|p| !(p.of_text && p.name == name)),
        }
        if let Err(e) = self.check_layers() {
            self.passes[at].layers[index] = old;
            self.params.retain(|p| !(p.of_text && p.name == name));
            if let Some(p) = held {
                self.params.insert(name, p);
            }
            return Err(e);
        }
        Ok(())
    }

    /// The new look `look` of the layer of its name, a layer as a scene
    /// declares it: every key that `look` leaves out takes its default, and
    /// an effect of the layer that `look` leaves out stays, bypassed, with
    /// its values, and so does a rect's border. None when the layer has that
    /// look already. A look that no layer takes, and a name that names no
    /// layer of the scene, fail.
    pub fn look(&self, look: &Value) -> Result<Option<Look>> {
        let new = Layer::from_dict("the new layer", look)?;
        let name = new.name().to_string();
        let Some(old) = self.layer(&name) else {
            bail!("the scene has no layer '{name}'");
        };
        let look = match kind_of(old) == kind_of(&new) {
            true => keep_bypassed(old, look),
            false => look.clone(),
        };
        let mut new = Layer::from_dict("the new layer", &look)?;
        let old_words = match self.words(&name) {
            Some(w) => w.to_string(),
            None => old.as_text().map_or(String::new(), |l| l.text.clone()),
        };
        let new_words = new.as_text().map_or(String::new(), |l| l.text.clone());
        let (was, will) = (old.to_dict(&old_words), new.to_dict(&new_words));
        if was == will {
            return Ok(None);
        }
        let rebuild = build_of(old, &was) != build_of(old, &will);
        // The same build: the fonts and the file of the layer stay.
        if !rebuild {
            match (&mut new, old) {
                (Layer::Text(n), Layer::Text(o)) => {
                    n.data.clone_from(&o.data);
                    n.emoji.clone_from(&o.emoji);
                }
                (Layer::Image(n), Layer::Image(o)) if n.image == o.image => {
                    n.data.clone_from(&o.data);
                }
                _ => {}
            }
        }
        Ok(Some(Look {
            old: old.clone(),
            old_words,
            new,
            new_words,
            look,
            rebuild,
        }))
    }

    /// Give the layer of the name of `look` that look, as look() finds it:
    /// through the checks of a scene that declares it there when a field
    /// that changes with a rebuild differs, and through the checks of a
    /// live set otherwise. False when the layer has that look already. A
    /// look that the scene refuses leaves the scene as it was.
    pub fn update_layer(&mut self, look: &Value) -> Result<bool> {
        let Some(found) = self.look(look)? else {
            return Ok(false);
        };
        match found.rebuild {
            true => self.replace_layer(found.new)?,
            false => self.swap_live(found.new)?,
        }
        Ok(true)
    }

    /// The pass index and the index among its layers of layer `name`.
    pub(crate) fn place_of(&self, name: &str) -> Option<(usize, usize)> {
        self.passes.iter().enumerate().find_map(|(p, pass)| {
            pass.layers
                .iter()
                .position(|l| l.name() == name)
                .map(|i| (p, i))
        })
    }

    /// Take layer `name` out of the scene, with the words param of a text
    /// layer.
    pub(crate) fn remove_layer(&mut self, name: &str) {
        let mut text = false;
        for p in &mut self.passes {
            p.layers.retain(|l| {
                text |= l.name() == name && l.as_text().is_some();
                l.name() != name
            });
        }
        if text {
            self.params.retain(|p| !(p.of_text && p.name == name));
        }
    }

    /// Hand `layer`, which joins the scene, its fonts or its image file as
    /// the scene's load did for its own layers. A scene whose files never
    /// loaded takes a layer that names no file.
    pub fn load_layer(&self, layer: &mut Layer) -> Result<()> {
        let emoji = match self.text_layers().find_map(|l| l.emoji.clone()) {
            Some(e) => Some(e),
            None => self
                .root
                .as_ref()
                .and_then(|r| crate::emoji::find(&r.dir, &r.more, &r.emoji_dir)),
        };
        let none = Root {
            dir: String::new(),
            reach: Reach::Inside,
            more: Vec::new(),
            emoji_dir: String::new(),
        };
        let root = self.root.as_ref().unwrap_or(&none);
        if let (Layer::Image(l), None) = (&*layer, &self.root) {
            if l.is_file() {
                bail!(
                    "layer '{}': '{}' is a file, and the scene was built without a directory",
                    l.name,
                    l.image
                );
            }
        }
        load_layer(&self.assets, layer, root, &emoji)
    }

    /// Every layer of the scene, in the order of its passes.
    pub fn layers(&self) -> impl Iterator<Item = &Layer> {
        self.passes.iter().flat_map(|p| &p.layers)
    }

    /// Every text layer of the scene, in the order of its passes.
    pub fn text_layers(&self) -> impl Iterator<Item = &TextLayer> {
        self.layers().filter_map(Layer::as_text)
    }

    /// The layer named `name`.
    pub fn layer(&self, name: &str) -> Option<&Layer> {
        self.layers().find(|l| l.name() == name)
    }

    /// Set the live field at `at` to `value`, written as the scene format
    /// writes the key, as Layer::set_live() does. A live image is an image
    /// asset of the scene's own, or a .png or .jpg file.
    pub fn set_layer(&mut self, at: &Address, value: &Value) -> Result<()> {
        let Some(layer) = self.layer(at.layer) else {
            bail!("the scene has no layer '{}'", at.layer);
        };
        let live = layer.live(at.effect, at.field)?;
        if live.form == Form::Image && !value.as_str().is_some_and(is_image_file) {
            let image = |a: &Asset| {
                a.from_filter.is_empty() && !a.is_font() && value.as_str() == Some(a.name.as_str())
            };
            if !self.assets.iter().any(image) {
                bail!("{}", not_an_image(&live.what, value));
            }
        }
        self.passes
            .iter_mut()
            .flat_map(|p| &mut p.layers)
            .find(|l| l.name() == at.layer)
            .expect("the layer was just found")
            .set_live(at.effect, at.field, value)
    }

    /// The text layer named `name`.
    pub fn text_layer(&self, name: &str) -> Option<&TextLayer> {
        self.text_layers().find(|l| l.name == name)
    }

    /// The names of the scene's text layers, as Pass::glsl() takes them.
    pub fn text_names(&self) -> Vec<&str> {
        self.text_layers().map(|l| l.name.as_str()).collect()
    }

    /// The words of text layer `name` as they stand: its string param's.
    pub fn words(&self, name: &str) -> Option<&str> {
        match self.params.get(name).map(|p| &p.value) {
            Some(ParamValue::Str(w)) => Some(w),
            _ => None,
        }
    }

    /// The error for the first declared name that the scene cannot take: a
    /// name that breaks the name rule, a built-in name, a name that two
    /// declarations share, or one GLSL name that two declarations make in one
    /// pass.
    ///
    /// The passes, the layers, the scene's assets and the scene params share
    /// one set of names. A pass param or an array takes a name unique in
    /// its pass and outside that set, so two passes may each declare a `gain`.
    ///
    /// A scene param reaches every pass, an image asset and a text layer every
    /// pass of the scene's own, a filter's asset the passes of its filter, and
    /// a pass param or an array its own pass.
    pub(crate) fn name_clash(&self) -> Option<String> {
        // Each declaration as its kind, its name, where it is declared, and
        // whether the built-in names are closed to it. A pass name never
        // reaches GLSL, and a layer's reaches it with a suffix, so both may
        // take a built-in name.
        let mut kinds: Vec<(&str, &str, String, bool)> = Vec::new();
        for p in &self.passes {
            kinds.push(("pass", &p.name, String::new(), false));
        }
        for l in self.layers() {
            kinds.push(("layer", l.name(), String::new(), false));
        }
        for p in self.params.values().filter(|p| !p.of_text) {
            kinds.push(("scene param", &p.name, String::new(), true));
        }
        for a in &self.assets {
            match a.from_filter.as_str() {
                "" => kinds.push(("asset", &a.name, String::new(), true)),
                f => kinds.push(("asset", a.local_name(), format!(" of filter '{f}'"), true)),
            }
        }
        for pass in &self.passes {
            let at = format!(" in pass '{}'", pass.name);
            for p in pass.params.values() {
                kinds.push(("param", &p.name, at.clone(), true));
            }
            for a in &pass.arrays {
                kinds.push(("array", &a.name, at.clone(), true));
            }
        }
        for (kind, name, at, closed) in &kinds {
            if let Some(why) = name_fault(name) {
                return Some(format!("{kind} name {}{at} {why}", py_str_repr(name)));
            }
            if *closed && built_in(name) {
                return Some(format!("{kind} '{name}'{at}: '{name}' is built in"));
            }
        }
        if let Some(e) = self.shared_name() {
            return Some(e);
        }
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
                for l in self.text_layers() {
                    for n in crate::text::names(&l.name) {
                        names.push((n, format!("layer '{}'", l.name)));
                    }
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

    /// The names that the passes, the layers, the scene's assets and the scene
    /// params share, each with its declaration, as "pass 'main'".
    fn scene_wide_names(&self) -> Vec<(&str, String)> {
        let mut wide: Vec<(&str, String)> = Vec::new();
        for p in &self.passes {
            wide.push((&p.name, format!("pass '{}'", p.name)));
        }
        for l in self.layers() {
            wide.push((l.name(), format!("layer '{}'", l.name())));
        }
        for a in self.assets.iter().filter(|a| a.from_filter.is_empty()) {
            wide.push((&a.name, format!("asset '{}'", a.name)));
        }
        for p in self.params.values().filter(|p| !p.of_text) {
            wide.push((&p.name, format!("scene param '{}'", p.name)));
        }
        wide
    }

    /// Every name that a pass added to the scene may not take: a name of the
    /// scene-wide set, or of any pass's params and arrays.
    pub fn taken_names(&self) -> Vec<&str> {
        let mut taken: Vec<&str> = self
            .scene_wide_names()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for pass in &self.passes {
            taken.extend(pass.params.keys().map(String::as_str));
            taken.extend(pass.arrays.iter().map(|a| a.name.as_str()));
        }
        taken
    }

    /// The error for two declarations that share a name: two of the
    /// scene-wide set, a pass param or an array and one of that set, or two
    /// in one pass.
    fn shared_name(&self) -> Option<String> {
        let wide = self.scene_wide_names();
        if let Some(e) = first_shared(&wide, "") {
            return Some(e);
        }
        for pass in &self.passes {
            let at = format!(" in pass '{}'", pass.name);
            let mut own: Vec<(&str, String)> = Vec::new();
            for p in pass.params.values() {
                own.push((&p.name, format!("param '{}'", p.name)));
            }
            for a in &pass.arrays {
                own.push((&a.name, format!("array '{}'", a.name)));
            }
            for (n, owner) in &own {
                if let Some((_, w)) = wide.iter().find(|(m, _)| m == n) {
                    return Some(format!("{w} and {owner}{at} share a name"));
                }
            }
            if let Some(e) = first_shared(&own, &at) {
                return Some(e);
            }
        }
        None
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

    /// The passes that the engine draws, as the scene's own when every pass
    /// is visible. A hidden pass draws a copy of the pass before it, or black
    /// as the first pass, and its layers draw nothing. The passes after the
    /// last visible one stay out, so that pass holds the frame, and a channel
    /// that reads one of them reads that pass, which their copies would show.
    pub(crate) fn drawn_passes(&self) -> Cow<'_, [Pass]> {
        if self.passes.iter().all(|p| p.visible) {
            return Cow::Borrowed(&self.passes);
        }
        let last = self
            .passes
            .iter()
            .rposition(|p| p.visible)
            .unwrap_or(self.passes.len() - 1);
        let shown = &self.passes[..=last];
        let gone: Vec<&str> = self.passes[last + 1..]
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        let drawn = shown
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut d = p.clone();
                if !p.visible {
                    d.layers.clear();
                    (d.fragment, d.channels) = match i {
                        0 => (BLACK.to_string(), Vec::new()),
                        _ => (
                            COPY.to_string(),
                            vec![Channel::new(Some(&shown[i - 1].name))],
                        ),
                    };
                }
                for c in &mut d.channels {
                    if c.buffer.as_deref().is_some_and(|b| gone.contains(&b)) {
                        c.buffer = Some(shown[last].name.clone());
                    }
                }
                d
            })
            .collect();
        Cow::Owned(drawn)
    }

    /// `path` names the scene in an error message. A file of GLSL resolves
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
                keys_on(&SCENE_KEYS, &crate::layer::layer_keys(), "a layer"),
                keys_on(&SCENE_KEYS, &ASSET_KEYS, "an asset"),
            ]
            .concat();
            known_keys(o, "the scene", &SCENE_KEYS, &elsewhere)?;
            let mut filters = Vec::new();
            let mut hidden_filters = Vec::new();
            for f in list_of(o, "filters")? {
                let Some((name, visible)) = filter_entry(f) else {
                    bail!(
                        "'filters' takes names and objects with a 'name' and a 'visible', got {}",
                        py_repr(f)
                    );
                };
                if !visible && !hidden_filters.contains(&name) {
                    hidden_filters.push(name.clone());
                }
                filters.push(name);
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
            let common = common_block(o, files)?;
            let mut scene = Scene::new(passes, version, filters, params, common, assets)?;
            scene.hidden_filters = hidden_filters;
            Ok(scene)
        })()
        .map_err(|e| anyhow!("{src}: {e}"))
    }

    pub fn to_dict(&self) -> Value {
        let mut out = Map::new();
        out.insert("version".to_string(), self.version.clone());
        // Each layer holds its words as they stand, which an edit of its
        // param changes.
        let words = |l: &TextLayer| self.words(&l.name).unwrap_or(&l.text).to_string();
        out.insert(
            "passes".to_string(),
            Value::from(
                self.passes
                    .iter()
                    .map(|p| p.dict(&words))
                    .collect::<Vec<_>>(),
            ),
        );
        // "" is a common block the console can fill.
        if let Some(c) = &self.common {
            out.insert("common".to_string(), Value::from(c.clone()));
        }
        if !self.filters.is_empty() {
            let entry = |n: &String| match self.hidden_filters.contains(n) {
                true => {
                    let mut m = Map::new();
                    m.insert("name".to_string(), Value::from(n.as_str()));
                    m.insert("visible".to_string(), Value::from(false));
                    Value::Object(m)
                }
                false => Value::from(n.as_str()),
            };
            out.insert(
                "filters".to_string(),
                self.filters.iter().map(entry).collect(),
            );
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
        let text = fs::read_to_string(path).map_err(|e| anyhow!("{path}: {}", strerror(&e)))?;
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

    /// Read and decode every declared asset file, and hand each layer its
    /// font or its image file, with the reach of a scene that its user runs,
    /// as load_assets_in() says.
    pub fn load_assets(&mut self, base_dir: &str) -> Result<()> {
        self.load_assets_in(base_dir, Reach::Any)
    }

    /// Read and decode every declared asset file, hand each text layer its
    /// font and the emoji font, and decode the file of each image layer that
    /// names one.
    ///
    /// An absolute path reads as it stands. Any other path resolves in
    /// `base_dir`, then through `$GHEE_ASSETS`, colon-separated directories, in
    /// order, and the error names the first path and the directories of
    /// `$GHEE_ASSETS`. Reach::Inside refuses a path outside `base_dir`. A
    /// filter's asset is the filter's to load. The emoji font is the one that
    /// crate::emoji finds from `base_dir`, under either reach.
    pub fn load_assets_in(&mut self, base_dir: &str, reach: Reach) -> Result<()> {
        self.load_assets_from(base_dir, reach, &asset_dirs(), crate::emoji::SYSTEM_DIR)
    }

    /// load_assets_in() with `more` in place of the directories of
    /// `$GHEE_ASSETS`, and `emoji_dir` in place of the system's directory of
    /// the emoji font.
    fn load_assets_from(
        &mut self,
        base_dir: &str,
        reach: Reach,
        more: &[String],
        emoji_dir: &str,
    ) -> Result<()> {
        let root = Root {
            dir: normpath(&abspath(base_dir)),
            reach,
            more: more.to_vec(),
            emoji_dir: emoji_dir.to_string(),
        };
        self.root = Some(root.clone());
        let emoji = match self.text_layers().next() {
            Some(_) => crate::emoji::find(&root.dir, more, emoji_dir),
            None => None,
        };
        for a in &mut self.assets {
            if !a.from_filter.is_empty() {
                continue; // the filter decoded it
            }
            if root.outside(&a.file) {
                bail!(
                    "asset '{}' file {} is outside the scene directory",
                    a.name,
                    py_str_repr(&a.file)
                );
            }
            match root.read(&a.file, &format!("asset '{}'", a.name)) {
                Ok((bytes, path)) => a.decode(&bytes, &path)?,
                // A font that fails to load leaves the scene running, and its
                // layers draw error boxes.
                Err(why) if a.is_font() => a.font = Some(Err(why)),
                Err(why) => bail!("{why}"),
            }
        }
        let assets = &self.assets;
        for layer in self.passes.iter_mut().flat_map(|p| &mut p.layers) {
            load_layer(assets, layer, &root, &emoji)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_dict())?)?;
        Ok(())
    }
}

/// Hand `layer` its fonts and the emoji font `emoji`, from the font assets
/// `assets` or a file, or the pixels of its image file, its files read
/// through `root`.
fn load_layer(
    assets: &[Asset],
    layer: &mut Layer,
    root: &Root,
    emoji: &Option<std::sync::Arc<crate::emoji::EmojiFont>>,
) -> Result<()> {
    let l = match layer {
        Layer::Text(l) => l,
        Layer::Image(l) if l.is_file() => {
            l.data = Some(root.image(&l.name, &l.image)?);
            return Ok(());
        }
        Layer::Image(_) | Layer::Shape(_) => return Ok(()),
    };
    l.emoji = emoji.clone();
    let Some(f) = l.font_family.clone() else {
        l.data = Some(vec![crate::text::built_in_font()]);
        return Ok(());
    };
    if !crate::text::is_font_file(&f) {
        l.data = match assets.iter().find(|a| a.name == f) {
            Some(a) => a.font.clone().map(|f| vec![f]),
            None => crate::text::built_in(Some(&f)).map(|f| vec![f]),
        };
        return Ok(());
    }
    if root.outside(&f) {
        bail!(
            "layer '{}' font_family {} is outside the scene directory",
            l.name,
            py_str_repr(&f)
        );
    }
    let what = format!("layer '{}' font_family {}", l.name, py_str_repr(&f));
    let font = root.read(&f, &what).and_then(|(bytes, path)| {
        crate::text::font(&bytes).map_err(|e| format!("{what} ({path}): unreadable font: {e}"))
    });
    l.data = Some(vec![font]);
    Ok(())
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
    /// output pass renders at the renderer's size whatever this sets.
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
            &p.fragment,
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
        let mut n = Scene::new(s.passes, s.version, s.filters, s.params, s.common, s.assets)?;
        n.hidden_filters = s.hidden_filters;
        Ok(n)
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

    /// `font_family: "pixel"` draws from Pixel Operator 8, as the font's own
    /// name table and its 800 units to the em say, whether the scene loads
    /// other assets or not. A font asset of that name is the scene's own.
    #[test]
    fn the_pixel_family_draws_from_the_built_in_pixel_font() {
        let fonts = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
        let pixel = json!({"name": "t", "text": "hi", "font_family": "pixel"});
        let scene = |assets: Value, layers: Value| {
            json!({"version": 1, "assets": assets,
                   "passes": [{"name": "m", "fragment": GOOD, "layers": layers}]})
        };
        let font_of = |sc: &Scene| -> Vec<u8> {
            let l = sc.text_layers().find(|l| l.name == "t").unwrap();
            l.data.as_ref().unwrap()[0].clone().unwrap().to_vec()
        };
        let parsed = Scene::from_dict(&scene(json!({}), json!([pixel])), None, None).unwrap();
        let bytes = font_of(&parsed);
        let face = ttf_parser::Face::parse(&bytes, 0).unwrap();
        let family = face
            .names()
            .into_iter()
            .filter(|n| n.name_id == ttf_parser::name_id::FAMILY)
            .find_map(|n| n.to_string());
        assert_eq!(family.as_deref(), Some("Pixel Operator 8"));
        assert_eq!(face.units_per_em(), 800);
        let sans = json!({"sans": {"file": "DejaVuSans-ASCII.ttf"}});
        let other = json!({"name": "u", "text": "x", "font_family": "sans"});
        let mut loaded = Scene::from_dict(&scene(sans, json!([pixel, other])), None, None).unwrap();
        loaded.load_assets(fonts).unwrap();
        assert_eq!(font_of(&loaded), bytes);
        let own = json!({"pixel": {"file": "DejaVuSans-ASCII.ttf"}});
        let mut sc = Scene::from_dict(&scene(own, json!([pixel])), None, None).unwrap();
        sc.load_assets(fonts).unwrap();
        let dejavu = fs::read(format!("{fonts}/DejaVuSans-ASCII.ttf")).unwrap();
        assert_eq!(font_of(&sc), dejavu);
    }
    const COMMON: &str = "float sat(float x){ return clamp(x, 0.0, 1.0); }";
    const CODE: &str = "void main(){ gl_FragColor = vec4(sat(uv.x), 0.0, 0.0, 1.0); }";
    const FLAT: &str = "void main(){ gl_FragColor = vec4(vec3(0.5), 1.0); }";

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("scene-{tag}"))
    }

    fn repo(rel: &str) -> String {
        format!("{}/{rel}", env!("CARGO_MANIFEST_DIR"))
    }

    fn asset_scene(assets: Value) -> Value {
        json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}], "assets": assets})
    }

    fn common_scene(common: Option<Value>, passes: Option<Vec<Value>>) -> Value {
        let mut d = Map::new();
        d.insert("version".to_string(), json!(1));
        d.insert(
            "passes".to_string(),
            Value::from(passes.unwrap_or_else(|| vec![json!({"name": "main", "fragment": CODE})])),
        );
        if let Some(c) = common {
            d.insert("common".to_string(), c);
        }
        Value::Object(d)
    }

    /// A scene of passes `a` and `b`, a text layer `label` in `b` and the font
    /// asset `sans`, with `extra` merged in at the top and `pass` merged into
    /// `a`.
    fn named(extra: Value, pass: Value) -> Value {
        let mut d = json!({
            "version": 1,
            "assets": {"sans": {"file": "sans.ttf"}},
            "passes": [{"name": "a", "fragment": GOOD}, {"name": "b", "fragment": GOOD,
                "layers": [{"name": "label", "text": "hi", "font_family": "sans"}]}],
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

    /// The scene as to_dict serialises it, with no space after a separator.
    fn canon(s: &Scene) -> String {
        serde_json::to_string(&s.to_dict()).unwrap()
    }

    #[test]
    fn a_sized_pass_keeps_its_layers_and_mipmaps() {
        let mut p = Pass::of("a", GOOD).unwrap();
        p.layers =
            vec![Layer::from_dict("a layer", &json!({"name": "t", "text": "words"})).unwrap()];
        p.mipmaps = true;
        let sized = Ok::<Pass, anyhow::Error>(p.clone()).size(8, 4).unwrap();
        assert_eq!(sized.size, Some((8, 4)));
        assert_eq!(sized.layers, p.layers);
        assert!(sized.mipmaps);
    }

    #[test]
    fn every_scene_pass_declares_the_glsl_names_that_a_scene_reads() {
        let f = |v: Value| json!({"type": "float", "value": v});
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "params": {"gain": f(json!(0.5)),
                           "n": {"type": "int", "value": 2},
                           "on": {"type": "bool", "value": true},
                           "tint": {"type": "color", "value": [1, 0, 0]},
                           "off": {"type": "vec2", "value": [0.1, 0.2]},
                           "dir": {"type": "vec3", "value": [0, 0, 1]},
                           "q": {"type": "vec4", "value": [0, 0, 0, 1]},
                           "title": {"type": "string", "value": "GHEE"}},
                "assets": {"logo": {"file": "logo.png"}, "sans": {"file": "sans.ttf"}},
                "passes": [
                    {"name": "a", "fragment": GOOD, "params": {"k": f(json!(1.0))},
                     "arrays": [{"name": "amps", "size": 8}, {"name": "mask", "size": [4, 2]}],
                     "layers": [{"name": "label", "text": "hi", "font_family": "sans"}]},
                    {"name": "b", "fragment": GOOD, "channels": [{"buffer": "a"}, {"buffer": null}]},
                ],
            }),
            None,
            None,
        )
        .unwrap();
        let every = [
            "varying vec2 uv;",
            "uniform vec3 iResolution;",
            "uniform float iTime;",
            "uniform float iTimeDelta;",
            "uniform int iFrame;",
            "uniform float gain;",
            "uniform int n;",
            "uniform bool on;",
            "uniform vec3 tint;",
            "uniform vec2 off;",
            "uniform vec3 dir;",
            "uniform vec4 q;",
            "uniform sampler2D logo_tex;",
            "uniform vec2 logo_size;",
            "uniform vec2 label_size;",
        ];
        let own = [
            vec![
                "uniform sampler2D amps_tex;",
                "const int amps_count = 8;",
                "float amps_at(int i0)",
                "float amps_sample(float t)",
                "uniform sampler2D mask_tex;",
                "float mask_sample2(vec2 p)",
                "uniform float k;",
            ],
            vec![
                "uniform sampler2D iChannel0;",
                "uniform sampler2D iChannel1;",
            ],
        ];
        for hw_linear in [false, true] {
            for (i, p) in sc.passes.iter().enumerate() {
                let g = p.glsl(
                    Some(&sc.params),
                    None,
                    &sc.assets,
                    &sc.text_names(),
                    hw_linear,
                );
                let has = |want: &str| g.lines().any(|l| l.starts_with(want));
                for want in every.iter().chain(&own[i]) {
                    assert!(has(want), "pass {} lacks {want}", p.name);
                }
                for other in &own[1 - i] {
                    assert!(!has(other), "pass {} declares {other}", p.name);
                }
                // A string param stays on the host.
                assert!(!g.contains("title"), "{g}");
            }
        }
    }

    #[test]
    fn a_pass_holds_the_cover_reader_of_a_text_that_its_code_or_the_common_block_calls() {
        // The common block calls title_cover, pass a calls label_cover, and
        // pass b calls neither of its own.
        let sc = Scene::from_dict(
            &json!({
                "version": 1,
                "assets": {"sans": {"file": "sans.ttf"}},
                "common": "float t(vec2 p){ return title_cover(p, 8.0); }",
                "passes": [
                    {"name": "a", "fragment": "void main(){ gl_FragColor = \
                        vec4(label_cover(uv, 8.0) + t(uv)); }",
                     "layers": [{"name": "label", "text": "hi", "font_family": "sans"},
                                {"name": "title", "text": "ho", "font_family": "sans"}]},
                    {"name": "b", "fragment": "void main(){ gl_FragColor = vec4(t(uv)); }"},
                ],
            }),
            None,
            None,
        )
        .unwrap();
        let reader = |n: &str| format!("float {n}_cover(vec2 p, float font_size)");
        for (p, label) in sc.passes.iter().zip([true, false]) {
            let g = p.glsl(
                Some(&sc.params),
                sc.common.as_deref(),
                &sc.assets,
                &sc.text_names(),
                false,
            );
            let has = |want: &str| g.lines().any(|l| l.starts_with(want));
            assert_eq!(has(&reader("label")), label, "pass {}", p.name);
            assert!(has(&reader("title")), "pass {}", p.name);
            for size in ["uniform vec2 label_size;", "uniform vec2 title_size;"] {
                assert!(has(size), "pass {} lacks {size}", p.name);
            }
        }
    }

    #[test]
    fn the_common_block_sits_between_the_preamble_and_the_code_of_every_scene_pass() {
        let sc = Scene::from_dict(
            &common_scene(
                Some(json!(COMMON)),
                Some(vec![
                    json!({"name": "a", "fragment": CODE}),
                    json!({"name": "b", "fragment": FLAT}),
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
            let pre = g.find("uniform int iFrame;").unwrap();
            let common_at = g.find(COMMON).unwrap();
            let code_at = g.find(&p.fragment).unwrap();
            assert!(pre < common_at && common_at < code_at, "{g}");
        }
        // An empty block adds nothing.
        let p = Pass::of("m", CODE).unwrap();
        assert_eq!(
            p.glsl(None, Some(""), &[], &[], false),
            p.glsl(None, None, &[], &[], false)
        );
    }

    #[test]
    fn a_hidden_pass_saves_visible_false_and_a_shown_one_leaves_it_out() {
        let d = json!({"version": 1, "passes": [
            {"name": "a", "visible": false, "fragment": GOOD},
            {"name": "b", "fragment": GOOD},
            {"name": "c", "visible": true, "fragment": GOOD},
        ]});
        let back = Scene::from_dict(&d, None, None).unwrap().to_dict();
        assert_eq!(
            back["passes"],
            json!([
                {"name": "a", "visible": false, "fragment": GOOD},
                {"name": "b", "fragment": GOOD},
                {"name": "c", "fragment": GOOD},
            ])
        );
    }

    #[test]
    fn a_color_param_takes_hex_a_name_or_three_numbers_and_writes_back_the_form_it_came_in() {
        // Each hex pair is a byte, and the channel is byte / 255.
        for (value, want, written) in [
            (
                json!("#ff8000"),
                [1.0, 128.0 / 255.0, 0.0],
                json!("#ff8000"),
            ),
            (
                json!("#0A0b0C"),
                [10.0 / 255.0, 11.0 / 255.0, 12.0 / 255.0],
                json!("#0a0b0c"),
            ),
            (
                json!([1, 0.5, 0.25]),
                [1.0, 0.5, 0.25],
                json!([1.0, 0.5, 0.25]),
            ),
            // CSS's darkorange is #ff8c00, and transparent is black.
            (
                json!("DarkOrange"),
                [1.0, 140.0 / 255.0, 0.0],
                json!("DarkOrange"),
            ),
            (json!("transparent"), [0.0; 3], json!("transparent")),
        ] {
            let d = json!({"type": "color", "value": value});
            let p = Param::from_dict("tint", &d).unwrap();
            let ParamValue::Color(c) = p.value else {
                panic!("{value}: {:?}", p.value);
            };
            assert_eq!(c.rgb, want, "{value}");
            assert_eq!(
                p.to_dict(),
                json!({"type": "color", "value": written}),
                "{value}"
            );
        }
    }

    #[test]
    fn a_scene_that_ghee_refuses_names_what_is_wrong() {
        let pass = |extra: Value| {
            let mut p = json!({"name": "m", "fragment": GOOD});
            p.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            json!({"version": 1, "passes": [p]})
        };
        let param = |ty: &str, v: Value| pass(json!({"params": {"g": {"type": ty, "value": v}}}));
        let array = |size: Value| pass(json!({"arrays": [{"name": "a", "size": size}]}));
        let asset = |a: Value| {
            json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}],
                                      "assets": {"d": a}})
        };
        let common = |c: Value| common_scene(Some(c), None);
        let size = "size needs two positive ints [w, h] up to 16384";
        let glsl = "needs GLSL text or {\"file\": <path>}";
        let rows = [
            // a pass
            (
                pass(json!({"size": [0, 4]})),
                format!("pass 'm' {size}, got [0, 4]"),
            ),
            (
                pass(json!({"size": [8]})),
                format!("pass 'm' {size}, got [8]"),
            ),
            (
                pass(json!({"size": "8x4"})),
                format!("pass 'm' {size}, got '8x4'"),
            ),
            (
                pass(json!({"size": [8.5, 4]})),
                format!("pass 'm' {size}, got [8.5, 4]"),
            ),
            (
                pass(json!({"size": [8, true]})),
                format!("pass 'm' {size}, got [8, True]"),
            ),
            (
                pass(json!({"size": [100000, 4]})),
                format!("pass 'm' {size}, got [100000, 4]"),
            ),
            (
                pass(json!({"size": [16384, 16384]})),
                "pass 'm' is 16384x16384, over the 16777216 pixel cap".to_string(),
            ),
            (
                pass(json!({"mipmaps": "yes"})),
                "pass 'm' mipmaps needs true or false, got 'yes'".to_string(),
            ),
            (
                pass(json!({"visible": "no"})),
                "pass 'm' visible needs true or false, got 'no'".to_string(),
            ),
            (
                pass(json!({"visible": false})),
                "every pass is hidden, and a scene shows one pass at least".to_string(),
            ),
            (
                json!({"version": 1, "passes": [{"name": "m"}]}),
                "pass 'm' has no 'fragment' and no 'layers'".to_string(),
            ),
            (
                pass(json!({"fragment": 5})),
                format!("pass 'm' fragment {glsl}, got 5"),
            ),
            (
                pass(json!({"fragment": {}})),
                format!("pass 'm' fragment {glsl}, got {{}}"),
            ),
            (
                pass(json!({"fragment": {"file": ["x.glsl"]}})),
                "pass 'm' fragment file needs a path, got ['x.glsl']".to_string(),
            ),
            (
                json!({"version": 1, "passes": [{"name": "a", "fragment": GOOD},
                                                {"name": "a", "fragment": GOOD}]}),
                "pass 'a' is declared twice; a pass name is the buffer name channels bind \
                 to, so it must be unique"
                    .to_string(),
            ),
            (
                pass(json!({"fragment": {"file": "/etc/hostname"}})),
                "pass 'm' fragment names the file '/etc/hostname' but the scene has no \
                 directory to resolve it against; give the GLSL text itself"
                    .to_string(),
            ),
            (
                json!({"version": 1, "passes": "oops"}),
                "'passes' needs a list, got 'oops'".to_string(),
            ),
            (
                pass(json!({"channels": ["m"]})),
                "pass 'm': a channel needs an object like {\"buffer\": \"pass\"}, got 'm'"
                    .to_string(),
            ),
            // a param
            (
                param("color", json!([0.1, 0.2])),
                "pass 'm': color param 'g' needs '#rrggbb', a CSS colour name or three numbers \
                 [r, g, b], got [0.1, 0.2]"
                    .to_string(),
            ),
            (
                param("color", json!("rde")),
                "pass 'm': color param 'g' needs '#rrggbb', a CSS colour name or three numbers \
                 [r, g, b], got 'rde'"
                    .to_string(),
            ),
            (
                param("color", json!([0.1, 0.2, "x"])),
                "pass 'm': color param 'g' needs '#rrggbb', a CSS colour name or three numbers \
                 [r, g, b], got [0.1, 0.2, 'x']"
                    .to_string(),
            ),
            (
                param("color", json!("#ff80")),
                "pass 'm': color param 'g' needs '#rrggbb', a CSS colour name or three numbers \
                 [r, g, b], got '#ff80'"
                    .to_string(),
            ),
            (
                param("color", json!("ff8000")),
                "pass 'm': color param 'g' needs '#rrggbb', a CSS colour name or three numbers \
                 [r, g, b], got 'ff8000'"
                    .to_string(),
            ),
            (
                param("vec2", json!([0.1, 0.2, 0.3])),
                "pass 'm': vec2 param 'g' needs 2 numbers, got [0.1, 0.2, 0.3]".to_string(),
            ),
            (
                param("vec3", json!("rgb")),
                "pass 'm': vec3 param 'g' needs 3 numbers, got 'rgb'".to_string(),
            ),
            (
                param("vec4", json!([0.1, 0.2, 0.3, "x"])),
                "pass 'm': vec4 param 'g' needs 4 numbers, got [0.1, 0.2, 0.3, 'x']".to_string(),
            ),
            (
                param("float", json!("<img src=x onerror=alert(1)>")),
                "pass 'm': float param 'g' needs a number, got '<img src=x onerror=alert(1)>'"
                    .to_string(),
            ),
            (
                param("int", json!(true)),
                "pass 'm': int param 'g' needs a number, got True".to_string(),
            ),
            (
                param("float", json!(null)),
                "pass 'm': float param 'g' needs a number, got None".to_string(),
            ),
            (
                param("string", json!(1.0)),
                "pass 'm': string param 'g' needs a string, got 1.0".to_string(),
            ),
            (
                pass(json!({"params": {"g": {"type": "string", "value": "a",
                                             "options": [1, 2]}}})),
                "pass 'm': string param 'g' options need a list of strings".to_string(),
            ),
            (
                pass(json!({"params": {"g": {"type": "float", "value": 0.5,
                                             "options": ["a"]}}})),
                "pass 'm': param 'g': options belong to string params only".to_string(),
            ),
            (
                pass(json!({"params": {"g": {"type": "float", "value": 0.5, "step": "lo"}}})),
                "pass 'm': param 'g' step needs a number, got 'lo'".to_string(),
            ),
            (
                param("flaot", json!(0.5)),
                "pass 'm': param 'g' has unknown type 'flaot'; known types: float, int, bool, \
                 color, string, vec2, vec3, vec4"
                    .to_string(),
            ),
            (
                pass(json!({"params": {"g": {"value": 1}}})),
                "pass 'm': param 'g' has no 'type'".to_string(),
            ),
            (
                json!({"version": 1,
                       "passes": [{"name": "p0", "fragment": GOOD,
                                   "params": {"g": {"type": "float", "value": 0.1}}}],
                       "params": {"g": {"type": "float", "value": 0.2}}}),
                "scene param 'g' and param 'g' in pass 'p0' share a name".to_string(),
            ),
            // an array
            (
                array(json!(0)),
                "pass 'm': array 'a' size needs an int in 1..16384, got 0".to_string(),
            ),
            (
                array(json!(1e11)),
                "pass 'm': array 'a' size needs an int in 1..16384, got 100000000000".to_string(),
            ),
            (
                array(json!("16")),
                "pass 'm': array 'a' size needs an int in 1..16384, got '16'".to_string(),
            ),
            (
                array(json!(4.5)),
                "pass 'm': array 'a' size needs an int in 1..16384, got 4.5".to_string(),
            ),
            (
                array(json!([0, 4])),
                "pass 'm': array 'a' size needs two ints in 1..16384 [w, h], got [0, 4]"
                    .to_string(),
            ),
            (
                array(json!([8, "4"])),
                "pass 'm': array 'a' size needs two ints in 1..16384 [w, h], got [8, '4']"
                    .to_string(),
            ),
            (
                array(json!([8, 4, 2])),
                "pass 'm': array 'a' size needs two ints [w, h], got [8, 4, 2]".to_string(),
            ),
            (
                array(json!([8192, 4096])),
                "pass 'm': array 'a' is 8192x4096, over the 16777216 element cap".to_string(),
            ),
            (
                pass(json!({"arrays": "oops"})),
                "pass 'm': 'arrays' needs a list, got 'oops'".to_string(),
            ),
            // an asset
            (
                asset(json!({"file": "d.raw", "width": 4, "height": 4})),
                "raw asset 'd' has no 'format'".to_string(),
            ),
            (
                asset(json!({"file": "logo.png", "components": 4})),
                "asset 'd' takes no 'components', a .png file carries its own".to_string(),
            ),
            (
                asset(json!({"file": "d.raw", "width": 4, "height": 4, "format": "u16"})),
                "asset 'd' format needs 'u8' or 'f32', got 'u16'".to_string(),
            ),
            (
                asset(
                    json!({"file": "d.raw", "width": 4, "height": 4, "format": "u8",
                             "components": 2}),
                ),
                "asset 'd' components needs 1, 3 or 4, got 2".to_string(),
            ),
            (
                asset(json!({"file": "logo.png", "sampling": "cubic"})),
                "asset 'd' sampling needs 'nearest' or 'linear', got 'cubic'".to_string(),
            ),
            (
                asset(json!({"file": "sans.ttf", "sampling": "linear"})),
                "asset 'd' takes no 'sampling', a .ttf file is a font".to_string(),
            ),
            (
                asset(json!({"file": "logo.bmp"})),
                "asset 'd' file 'logo.bmp' needs a .raw, .png, .jpg, .jpeg, .ttf or .otf \
                 extension"
                    .to_string(),
            ),
            // the common block
            (common(json!(["x"])), format!("'common' {glsl}, got ['x']")),
            (common(json!({})), format!("'common' {glsl}, got {{}}")),
            (
                common(json!({"file": 5})),
                "'common' file needs a path, got 5".to_string(),
            ),
            (
                common(json!({"file": "/etc/hostname"})),
                "'common' names the file '/etc/hostname' but the scene has no directory to \
                 resolve it against; give the GLSL text itself"
                    .to_string(),
            ),
        ];
        for (d, want) in rows {
            let got = Scene::from_dict(&d, None, None).map(|_| ()).unwrap_err();
            assert_eq!(got.to_string(), format!("scene: {want}"), "{d}");
        }
    }

    #[test]
    fn a_key_that_ghee_does_not_take_is_refused_naming_the_closest_key() {
        // Each row puts one key into a scene that loads without it.
        let pass = |extra: Value| {
            let mut p = json!({"name": "main", "fragment": GOOD});
            p.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            p
        };
        let layer = |extra: Value| {
            let mut d = json!({"name": "t", "text": "words"});
            d.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            json!({"version": 1, "passes": [pass(json!({"layers": [d]}))]})
        };
        let rows = [
            (
                json!({"version": 1, "paramz": {}, "passes": [pass(json!({}))]}),
                "scene: the scene takes no 'paramz', the closest key is 'params'",
            ),
            (
                json!({"version": 1, "fragment": GOOD, "passes": [pass(json!({}))]}),
                "scene: the scene takes no 'fragment', which sits on a pass",
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
                json!({"version": 1, "texts": {}, "passes": [pass(json!({}))]}),
                "scene: the scene takes no 'texts', the closest key is 'text', which sits on a \
                 layer",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"chanels": []}))]}),
                "scene: pass 'main' takes no 'chanels', the closest key is 'channels'",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"font": "a.ttf"}))]}),
                "scene: pass 'main' takes no 'font', the closest key is 'font_family', which sits \
                 on a layer",
            ),
            (
                layer(json!({"font": "a.ttf"})),
                "scene: layer 't' takes no 'font', the closest key is 'font_family'",
            ),
            (
                layer(json!({"text_alig": "center"})),
                "scene: layer 't' takes no 'text_alig', the closest key is 'text_align'",
            ),
            (
                layer(json!({"fragment": GOOD})),
                "scene: layer 't' takes no 'fragment', which sits on its pass",
            ),
            (
                layer(json!({"vertex": GOOD})),
                "scene: layer 't' takes no 'vertex', only name, visible, x, y, color, opacity, \
                 blend, text, font_size, font_family, text_align, render, text_shadow, glow, \
                 text_stroke, karaoke, wave, scroll and blink",
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
                json!({"version": 1, "common": {"fil": "c.glsl"}, "passes": [pass(json!({}))]}),
                "scene: 'common' takes no 'fil', the closest key is 'file'",
            ),
            (
                json!({"version": 1, "passes": [pass(json!({"fragment": {"path": "m.glsl"}}))]}),
                "scene: pass 'main' fragment takes no 'path', only file",
            ),
        ];
        for (d, want) in rows {
            let got = Scene::from_dict(&d, None, None).map(|_| ()).unwrap_err();
            assert_eq!(got.to_string(), want);
        }
    }

    #[test]
    fn the_constructors_refuse_a_pass_or_an_asset_named_twice() {
        let e = Scene::of(vec![
            Pass::of("a", GOOD).unwrap(),
            Pass::of("a", GOOD).unwrap(),
        ])
        .unwrap_err()
        .to_string();
        assert!(e.contains("twice"), "{e}");
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

    #[test]
    fn an_error_in_a_scene_file_names_the_file() {
        let dir = tmpdir("file-error");
        let p = format!("{dir}/x.scene.json");
        fs::write(&p, "{oops").unwrap();
        let e = Scene::load(&p).unwrap_err().to_string();
        assert!(e.starts_with(&format!("{p}: invalid JSON: ")), "{e}");
        fs::write(
            &p,
            json!({"version": 1, "passes": [{"name": "m"}]}).to_string(),
        )
        .unwrap();
        let e = Scene::load(&p).unwrap_err().to_string();
        assert!(
            e.starts_with(&format!("{p}: pass 'm' has no 'fragment'")),
            "{e}"
        );
    }

    #[test]
    fn a_missing_file_names_its_path() {
        let dir = tmpdir("missing");
        let p = format!("{dir}/x.scene.json");
        let gone = |what: &str| format!("{what}: No such file or directory");
        assert_eq!(
            Scene::load(&p).unwrap_err().to_string(),
            gone(&p),
            "the scene file"
        );
        for (d, want) in [
            (
                json!({"version": 1,
                       "passes": [{"name": "main", "fragment": {"file": "gone.glsl"}}]}),
                gone(&format!("{p}: pass 'main' fragment file 'gone.glsl'")),
            ),
            (
                common_scene(Some(json!({"file": "gone.glsl"})), None),
                gone(&format!("{p}: 'common' file 'gone.glsl'")),
            ),
        ] {
            fs::write(&p, d.to_string()).unwrap();
            assert_eq!(Scene::load(&p).unwrap_err().to_string(), want);
        }
        let mut sc = Scene::from_dict(
            &asset_scene(json!({"logo": {"file": "logo.png"}})),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            sc.load_assets_from(&dir, Reach::Any, &[], &dir)
                .unwrap_err()
                .to_string(),
            format!(
                "{}, and $GHEE_ASSETS names no directory",
                gone(&format!("asset 'logo' ({dir}/logo.png)"))
            )
        );
    }

    #[test]
    fn a_common_file_is_inlined_like_a_pass_fragment() {
        let base = tmpdir("common-file");
        fs::write(format!("{base}/shared.glsl"), COMMON).unwrap();
        let sc = Scene::from_dict(
            &common_scene(Some(json!({"file": "shared.glsl"})), None),
            Some(&base),
            None,
        )
        .unwrap();
        assert_eq!(sc.common.as_deref(), Some(COMMON));
        // The console posts the GLSL back as text, and a save writes the same form.
        assert_eq!(sc.to_dict()["common"], json!(COMMON));
        let p = format!("{base}/x.scene.json");
        fs::write(
            &p,
            common_scene(Some(json!({"file": "shared.glsl"})), None).to_string(),
        )
        .unwrap();
        assert_eq!(Scene::load(&p).unwrap().common.as_deref(), Some(COMMON));
    }

    #[test]
    fn serialized_scene_reloads_without_a_directory() {
        let scene = Scene::load(&repo("ghee/examples/bars.scene.json")).unwrap();
        let d = scene.to_dict();
        // The console posts the GLSL back as text.
        assert!(d["passes"][0]["fragment"].is_string());
        assert_eq!(
            Scene::from_dict(&d, None, None).unwrap().passes[0].fragment,
            scene.passes[0].fragment
        );
    }

    #[test]
    fn an_asset_resolves_through_ghee_assets_after_the_scene_directory() {
        let scene_dir = tmpdir("asset-scene");
        let asset_dir = tmpdir("asset-store");
        fs::write(format!("{asset_dir}/n.raw"), [1u8, 2, 3, 4]).unwrap();
        fs::write(format!("{scene_dir}/m.raw"), [5u8, 6, 7, 8]).unwrap();
        fs::write(format!("{asset_dir}/m.raw"), [0u8; 4]).unwrap();
        let raw = |file: &str| json!({"file": file, "width": 2, "height": 2, "format": "u8"});
        let doc = asset_scene(json!({"n": raw("n.raw"), "mask": raw("m.raw")}));
        let mut scene = Scene::from_dict(&doc, None, None).unwrap();
        let e = scene
            .load_assets_from(&scene_dir, Reach::Any, &[], &scene_dir)
            .unwrap_err()
            .to_string();
        assert_eq!(
            e,
            format!(
                "asset 'n' ({scene_dir}/n.raw): No such file or directory, and \
                 $GHEE_ASSETS names no directory"
            )
        );
        scene
            .load_assets_from(&scene_dir, Reach::Any, &[asset_dir.to_string()], &scene_dir)
            .unwrap();
        let px = |i: usize| scene.assets[i].image.as_ref().unwrap().pixels.clone();
        assert_eq!(px(0), crate::assets::Pixels::U8(vec![1, 2, 3, 4]));
        // the scene directory comes first
        assert_eq!(px(1), crate::assets::Pixels::U8(vec![5, 6, 7, 8]));
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
        assert_eq!(
            Scene::load(&p).unwrap_err().to_string(),
            format!("{p}: asset 'd' ({dir}/d.raw): 5 bytes, but 2x2x1 u8 needs 4")
        );
    }

    #[test]
    fn an_f32_raw_asset_reads_little_endian() {
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
        d["passes"][0] = json!({"name": "main", "fragment": {"file": format!("{root}/in.glsl")}});
        let p = format!("{base}/x.scene.json");
        fs::write(&p, d.to_string()).unwrap();
        let sc = Scene::load(&p).unwrap();
        let px = |i: usize| sc.assets[i].image.as_ref().unwrap().pixels.clone();
        assert_eq!(px(0), crate::assets::Pixels::U8(vec![7u8; 4]));
        assert_eq!(px(1), crate::assets::Pixels::U8(vec![9u8; 4]));
        assert_eq!(sc.passes[0].fragment, GOOD);
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
        let d = json!({"version": 1,
                       "passes": [{"name": "main", "fragment": {"file": format!("{rel}/in.glsl")}}]});
        let sc = Scene::from_dict(&d, Some("/"), None).unwrap();
        assert_eq!(sc.passes[0].fragment, GOOD);
        let mut missing = Scene::from_dict(
            &asset_scene(json!({"mask": {"file": "ghee-no-such.png"}})),
            None,
            None,
        )
        .unwrap();
        let e = missing.load_assets("/").unwrap_err().to_string();
        assert!(e.contains("(/ghee-no-such.png)"), "{e}");
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
    fn an_image_layer_decodes_its_file_and_names_it_in_a_failure() {
        let dir = tmpdir("layerfile");
        let mut png = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut png, 2, 1);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[255, 0, 0, 255, 0, 0, 255, 128])
                .unwrap();
        }
        fs::write(dir.join("pic.png"), &png).unwrap();
        fs::write(dir.join("bad.png"), b"not a png").unwrap();
        let load = |image: &str| {
            let doc = json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD,
                "layers": [{"name": "icon", "image": image}]}]});
            let mut sc = Scene::from_dict(&doc, None, None).unwrap();
            sc.load_assets_from(&dir, Reach::Any, &[], &dir)
                .map(|()| sc)
        };
        let sc = load("pic.png").unwrap();
        let l = sc.layers().find_map(Layer::as_image).unwrap();
        assert_eq!(
            l.data,
            Some(crate::assets::Image {
                width: 2,
                height: 1,
                components: 4,
                pixels: crate::assets::Pixels::U8(vec![255, 0, 0, 255, 0, 0, 255, 128]),
            })
        );
        let gone = load("gone.png").unwrap_err().to_string();
        assert_eq!(
            gone,
            format!(
                "layer 'icon' image 'gone.png' ({}): No such file or directory, and \
                 $GHEE_ASSETS names no directory",
                dir.join("gone.png").display()
            )
        );
        let bad = load("bad.png").unwrap_err().to_string();
        let key = format!(
            "layer 'icon' image 'bad.png' ({}): ",
            dir.join("bad.png").display()
        );
        assert!(bad.starts_with(&key) && bad.len() > key.len(), "{bad}");
    }

    #[test]
    fn a_scene_of_the_console_reads_no_file_outside_its_directory() {
        let _g = crate::filters::env_guard();
        let root = tmpdir("outside");
        let base = format!("{root}/scene");
        fs::create_dir_all(format!("{base}/sub")).unwrap();
        fs::write(format!("{base}/sub/in.glsl"), GOOD).unwrap();
        for f in [
            "secret.glsl",
            "secret.raw",
            "up.ttf",
            "up.png",
            "scene/m.raw",
        ] {
            fs::write(format!("{root}/{f}"), [0u8; 4]).unwrap();
        }
        let shader = |path: &str| json!({"version": 1, "passes": [{"name": "main", "fragment": {"file": path}}]});
        let secret = format!("{root}/secret.glsl");
        for (d, what) in [
            (
                shader("../secret.glsl"),
                "pass 'main' fragment file '../secret.glsl'".to_string(),
            ),
            (
                shader(&secret),
                format!("pass 'main' fragment file '{secret}'"),
            ),
            (
                shader("sub/../../secret.glsl"),
                "pass 'main' fragment file 'sub/../../secret.glsl'".to_string(),
            ),
            (
                common_scene(Some(json!({"file": "../secret.glsl"})), None),
                "'common' file '../secret.glsl'".to_string(),
            ),
        ] {
            let e = Scene::from_dict_in(&d, &base, Reach::Inside, None).unwrap_err();
            assert_eq!(
                e.to_string(),
                format!("scene: {what} is outside the scene directory")
            );
        }
        let inside = Scene::from_dict_in(&shader("sub/in.glsl"), &base, Reach::Inside, None);
        assert_eq!(inside.unwrap().passes[0].fragment, GOOD);
        let raw = json!({"file": "../secret.raw", "width": 2, "height": 2, "format": "u8"});
        for (d, what) in [
            (
                asset_scene(json!({"d": raw})),
                "asset 'd' file '../secret.raw'",
            ),
            (
                json!({"version": 1, "assets": {"face": {"file": "../up.ttf"}},
                       "passes": [{"name": "m", "fragment": GOOD,
                           "layers": [{"name": "label", "text": "hi", "font_family": "face"}]}]}),
                "asset 'face' file '../up.ttf'",
            ),
            (
                json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD,
                    "layers": [{"name": "label", "text": "hi", "font_family": "../up.ttf"}]}]}),
                "layer 'label' font_family '../up.ttf'",
            ),
            (
                json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD,
                    "layers": [{"name": "icon", "image": "../up.png"}]}]}),
                "layer 'icon' image '../up.png'",
            ),
        ] {
            let mut sc = Scene::from_dict(&d, None, None).unwrap();
            let e = sc
                .load_assets_from(&base, Reach::Inside, &[], &base)
                .unwrap_err();
            assert_eq!(
                e.to_string(),
                format!("{what} is outside the scene directory")
            );
        }
        // A filter's asset stays inside the filter's directory.
        let filters = format!("{base}/filters");
        fs::create_dir_all(&filters).unwrap();
        fs::write(
            format!("{filters}/esc.filter.json"),
            json!({"name": "esc", "fragment": GOOD,
                   "assets": {"m": {"file": "../m.raw", "width": 2, "height": 2, "format": "u8"}}})
            .to_string(),
        )
        .unwrap();
        let reg = crate::filters::load_filters(&[&filters]);
        let mut sc = Scene::of(vec![Pass::of("m", GOOD).unwrap()]).unwrap();
        let e = crate::filters::apply_filters(&mut sc, &["esc".to_string()], Some(&reg))
            .unwrap_err()
            .to_string();
        assert_eq!(
            e,
            "filter 'esc' asset 'm' file '../m.raw' is outside the filter directory"
        );
    }

    /// Noto Color Emoji cut to space, U+200D, U+1F600 and U+1F642, which
    /// tests copy under the name of the emoji font's file.
    const EMOJI: &str = "tools/fixtures/fonts/NotoColorEmoji-Smiles.ttf";

    /// A scene of one pass, with a text layer of `words` for each name of
    /// `names`.
    fn words_scene(names: &[&str], words: &str) -> Value {
        let layers: Vec<Value> = names
            .iter()
            .map(|n| json!({"name": n, "text": words}))
            .collect();
        json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD, "layers": layers}]})
    }

    #[test]
    fn a_scene_takes_the_emoji_font_beside_it_then_from_ghee_assets_then_from_the_system() {
        let root = tmpdir("emoji-order");
        // The scene's directory, two on $GHEE_ASSETS, and the system's.
        let dirs = ["scene", "a", "b", "system"].map(|d| format!("{}/{d}", &*root));
        let path = |k: usize| format!("{}/{}", dirs[k], crate::emoji::FILE);
        for d in &dirs {
            fs::create_dir_all(d).unwrap();
        }
        let doc = words_scene(&["label", "title"], "\u{1f600}");
        let load = || {
            let mut sc = Scene::from_dict(&doc, None, None).unwrap();
            let (done, read) = crate::sources::collect(|| {
                sc.load_assets_from(&dirs[0], Reach::Any, &dirs[1..3], &dirs[3])
            });
            done.unwrap();
            (sc, read)
        };
        // The directories that hold the font, and the one that the layers
        // take it from. The scene depends on each path tried until then.
        for (holding, takes) in [
            (&[0, 1, 2, 3][..], Some(0)),
            (&[1, 2, 3], Some(1)),
            (&[2, 3], Some(2)),
            (&[3], Some(3)),
            (&[], None),
        ] {
            for k in 0..4 {
                let _ = fs::remove_file(path(k));
            }
            for &k in holding {
                fs::copy(repo(EMOJI), path(k)).unwrap();
            }
            let (sc, read) = load();
            let got: Vec<Option<String>> = sc
                .text_layers()
                .map(|l| l.emoji.as_ref().map(|e| e.path().to_string()))
                .collect();
            assert_eq!(got, [takes.map(path), takes.map(path)], "{holding:?}");
            let tried: std::collections::HashSet<std::path::PathBuf> =
                (0..=takes.unwrap_or(3)).map(|k| path(k).into()).collect();
            assert_eq!(read, tried, "{holding:?}");
        }
        // The font showing up where the scene looked marks the scene stale.
        let (_, read) = load();
        let mut snap = crate::sources::Snapshot::with(read, 0.0, Box::new(|| 0.0));
        assert!(!snap.stale());
        fs::copy(repo(EMOJI), path(3)).unwrap();
        assert!(snap.stale());
    }

    #[test]
    fn the_console_reads_the_emoji_font_outside_its_scenes_and_no_other_file_there() {
        let root = tmpdir("emoji-outside");
        let (base, system) = (format!("{}/scene", &*root), format!("{}/noto", &*root));
        fs::create_dir_all(&base).unwrap();
        fs::create_dir_all(&system).unwrap();
        let font = format!("{system}/{}", crate::emoji::FILE);
        fs::copy(repo(EMOJI), &font).unwrap();
        let mut sc = Scene::from_dict(&words_scene(&["label"], "\u{1f600}"), None, None).unwrap();
        sc.load_assets_from(&base, Reach::Inside, &[], &system)
            .unwrap();
        let l = sc.text_layer("label").unwrap();
        assert_eq!(l.emoji.as_ref().map(|e| e.path()), Some(font.as_str()));
        // The scene names the same file in vain.
        let mut named = words_scene(&["label"], "\u{1f600}");
        named["passes"][0]["layers"][0]["font_family"] = json!(font);
        let mut asset = named.clone();
        asset["passes"][0]["layers"][0]["font_family"] = json!("emoji");
        asset["assets"] = json!({"emoji": {"file": font}});
        for (doc, what) in [
            (named, "layer 'label' font_family"),
            (asset, "asset 'emoji' file"),
        ] {
            let mut sc = Scene::from_dict(&doc, None, None).unwrap();
            let e = sc
                .load_assets_from(&base, Reach::Inside, &[], &system)
                .unwrap_err()
                .to_string();
            assert!(
                e.starts_with(what) && e.ends_with("is outside the scene directory"),
                "{e}"
            );
        }
    }

    #[test]
    fn every_layer_of_every_load_shares_one_read_of_the_emoji_font() {
        let root = tmpdir("emoji-shared");
        let font = format!("{}/{}", &*root, crate::emoji::FILE);
        fs::copy(repo(EMOJI), &font).unwrap();
        let doc = words_scene(&["label", "title"], "\u{1f600}");
        // The bytes of the emoji font, as each layer of a new load draws
        // from them.
        let load = || {
            let mut sc = Scene::from_dict(&doc, None, None).unwrap();
            sc.load_assets_from(&root, Reach::Any, &[], &root).unwrap();
            let bytes: Vec<std::sync::Arc<Vec<u8>>> =
                sc.text_layers().map(|l| l.font(1).unwrap()).collect();
            bytes
        };
        let all = [load(), load()].concat();
        assert!(all.iter().all(|b| std::sync::Arc::ptr_eq(b, &all[0])));
        // A changed file reads again.
        let mut grown = fs::read(&font).unwrap();
        grown.push(0);
        fs::write(&font, &grown).unwrap();
        assert_eq!(*load()[0], grown);
    }

    #[test]
    fn two_declarations_of_one_glsl_name_in_a_pass_are_refused_naming_both() {
        let f = json!({"type": "float", "value": 1.0});
        let c = json!({"type": "color", "value": [1.0, 0.0, 0.0]});
        for (extra, pass, want) in [
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
                json!({"params": {"label_cover": c}}),
                json!({}),
                "scene param 'label_cover' and layer 'label' both declare 'label_cover'",
            ),
            (
                json!({}),
                json!({"params": {"label_size": f}}),
                "layer 'label' and param 'label_size' in pass 'a' both declare 'label_size'",
            ),
            (
                json!({"params": {"gain_tex": f}, "assets": {"gain": {"file": "g.png"}}}),
                json!({}),
                "scene param 'gain_tex' and asset 'gain' both declare 'gain_tex'",
            ),
            (
                json!({"assets": {"gain": {"file": "g.png"}}}),
                json!({"params": {"gain_tex": f}}),
                "asset 'gain' and param 'gain_tex' in pass 'a' both declare 'gain_tex'",
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
        // a name that only looks like ghee's or GLSL's.
        let extra = json!({
            "params": {"gains": f, "amps_ats": f, "label_covers": f, "glyphs": f,
                       "glyph": f, "Ghee_x": f, "GL_x": f, "gl": f, "iFrames": f,
                       "iTimeDeltas": f, "iChannel": f, "iChannelx": f, "iTimes": f,
                       "uvs": f, "a_b_c": f,
                       "words": {"type": "string", "value": "w"}},
            "assets": {"logo": {"file": "l.png"}, "logos": {"file": "l.png"}},
        });
        let pass = json!({"params": {"gain": f, "word": f, "logo_sizes": f},
                          "arrays": [{"name": "labels", "size": [4, 4]},
                                     {"name": "state", "size": 4}]});
        let mut d = named(extra, pass);
        // Two passes may each declare one param name and one array name.
        d["passes"][1]["params"] = json!({"gain": f});
        d["passes"][1]["arrays"] = json!([{"name": "state", "size": 4}]);
        assert!(Scene::from_dict(&d, None, None).is_ok());
    }

    #[test]
    fn a_built_in_name_is_refused_as_any_declared_name() {
        let f = json!({"type": "float", "value": 1.0});
        for name in [
            "iTime",
            "iTimeDelta",
            "iFrame",
            "iResolution",
            "uv",
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
            // A layer gives GLSL its name with a suffix, so it takes a built-in
            // name.
            let mut d = named(json!({}), json!({}));
            d["passes"][1]["layers"][0]["name"] = json!(name);
            assert!(Scene::from_dict(&d, None, None).is_ok(), "layer {name}");
        }
    }

    #[test]
    fn a_name_that_breaks_the_name_rule_is_refused_saying_how() {
        let f = json!({"type": "float", "value": 1.0});
        let words = json!({"type": "string", "value": "w"});
        let chars = "needs ASCII letters, digits and '_', and a letter first";
        for (extra, pass, want) in [
            (
                json!({}),
                json!({"name": "my pass"}),
                format!("pass name 'my pass' {chars}"),
            ),
            (
                json!({}),
                json!({"name": "caf\u{e9}"}),
                format!("pass name 'caf\u{e9}' {chars}"),
            ),
            (
                json!({"params": {"2x": f}}),
                json!({}),
                format!("scene param name '2x' {chars}"),
            ),
            (
                json!({"params": {"two words": words}}),
                json!({}),
                format!("scene param name 'two words' {chars}"),
            ),
            (
                json!({}),
                json!({"params": {"_x": f}}),
                format!("param name '_x' in pass 'a' {chars}"),
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "a-b", "size": 4}]}),
                format!("array name 'a-b' in pass 'a' {chars}"),
            ),
            (
                json!({}),
                json!({"layers": [{"name": "my glow", "text": "w"}]}),
                format!("layer name 'my glow' {chars}"),
            ),
            (
                json!({"assets": {"": {"file": "l.png"}}}),
                json!({}),
                format!("asset name '' {chars}"),
            ),
            (
                json!({}),
                json!({"name": "a_"}),
                "pass name 'a_' needs a letter or a digit last".to_string(),
            ),
            (
                json!({}),
                json!({"arrays": [{"name": "a__b", "size": 4}]}),
                "array name 'a__b' in pass 'a' holds '__', which GLSL reserves".to_string(),
            ),
            (
                json!({}),
                json!({"params": {"gl_x": f}}),
                "param name 'gl_x' in pass 'a' starts with 'gl_', which GLSL reserves".to_string(),
            ),
            (
                json!({"assets": {"gheex": {"file": "l.png"}}}),
                json!({}),
                "asset name 'gheex' starts with 'ghee', which ghee reserves".to_string(),
            ),
        ] {
            assert_eq!(named_err(extra, pass), format!("scene: {want}"));
        }
    }

    #[test]
    fn a_name_is_unique_across_the_scene_and_within_its_pass() {
        let f = json!({"type": "float", "value": 1.0});
        let words = json!({"type": "string", "value": "w"});
        for (extra, pass, want) in [
            (
                json!({"params": {"a": f}}),
                json!({}),
                "pass 'a' and scene param 'a' share a name",
            ),
            (
                json!({"assets": {"a": {"file": "a.png"}}}),
                json!({}),
                "pass 'a' and asset 'a' share a name",
            ),
            (
                json!({"assets": {"gain": {"file": "g.png"}}, "params": {"gain": words}}),
                json!({}),
                "asset 'gain' and scene param 'gain' share a name",
            ),
            (
                json!({"params": {"gain": f}}),
                json!({"params": {"gain": f}}),
                "scene param 'gain' and param 'gain' in pass 'a' share a name",
            ),
            (
                json!({}),
                json!({"params": {"b": words}}),
                "pass 'b' and param 'b' in pass 'a' share a name",
            ),
            (
                json!({"assets": {"amps": {"file": "amps.png"}}}),
                json!({"arrays": [{"name": "amps", "size": 8}]}),
                "asset 'amps' and array 'amps' in pass 'a' share a name",
            ),
            (
                json!({}),
                json!({"params": {"amps": f}, "arrays": [{"name": "amps", "size": 8}]}),
                "param 'amps' and array 'amps' in pass 'a' share a name",
            ),
            (
                json!({}),
                json!({"layers": [{"name": "b", "text": "w"}]}),
                "pass 'b' and layer 'b' share a name",
            ),
            (
                json!({}),
                json!({"layers": [{"name": "label", "text": "w"}]}),
                "layer 'label' is declared twice",
            ),
            (
                json!({"params": {"label": words}}),
                json!({}),
                "layer 'label' and scene param 'label' share a name",
            ),
            (
                json!({"assets": {"label": {"file": "l.png"}}}),
                json!({}),
                "layer 'label' and asset 'label' share a name",
            ),
            (
                json!({}),
                json!({"params": {"label": words}}),
                "layer 'label' and param 'label' in pass 'a' share a name",
            ),
        ] {
            assert_eq!(named_err(extra, pass), format!("scene: {want}"));
        }
    }

    #[test]
    fn a_filters_entry_is_a_name_or_a_name_with_its_eye() {
        // The entry as a file writes it, and as to_dict writes it back.
        for (entry, back) in [
            (json!("dither"), json!("dither")),
            (json!({"name": "dither"}), json!("dither")),
            (json!({"name": "dither", "visible": true}), json!("dither")),
            (
                json!({"name": "dither", "visible": false}),
                json!({"name": "dither", "visible": false}),
            ),
        ] {
            let d = json!({"passes": [{"name": "m", "fragment": "x"}], "filters": [entry]});
            let sc = Scene::from_dict(&d, None, None).unwrap();
            assert_eq!(sc.filters, ["dither"], "{entry}");
            assert_eq!(sc.to_dict()["filters"], json!([back]), "{entry}");
        }
    }

    #[test]
    fn a_filters_entry_of_another_shape_is_refused() {
        for (entry, got) in [
            (json!(5), "5"),
            (json!({"name": 1}), "{'name': 1}"),
            (json!({"visible": false}), "{'visible': False}"),
            (
                json!({"name": "dither", "visible": "no"}),
                "{'name': 'dither', 'visible': 'no'}",
            ),
            (
                json!({"name": "dither", "visble": false}),
                "{'name': 'dither', 'visble': False}",
            ),
        ] {
            let d = json!({"passes": [{"name": "m", "fragment": "x"}], "filters": [entry]});
            assert_eq!(
                Scene::from_dict(&d, None, None).unwrap_err().to_string(),
                format!(
                    "scene: 'filters' takes names and objects with a 'name' and a 'visible', got {got}"
                )
            );
        }
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
            "scene param 'amount' and param 'amount' in pass 'dither' share a name"
        );
        let doc = named(json!({"params": {"frc256_bluenoise_tex": f}}), json!({}));
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        let e = crate::filters::apply_filters(&mut sc, &["frc256".to_string()], None).unwrap_err();
        assert_eq!(
            e.to_string(),
            "filter 'frc256': scene param 'frc256_bluenoise_tex' and asset 'bluenoise' of filter \
             'frc256' both declare 'frc256_bluenoise_tex'"
        );
        // A layer's names stay out of a filter pass.
        let doc = named(json!({"params": {"bluenoise": f}}), json!({}));
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        crate::filters::apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
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

    #[test]
    fn to_dict_matches_the_reference_json() {
        // What to_dict serialises for this input: key order, dropped keys,
        // and int or float per field.
        let d = json!({
            "version": 2,
            "passes": [
                {"name": "a", "fragment": "x", "size": [8, 8], "mipmaps": true},
                {"name": "b", "fragment": "y",
                 "arrays": [{"name": "v", "size": 4}, {"name": "m", "size": [3, 5]}],
                 "channels": [{"buffer": "a"}, {"buffer": null}],
                 "params": {"g": {"type": "float", "value": 1, "min": 0, "max": 2, "step": 0.5},
                            "n": {"type": "int", "value": 3},
                            "on": {"type": "bool", "value": true},
                            "c": {"type": "color", "value": [1, 0, 0.5]},
                            "v2": {"type": "vec2", "value": [1, 2]},
                            "v3": {"type": "vec3", "value": [1, 2, 3]},
                            "v4": {"type": "vec4", "value": [1, 2, 3, 4]},
                            "s": {"type": "string", "value": "hi", "options": ["hi", "bye"]}},
                 "filter": "dither"}],
            "filters": ["dither", "dither"],
            "params": {"S": {"type": "float", "value": 0.5}},
            "common": "float f(){ return 1.0; }",
            "assets": {"stbn": {"file": "stbn.raw", "width": 64, "height": 32, "format": "f32"},
                       "mask": {"file": "mask.raw", "width": 8, "height": 8, "format": "u8",
                                "components": 4},
                       "logo": {"file": "logo.png", "sampling": "linear"},
                       "photo": {"file": "photo.jpg"}},
        });
        assert_eq!(
            canon(&Scene::from_dict(&d, None, None).unwrap()),
            r#"{"version":2,"passes":[{"name":"a","fragment":"x","size":[8,8],"mipmaps":true},{"name":"b","fragment":"y","arrays":[{"name":"v","size":4},{"name":"m","size":[3,5]}],"channels":[{"buffer":"a"},{"buffer":null}],"params":{"g":{"type":"float","value":1.0,"min":0,"max":2,"step":0.5},"n":{"type":"int","value":3},"on":{"type":"bool","value":true},"c":{"type":"color","value":[1.0,0.0,0.5]},"v2":{"type":"vec2","value":[1.0,2.0]},"v3":{"type":"vec3","value":[1.0,2.0,3.0]},"v4":{"type":"vec4","value":[1.0,2.0,3.0,4.0]},"s":{"type":"string","value":"hi","options":["hi","bye"]}},"filter":"dither"}],"common":"float f(){ return 1.0; }","filters":["dither","dither"],"params":{"S":{"type":"float","value":0.5}},"assets":{"stbn":{"file":"stbn.raw","width":64,"height":32,"format":"f32"},"mask":{"file":"mask.raw","width":8,"height":8,"format":"u8","components":4},"logo":{"file":"logo.png","sampling":"linear"},"photo":{"file":"photo.jpg"}}}"#
        );
        // A null range or option list is no range or option list. to_dict drops
        // an empty filter mark and a false mipmaps. An absent version takes the
        // default.
        let sparse = json!({
            "passes": [{"name": "m", "fragment": "x", "filter": "", "mipmaps": false,
                        "params": {"g": {"type": "int", "value": 2.0, "min": null,
                                         "max": null, "step": null, "options": null}}}],
        });
        assert_eq!(
            canon(&Scene::from_dict(&sparse, None, None).unwrap()),
            r#"{"version":1,"passes":[{"name":"m","fragment":"x","params":{"g":{"type":"int","value":2}}}]}"#
        );
        // "" is a common block the console makes. None is no block.
        let empty_common = json!({"passes": [{"name": "m", "fragment": "z"}], "common": ""});
        assert_eq!(
            canon(&Scene::from_dict(&empty_common, None, None).unwrap()),
            r#"{"version":1,"passes":[{"name":"m","fragment":"z"}],"common":""}"#
        );
        // to_dict carries any version through unchanged.
        let odd = json!({"version": "x", "passes": [{"name": "m", "fragment": "x"}]});
        assert_eq!(
            canon(&Scene::from_dict(&odd, None, None).unwrap()),
            r#"{"version":"x","passes":[{"name":"m","fragment":"x"}]}"#
        );
    }

    #[test]
    fn to_dict_round_trips_the_shipped_scenes() {
        // A scene reloaded from its own to_dict, or from the file that save
        // writes, serialises to the same bytes. A second save never rewrites
        // the file.
        let dir = repo("ghee/examples");
        let out = tmpdir("save");
        let mut names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".scene.json"))
            .collect();
        names.sort();
        assert!(!names.is_empty());
        for n in names {
            let sc = Scene::load(&format!("{dir}/{n}")).unwrap();
            let one = canon(&sc);
            let back: Value = serde_json::from_str(&one).unwrap();
            assert_eq!(
                canon(&Scene::from_dict(&back, None, None).unwrap()),
                one,
                "{n}"
            );
            let saved = format!("{out}/{n}");
            sc.save(&saved).unwrap();
            let back: Value = serde_json::from_str(&fs::read_to_string(&saved).unwrap()).unwrap();
            assert_eq!(
                canon(&Scene::from_dict(&back, None, None).unwrap()),
                one,
                "{n} saved"
            );
        }
    }

    #[test]
    fn a_live_image_is_an_image_asset_of_the_scenes_own_or_an_image_file() {
        let doc = json!({
            "assets": {"pic": {"file": "pic.png"}, "pic2": {"file": "pic2.png"},
                       "sans": {"file": "sans.ttf"}},
            "passes": [{"name": "main", "fragment": GOOD,
                        "layers": [{"name": "i", "image": "pic"}]}],
        });
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        crate::filters::apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
        let filter = sc.assets.iter().find(|a| !a.from_filter.is_empty());
        let filter = filter.unwrap().name.clone();
        let image = Address {
            layer: "i",
            effect: None,
            field: "image",
        };
        for value in ["pic2", "pic.png", "wifi.JPG"] {
            sc.set_layer(&image, &json!(value)).unwrap();
            assert_eq!(sc.layer("i").unwrap().as_image().unwrap().image, value);
        }
        let mut refused = |value: Value| sc.set_layer(&image, &value).unwrap_err().to_string();
        for value in ["sans", "nope", &filter] {
            assert_eq!(
                refused(json!(value)),
                format!(
                    "layer 'i' image needs an image asset of the scene or a .png or .jpg file, \
                     got '{value}'"
                )
            );
        }
        let nope = Address {
            layer: "nope",
            ..image
        };
        assert_eq!(
            sc.set_layer(&nope, &json!(1)).unwrap_err().to_string(),
            "the scene has no layer 'nope'"
        );
    }
}
