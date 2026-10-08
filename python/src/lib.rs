//! ghee for Python: open a scene, feed it, ask for frames.
//!
//! ```python
//! import ghee
//!
//! with ghee.open("scene.json", display="kms", fps=30) as h:
//!     while playing:
//!         h.set_param("gain", 1.5)
//!         h.feed("level", bars)
//!         h.frame()
//! ```
//!
//! The engine is thread-affine. Open a handle on one thread, and use it on that
//! same thread.

mod draw;
mod serve;
mod session;
mod wire;

use std::path::PathBuf;
use std::thread::{self, ThreadId};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyString, PyTuple};
use serde_json::{Map, Value};

use session::{Fault, Fed, Opening, Session, Source, StreamMax, Tick};

/// A scene error is the caller's text: ValueError. A system error is the
/// host's: RuntimeError.
fn fault(f: Fault) -> PyErr {
    match f.scene {
        true => PyValueError::new_err(f.message),
        false => PyRuntimeError::new_err(f.message),
    }
}

/// A value that belongs to one thread.
///
/// The engine makes its GL context current on the thread that built it, and
/// the context closes nowhere else.
struct Affine<T> {
    owner: ThreadId,
    held: Option<T>,
}

// get() is what holds the value on its thread.
unsafe impl<T> Send for Affine<T> {}
unsafe impl<T> Sync for Affine<T> {}

impl<T> Affine<T> {
    fn new(value: T) -> Affine<T> {
        Affine {
            owner: thread::current().id(),
            held: Some(value),
        }
    }

    fn mine(&self) -> PyResult<()> {
        if thread::current().id() != self.owner {
            return Err(PyRuntimeError::new_err(
                "ghee is thread-affine: this handle belongs to the thread that opened it",
            ));
        }
        Ok(())
    }

    fn get(&mut self) -> PyResult<&mut T> {
        self.mine()?;
        self.held
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("this handle is closed"))
    }

    fn take(&mut self) -> PyResult<Option<T>> {
        self.mine()?;
        Ok(self.held.take())
    }
}

impl<T> Drop for Affine<T> {
    fn drop(&mut self) {
        if thread::current().id() != self.owner {
            // The GL context is this thread's to close. A drop anywhere else
            // leaks it instead.
            std::mem::forget(self.held.take());
        }
    }
}

/// The session stays on this thread. Detaching only lets the other Python
/// threads run while the frame renders.
struct Held<'a>(&'a mut Session);

unsafe impl Send for Held<'_> {}

impl Held<'_> {
    fn frame(&mut self) -> session::Outcome<Tick> {
        self.0.frame()
    }

    fn skip(&mut self) -> session::Outcome<Tick> {
        self.0.skip()
    }
}

/// A Python value as JSON. A param value and the display options travel
/// this way.
fn to_json(v: &Bound<'_, PyAny>) -> PyResult<Value> {
    if v.is_none() {
        return Ok(Value::Null);
    }
    // a Python bool is an int, so it answers first
    if let Ok(b) = v.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(s) = v.cast::<PyString>() {
        return Ok(Value::from(s.extract::<String>()?));
    }
    if let Ok(d) = v.cast::<PyDict>() {
        let mut out = Map::new();
        for (k, item) in d.iter() {
            out.insert(k.str()?.extract::<String>()?, to_json(&item)?);
        }
        return Ok(Value::Object(out));
    }
    // A float answers before the int, whose failed extraction raises and
    // drops a Python error.
    if let Ok(f) = v.cast::<PyFloat>() {
        return Ok(Value::from(f.value()));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(Value::from(f));
    }
    if let Ok(items) = v.try_iter() {
        let mut out: Vec<Value> = Vec::new();
        for item in items {
            out.push(to_json(&item?)?);
        }
        return Ok(Value::Array(out));
    }
    Err(PyValueError::new_err(format!(
        "ghee takes no value of type {}",
        v.get_type().name()?
    )))
}

/// The values in a feed. A uint8 buffer reaches the texture as byte / 255.
/// Any other buffer or iterable of numbers converts to float32 in the shim's
/// `_f32`, the conversion that the wire takes too.
///
/// The buffer protocol is no part of the stable ABI, so `bytes` reads straight
/// out and any other exporter goes through a memoryview.
fn fed(name: &str, data: &Bound<'_, PyAny>) -> PyResult<Fed> {
    if let Ok(b) = data.cast::<PyBytes>() {
        return Ok(Fed::U8(b.as_bytes().to_vec()));
    }
    let py = data.py();
    let view = py.import("builtins")?.getattr("memoryview")?.call1((data,));
    if let Ok(view) = view {
        if view.getattr("format")?.extract::<String>()? == "B" {
            let raw = view.call_method0("tobytes")?;
            return Ok(Fed::U8(raw.cast::<PyBytes>()?.as_bytes().to_vec()));
        }
    }
    let converted = py
        .import("ghee._shim")?
        .getattr("_f32")?
        .call1((data, format!("feed '{name}'")))?;
    let raw = converted.get_item(1)?;
    Ok(Fed::F32(
        raw.cast::<PyBytes>()?
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    ))
}

fn tick_dict(py: Python<'_>, tick: Tick) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frame", tick.frame)?;
    d.set_item("time", tick.time)?;
    d.set_item("dt", tick.dt)?;
    let params = match tick.params.is_empty() {
        true => PyDict::new(py).into_any(),
        false => py
            .import("json")?
            .call_method1("loads", (Value::Object(tick.params).to_string(),))?,
    };
    d.set_item("params", params)?;
    Ok(d)
}

/// What a drawing paints on, as the shim's drawing reads it.
fn drawing_dict(py: Python<'_>, at: session::Drawing) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("pass", at.pass)?;
    d.set_item("layers", at.layers)?;
    d.set_item("taken", at.taken)?;
    d.set_item("all", at.all)?;
    Ok(d)
}

/// The GLSL of the scene of a host that names none: one pass, black.
const BLACK: &str = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";

/// The scene named by a caller: a path, a JSON document, the JSON text of
/// one, a ghee.Scene, or None for one pass, `main`, that clears to black. A
/// document's files resolve in `base_dir`, else in the working directory.
fn source_of(scene: Option<&Bound<'_, PyAny>>, base_dir: Option<PathBuf>) -> PyResult<Source> {
    let dir = base_dir.map_or(".".to_string(), |d| d.to_string_lossy().into_owned());
    let doc = |doc: Value| Source::Doc {
        doc,
        dir: dir.clone(),
    };
    let Some(scene) = scene.filter(|s| !s.is_none()) else {
        return Ok(doc(serde_json::json!({
            "version": 1,
            "passes": [{"name": "main", "fragment": BLACK}],
        })));
    };
    if scene.cast::<PyDict>().is_ok() {
        return Ok(doc(to_json(scene)?));
    }
    let shim = scene.py().import("ghee._shim")?.getattr("Scene")?;
    if scene.is_instance(&shim)? {
        return Ok(doc(to_json(&scene.call_method0("to_dict")?)?));
    }
    if let Ok(s) = scene.extract::<String>() {
        if s.trim_start().starts_with('{') {
            let parsed: Value = serde_json::from_str(&s)
                .map_err(|e| PyValueError::new_err(format!("scene: invalid JSON: {e}")))?;
            return Ok(doc(parsed));
        }
        return Ok(Source::Path(s));
    }
    if let Ok(p) = scene.extract::<PathBuf>() {
        return Ok(Source::Path(p.to_string_lossy().into_owned()));
    }
    Err(PyValueError::new_err(format!(
        "scene takes a path, a JSON string, a dict or a ghee.Scene; got {}",
        scene.get_type().name()?
    )))
}

/// The caller's stream cap: pixels (an int >= 2), or a fraction of the render
/// size.
fn stream_max_of(v: &Bound<'_, PyAny>) -> PyResult<StreamMax> {
    if let Ok(n) = v.extract::<i64>() {
        if n >= 2 {
            return Ok(StreamMax::Pixels(n.min(i64::from(i32::MAX)) as i32));
        }
    }
    if let Ok(f) = v.extract::<f64>() {
        if f > 0.0 && f <= 1.0 {
            return Ok(StreamMax::Fraction(f));
        }
    }
    Err(PyValueError::new_err(
        "stream_max takes pixels (an int >= 2) or a fraction of the render size (0 < f <= 1)",
    ))
}

fn bind_of(serve: &Bound<'_, PyAny>) -> PyResult<(String, u16)> {
    if let Ok(port) = serve.extract::<i64>() {
        return Ok((serve::LOOPBACK.to_string(), port_of(port)?));
    }
    if let Ok((host, port)) = serve.extract::<(String, i64)>() {
        return Ok((host, port_of(port)?));
    }
    Err(PyValueError::new_err(format!(
        "serve takes a port, or a (host, port) pair; got {}",
        serve.get_type().name()?
    )))
}

fn port_of(port: i64) -> PyResult<u16> {
    u16::try_from(port)
        .map_err(|_| PyValueError::new_err(format!("serve: {port} is no port (0-65535)")))
}

/// A rendering scene, and the output it draws to.
#[pyclass(module = "ghee")]
struct Handle {
    /// (width, height) of the frame: of pixels() and of a saved frame
    #[pyo3(get)]
    size: (i32, i32),
    held: Affine<Session>,
    /// What the screen's drawing keeps from one frame to the next.
    screen: Option<Py<draw::Screen>>,
}

impl Handle {
    /// Give the layers of `calls` their looks, as Session::update_calls()
    /// does. Returns the indexes of the calls left for _update_layer().
    pub(crate) fn update_calls(&mut self, calls: &[ghee::draw::Call]) -> PyResult<Vec<usize>> {
        self.held.get()?.update_calls(calls).map_err(fault)
    }
}

#[pymethods]
impl Handle {
    /// Set one live param, with no recompile.
    ///
    /// `pass_` addresses the param of one pass. A dotted `pass.name` does the
    /// same. A bare name is the scene param, else the param of the first pass
    /// that declares it.
    #[pyo3(signature = (name, value, pass_=None))]
    fn set_param(
        &mut self,
        name: &str,
        value: &Bound<'_, PyAny>,
        pass_: Option<&str>,
    ) -> PyResult<()> {
        let value = to_json(value)?;
        self.held
            .get()?
            .set_param(pass_, name, &value)
            .map_err(fault)
    }

    /// Set a live field of the layer `name`, or of its effect `effect`, with
    /// no rebuild: `x`, `y`, `color`, `opacity` and `visible`, a text
    /// layer's `text`, the words, an image layer's `image`, the name of an
    /// image asset of the scene, and an effect's `visible`, `color`,
    /// `strength`, `sung`, `speed` and `rate`. `value` takes the form that
    /// the scene format gives the field, and None takes `x`, `y` or `color`
    /// back to its default.
    ///
    /// Raises ValueError for a layer, an effect or a field that the scene
    /// lacks, for a field that changes with a rebuild, and for a value that
    /// the field does not take.
    #[pyo3(signature = (name, field, value, effect=None))]
    fn set_layer(
        &mut self,
        name: &str,
        field: &str,
        value: &Bound<'_, PyAny>,
        effect: Option<&str>,
    ) -> PyResult<()> {
        let value = to_json(value)?;
        self.held
            .get()?
            .set_layer(name, effect, field, &value)
            .map_err(fault)
    }

    /// The scene's layers, to set their live fields as set_layer() does, by
    /// attribute or by key: `h.layers.title.x = 10`,
    /// `h.layers["title"]["karaoke"]["sung"] = 12`.
    #[getter]
    fn layers<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("ghee._shim")?
            .getattr("Layers")?
            .call1((slf.getattr("set_layer")?,))
    }

    /// A drawing of one frame, for a `with` block: its calls write layers,
    /// and leaving the block draws the frame.
    fn draw<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, draw::Drawing>> {
        let py = slf.py();
        let screen = {
            let mut h = slf.try_borrow_mut()?;
            h.held.get()?;
            match &h.screen {
                Some(s) => s.clone_ref(py),
                None => {
                    let s = Py::new(py, draw::Screen::default())?;
                    h.screen = Some(s.clone_ref(py));
                    s
                }
            }
        };
        let handle = slf.clone().into_any().unbind();
        Bound::new(py, draw::Drawing::of(py, handle, screen, true))
    }

    /// The width and height in pixels of the block that `words` lay out in,
    /// as a drawing's text lays them out: at `font_size`, or at the size
    /// that fits the screen when None, in `font_family`, the built-in font
    /// when None.
    #[pyo3(signature = (words, font_size=None, font_family=None))]
    fn measure(
        &mut self,
        words: &str,
        font_size: Option<f64>,
        font_family: Option<PathBuf>,
    ) -> PyResult<(f64, f64)> {
        let family = font_family.map(|f| f.to_string_lossy().into_owned());
        self.held
            .get()?
            .measure(words, font_size, family.as_deref())
            .map_err(fault)
    }

    /// The backlight level of the display, 0 to 1, None for a display with
    /// no backlight device. Setting it writes the level.
    #[getter]
    fn get_brightness(&mut self) -> PyResult<Option<f64>> {
        self.held.get()?.brightness().map_err(fault)
    }

    #[setter]
    fn set_brightness(&mut self, level: f64) -> PyResult<()> {
        self.held.get()?.set_brightness(level).map_err(fault)?;
        Ok(())
    }

    /// Open this iteration and give its tick, which the next frame() or
    /// skip() takes and returns: the shim's hooks run between the two.
    fn _tick<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let tick = self.held.get()?.begin().map_err(fault)?;
        tick_dict(py, tick)
    }

    /// The pass that a drawing paints on and its layers, every name that
    /// the scene holds, and every layer of the scene.
    fn _drawing<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        drawing_dict(py, self.held.get()?.drawing())
    }

    /// Add `layer`, a layer as a scene declares it, at `index` among the
    /// layers of pass `pass`, with no rebuild.
    fn _add_layer(&mut self, pass: &str, index: usize, layer: &Bound<'_, PyAny>) -> PyResult<()> {
        let layer = to_json(layer)?;
        self.held
            .get()?
            .add_layer(pass, index, &layer)
            .map_err(fault)
    }

    /// Give the layer of the name of `layer` the look of `layer`, live where
    /// the change is live and rebuilt alone otherwise. False when the layer
    /// had that look.
    fn _update_layer(&mut self, layer: &Bound<'_, PyAny>) -> PyResult<bool> {
        let layer = to_json(layer)?;
        self.held.get()?.update_layer(&layer).map_err(fault)
    }

    /// Hold data for one declared array. The next frame consumes it once.
    ///
    /// `data` is a uint8 buffer, read as byte / 255, or any other buffer or
    /// iterable of numbers. An array with no feed keeps what its texture holds.
    fn feed(&mut self, name: &str, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = fed(name, data)?;
        self.held.get()?.feed(name, data).map_err(fault)
    }

    /// Draw one frame and hand it to the output.
    ///
    /// Returns this tick: {"frame", "time", "dt", "params"}. `dt` is the
    /// iTimeDelta that the frame draws with, and `params` holds the params
    /// that changed since the last tick, a text layer's words as its `text`,
    /// the first tick all of them.
    fn frame<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut held = Held(self.held.get()?);
        let tick = py.detach(move || held.frame()).map_err(fault)?;
        tick_dict(py, tick)
    }

    /// Take this tick with no draw. The display keeps its last frame.
    ///
    /// Returns the same tick as frame().
    fn skip<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut held = Held(self.held.get()?);
        let tick = py.detach(move || held.skip()).map_err(fault)?;
        tick_dict(py, tick)
    }

    /// The last frame, row-major (height, width, components).
    fn pixels<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.held.get()?.pixels().map_err(fault)?;
        Ok(PyBytes::new(py, data))
    }

    /// Write the last frame to an image file: a .png, or a .jpg, which holds
    /// no alpha. A run of `#` or a printf field in `path` takes the number of
    /// that frame's tick, as `ghee render -o` numbers its files.
    fn save(&mut self, path: PathBuf) -> PyResult<()> {
        let path = path.to_string_lossy().into_owned();
        self.held.get()?.save(&path).map_err(fault)
    }

    /// The components each pixel of pixels() carries: 3 for RGB, 4 for RGBA.
    #[getter]
    fn components(&mut self) -> PyResult<i32> {
        Ok(self.held.get()?.components())
    }

    /// The GL that renders, GL_RENDERER and GL_VERSION in one string. A frame
    /// pinned by its sha holds on that GL alone.
    #[getter]
    fn _gl(&mut self) -> PyResult<String> {
        Ok(self.held.get()?.gl())
    }

    /// Drop everything the handle holds. A second call is no error.
    fn close(&mut self) -> PyResult<()> {
        if let Some(mut session) = self.held.take()? {
            session.close();
        }
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "<ghee {}x{}{}>",
            self.size.0,
            self.size.1,
            match self.held.held.is_some() {
                true => "",
                false => " closed",
            }
        )
    }
}

/// Open a scene and render it.
///
/// `scene` is a path to a .scene.json, the JSON text of a scene, a dict, or a
/// ghee.Scene. None makes one pass, `main`, that clears to black.
/// `display` is "kms" for the connected display. None renders headless, and
/// pixels() reads the frame back.
/// `opts` holds the display's options: {"rgb565": True} scans out in 16-bit
/// color, as `ghee render --rgb565` does.
/// `dri` is the card, as `--dri` takes it: with "kms", the card whose display
/// shows the frames, the first card with a display when None.
/// `rotate` turns the frame clockwise on its way out, 0, 90, 180 or 270
/// degrees, and at 90 and 270 the scene renders at the frame's size turned.
/// `width` and `height` size a headless frame. A display states its own
/// size.
/// `fps` caps the pace. None runs free.
/// `timestep` pins time to frame*S, so a render repeats bit for bit.
/// `alpha` keeps the output pass's alpha in pixels(), opaque without it.
/// `base_dir` is where a scene given as a dict, as JSON text or as a
/// ghee.Scene finds its files, the working directory when None.
/// `serve` serves the ghee console on that port, on 127.0.0.1. A (host, port)
/// pair binds it elsewhere. `stream_max` caps the console's h264 stream on
/// its longest side: pixels, or a fraction of the render size.
#[pyfunction]
#[pyo3(signature = (
    scene=None,
    display=None,
    opts=None,
    dri=None,
    rotate=0,
    width=None,
    height=None,
    fps=None,
    serve=None,
    stream_max=None,
    verbose=false,
    timestep=None,
    alpha=false,
    base_dir=None,
))]
#[allow(clippy::too_many_arguments)]
fn open(
    scene: Option<&Bound<'_, PyAny>>,
    display: Option<String>,
    opts: Option<&Bound<'_, PyAny>>,
    dri: Option<PathBuf>,
    rotate: i64,
    width: Option<i32>,
    height: Option<i32>,
    fps: Option<f64>,
    serve: Option<&Bound<'_, PyAny>>,
    stream_max: Option<&Bound<'_, PyAny>>,
    verbose: bool,
    timestep: Option<f64>,
    alpha: bool,
    base_dir: Option<PathBuf>,
) -> PyResult<Handle> {
    let source = source_of(scene, base_dir)?;
    let serve = serve.map(bind_of).transpose()?;
    let stream_max = stream_max.map(stream_max_of).transpose()?;
    let opts = match opts {
        None => Vec::new(),
        Some(d) => match to_json(d)? {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(PyValueError::new_err(
                    "opts takes a dict of display options",
                ))
            }
        },
    };
    let size = match (width, height) {
        (Some(w), Some(h)) => Some((w, h)),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "width and height go together; pass both or neither",
            ))
        }
    };
    let session = Session::open(Opening {
        source,
        display,
        opts,
        dri: dri.map(|d| d.to_string_lossy().into_owned()),
        rotate,
        size,
        fps,
        timestep,
        serve,
        stream_max,
        verbose,
        alpha,
    })
    .map_err(fault)?;
    Ok(Handle {
        size: session.size(),
        held: Affine::new(session),
        screen: None,
    })
}

/// The mode of the connected display on card `dri`, else on the first card
/// with one, as open() takes it with display="kms", worked out before
/// anything opens it. None for no connected display.
#[pyfunction]
#[pyo3(signature = (dri=None))]
fn _display_size(dri: Option<PathBuf>) -> PyResult<Option<(i32, i32)>> {
    let dri = dri.map(|d| d.to_string_lossy().into_owned());
    Ok(ghee::engine::probe_display(dri.as_deref()))
}

/// True when the image file name `pattern` numbers its frames: a run of `#`
/// or a printf field.
#[pyfunction]
fn _numbered(pattern: &str) -> PyResult<bool> {
    let first = ghee::output::expand_pattern(pattern, 0)
        .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
    Ok(first != pattern)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // the ghee package imports this module on the first open(), and re-exports
    // what __all__ names
    m.add("__all__", vec!["Handle", "open", "__version__"])?;
    m.add_class::<Handle>()?;
    // the shim's own, so out of __all__
    m.setattr("Drawing", m.py().get_type::<draw::Drawing>())?;
    m.setattr("_Screen", m.py().get_type::<draw::Screen>())?;
    m.setattr("_LayerWire", m.py().get_type::<wire::LayerWire>())?;
    m.setattr("_colour_name", wrap_pyfunction!(wire::_colour_name, m)?)?;
    m.setattr("_numbered", wrap_pyfunction!(_numbered, m)?)?;
    m.setattr("_display_size", wrap_pyfunction!(_display_size, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    Ok(())
}
