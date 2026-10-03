//! ghee for Python: open a scene, feed it, ask for frames.
//!
//! ```python
//! import ghee
//!
//! with ghee.open("scene.json", output="st7789", fps=30) as h:
//!     while playing:
//!         h.set_param("gain", 1.5)
//!         h.feed("level", bars)
//!         h.frame()
//! ```
//!
//! The engine is thread-affine. Open a handle on one thread, and use it on that
//! same thread.

mod serve;
mod session;

use std::path::PathBuf;
use std::thread::{self, ThreadId};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyString, PyTuple};
use serde_json::{Map, Value};

use session::{Fault, Fed, Opening, Session, Source, StreamMax, Tick};

/// A scene error is the caller's text: ValueError. A driver or system error is
/// the host's: RuntimeError.
fn fault(f: Fault) -> PyErr {
    match f.scene {
        true => PyValueError::new_err(f.message),
        false => PyRuntimeError::new_err(f.message),
    }
}

/// A value that belongs to one thread.
///
/// The engine makes its GL context current on the thread that built it, and a
/// panel driver binds its SPI and GPIO handles to that thread. Neither closes
/// anywhere else.
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
            // The GL context and the panel handles are this thread's to close.
            // A drop anywhere else leaks them instead.
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

    fn skip(&mut self) -> Tick {
        self.0.skip()
    }
}

/// A Python value as JSON. A param value and the driver options travel this
/// way.
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

/// The values in a feed. A uint8 buffer reaches the texture as byte / 255,
/// and a float32 buffer passes through. Any other sequence of numbers converts
/// to float32.
///
/// The buffer protocol is no part of the stable ABI, so `bytes` reads straight
/// out and any other exporter goes through a memoryview.
fn fed(data: &Bound<'_, PyAny>) -> PyResult<Fed> {
    if let Ok(b) = data.cast::<PyBytes>() {
        return Ok(Fed::U8(b.as_bytes().to_vec()));
    }
    let view = data
        .py()
        .import("builtins")?
        .getattr("memoryview")?
        .call1((data,));
    if let Ok(view) = view {
        return from_view(&view);
    }
    match data.extract::<Vec<f32>>() {
        Ok(values) => Ok(Fed::F32(values)),
        Err(_) => Err(PyValueError::new_err(format!(
            "a feed takes a buffer of uint8 or float32 values, or numbers; got {}",
            data.get_type().name()?
        ))),
    }
}

fn from_view(view: &Bound<'_, PyAny>) -> PyResult<Fed> {
    if !view.getattr("c_contiguous")?.extract::<bool>()? {
        return Err(PyValueError::new_err("a feed takes a contiguous buffer"));
    }
    let format: String = view.getattr("format")?.extract()?;
    let raw = view.call_method0("tobytes")?;
    let bytes = raw.cast::<PyBytes>()?;
    let bytes = bytes.as_bytes();
    match format.as_str() {
        "B" => Ok(Fed::U8(bytes.to_vec())),
        "f" => Ok(Fed::F32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )),
        other => Err(PyValueError::new_err(format!(
            "a feed takes a uint8 or float32 buffer; this one is '{other}'"
        ))),
    }
}

fn tick_dict(py: Python<'_>, tick: Tick) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frame", tick.frame)?;
    d.set_item("time", tick.time)?;
    d.set_item("dt", tick.dt)?;
    Ok(d)
}

/// The scene named by a caller: a path, a JSON document, the JSON text of
/// one, or a ghee.Scene. A document's files resolve in `base_dir`, else in
/// the working directory.
fn source_of(scene: &Bound<'_, PyAny>, base_dir: Option<PathBuf>) -> PyResult<Source> {
    let dir = base_dir.map_or(".".to_string(), |d| d.to_string_lossy().into_owned());
    let doc = |doc: Value| Source::Doc {
        doc,
        dir: dir.clone(),
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

fn bind_of(stats: &Bound<'_, PyAny>) -> PyResult<(String, u16)> {
    if let Ok(port) = stats.extract::<i64>() {
        return Ok((serve::LOOPBACK.to_string(), port_of(port)?));
    }
    if let Ok((host, port)) = stats.extract::<(String, i64)>() {
        return Ok((host, port_of(port)?));
    }
    Err(PyValueError::new_err(format!(
        "stats takes a port, or a (host, port) pair; got {}",
        stats.get_type().name()?
    )))
}

fn port_of(port: i64) -> PyResult<u16> {
    u16::try_from(port)
        .map_err(|_| PyValueError::new_err(format!("stats: {port} is no port (0-65535)")))
}

/// A rendering scene, and the output it draws to.
#[pyclass(module = "ghee")]
struct Handle {
    /// (width, height) of the scene's render
    #[pyo3(get)]
    size: (i32, i32),
    held: Affine<Session>,
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

    /// Hold data for one declared array. The next frame consumes it once.
    ///
    /// An array with no feed keeps what its texture holds.
    fn feed(&mut self, name: &str, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = fed(data)?;
        self.held.get()?.feed(name, data).map_err(fault)
    }

    /// Draw one frame and hand it to the output.
    ///
    /// Returns this tick: {"frame", "time", "dt"}.
    fn frame<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut held = Held(self.held.get()?);
        let tick = py.detach(move || held.frame()).map_err(fault)?;
        tick_dict(py, tick)
    }

    /// Take this tick with no draw. The panel keeps its last frame.
    ///
    /// Returns the same tick as frame().
    fn skip<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut held = Held(self.held.get()?);
        let tick = py.detach(move || held.skip());
        tick_dict(py, tick)
    }

    /// The last frame, row-major (height, width, components).
    fn pixels<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.held.get()?.pixels().map_err(fault)?;
        Ok(PyBytes::new(py, data))
    }

    /// The components each pixel of pixels() carries: 3 for RGB, 4 for RGBA.
    #[getter]
    fn components(&mut self) -> PyResult<i32> {
        Ok(self.held.get()?.components())
    }

    /// Blank the panel and drop everything the handle holds. A second call is
    /// no error.
    fn close(&mut self) -> PyResult<()> {
        match self.held.take()? {
            Some(mut session) => session.close().map_err(fault),
            None => Ok(()),
        }
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
/// ghee.Scene.
/// `output` names a display driver as `ghee render --display` takes it, or
/// "kms" for the connected display. None renders headless, and pixels() reads
/// the frame back.
/// `opts` holds the driver options, as `--display-opts` takes them.
/// `width` and `height` size a headless render. A panel or a display states its
/// own size.
/// `fps` caps the pace. None runs free.
/// `timestep` pins time to frame*S, so a render repeats bit for bit.
/// `alpha` keeps the output pass's alpha in pixels(), opaque without it.
/// `base_dir` is where a scene given as a dict, as JSON text or as a
/// ghee.Scene finds its files, the working directory when None.
/// `stats` serves the ghee console on that port, on 127.0.0.1. A (host, port)
/// pair binds it elsewhere. `stream_max` caps the console's h264 stream on
/// its longest side: pixels, or a fraction of the render size.
#[pyfunction]
#[pyo3(signature = (
    scene,
    output=None,
    opts=None,
    width=None,
    height=None,
    fps=None,
    stats=None,
    stream_max=None,
    verbose=false,
    timestep=None,
    alpha=false,
    base_dir=None,
))]
#[allow(clippy::too_many_arguments)]
fn open(
    scene: &Bound<'_, PyAny>,
    output: Option<String>,
    opts: Option<&Bound<'_, PyAny>>,
    width: Option<i32>,
    height: Option<i32>,
    fps: Option<f64>,
    stats: Option<&Bound<'_, PyAny>>,
    stream_max: Option<&Bound<'_, PyAny>>,
    verbose: bool,
    timestep: Option<f64>,
    alpha: bool,
    base_dir: Option<PathBuf>,
) -> PyResult<Handle> {
    let source = source_of(scene, base_dir)?;
    let stats = stats.map(bind_of).transpose()?;
    let stream_max = stream_max.map(stream_max_of).transpose()?;
    let opts = match opts {
        None => Vec::new(),
        Some(d) => match to_json(d)? {
            Value::Object(map) => map.into_iter().collect(),
            _ => return Err(PyValueError::new_err("opts takes a dict of driver options")),
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
        output,
        opts,
        size,
        fps,
        timestep,
        stats,
        stream_max,
        verbose,
        alpha,
    })
    .map_err(fault)?;
    Ok(Handle {
        size: session.size(),
        held: Affine::new(session),
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // the ghee package imports this module on the first open(), and re-exports
    // what __all__ names
    m.add("__all__", vec!["Handle", "open", "__version__"])?;
    m.add_class::<Handle>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    Ok(())
}
