//! The live layer sets of a scene process that ghee runs, as the shim writes
//! them on the wire.
//!
//! The shim holds a LayerWire for the scene that it hands ghee. Each set goes
//! through the scene's own checks there, so a set that ghee would refuse
//! raises in the scene process, as it does on a handle.

use ghee::console::server::coerce;
use ghee::layer::{Address, Form, Rgb};
use ghee::scene::{ParamValue, Scene};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use serde_json::Value;

use crate::session::{Fault, Index, Outcome};
use crate::{fault, to_json};

/// A scene that a scene process runs under ghee, to check each live layer set
/// against and to give the set's message, and to lay words out in.
#[pyclass(module = "ghee", name = "_LayerWire", unsendable)]
pub struct LayerWire {
    scene: Scene,
    index: Index,
    /// The scene with its files loaded, for measure(), once it first asks.
    loaded: Option<Scene>,
}

/// True for a CSS colour name, in any case, `transparent` too.
#[pyfunction]
pub fn _colour_name(text: &str) -> bool {
    ghee::css::named(text).is_some()
}

/// The body of a message: the floats of kind 1, the text of kind 3, or the
/// array of kind 0.
enum Body {
    Floats(Vec<f64>),
    Text(String),
    Array(Vec<f64>),
}

fn scene_fault(e: anyhow::Error) -> Fault {
    Fault::scene(format!("{e:#}"))
}

impl LayerWire {
    /// The set of `field` of layer `layer`, or of its effect `effect`, to
    /// `value`, checked as Session::set_layer checks it, as its wire name and
    /// its body.
    fn set(
        &mut self,
        layer: &str,
        effect: Option<&str>,
        field: &str,
        value: &Value,
    ) -> Outcome<(String, Body)> {
        if self.index.words(layer, effect, field)? {
            let p = self.index.param(None, layer)?;
            return match coerce(&p.wire, &p.ty, value).map_err(scene_fault)? {
                ParamValue::Str(words) => Ok((p.wire.clone(), Body::Text(words))),
                _ => unreachable!("the words param of a text layer is a string"),
            };
        }
        let at = Address {
            layer,
            effect,
            field,
        };
        self.scene.set_layer(&at, value).map_err(scene_fault)?;
        let set = self.scene.layer(layer).expect("set_layer found the layer");
        let live = set.live(effect, field).map_err(scene_fault)?;
        let wire = match effect {
            Some(effect) => format!("{layer}.{effect}.{field}"),
            None => format!("{layer}.{field}"),
        };
        let floats = match (live.form, value) {
            // A colour's text keeps its form, and transparent its alpha.
            (Form::Image | Form::Colour { .. }, Value::String(text)) => {
                return Ok((wire, Body::Text(text.clone())))
            }
            // A gradient goes as its JSON, so its ends keep theirs.
            (Form::Colour { .. }, v @ Value::Object(_)) => {
                return Ok((wire, Body::Text(v.to_string())))
            }
            // A line's points go as an array, x and y in turn.
            (Form::Points, Value::Array(points)) => {
                let flat = points
                    .iter()
                    .flat_map(|p| p.as_array().into_iter().flatten())
                    .filter_map(Value::as_f64)
                    .collect();
                return Ok((wire, Body::Array(flat)));
            }
            (_, Value::Null) => vec![],
            (Form::Flag, v) => vec![if v.as_bool() == Some(true) { 1.0 } else { 0.0 }],
            (Form::Colour { .. }, v) => match Rgb::from_json(v) {
                Some(c) => c.rgb.to_vec(),
                None => unreachable!("the layer took the colour"),
            },
            (_, v) => vec![v.as_f64().expect("the layer took a number")],
        };
        Ok((wire, Body::Floats(floats)))
    }
}

#[pymethods]
impl LayerWire {
    /// `scene` is the scene that the process hands ghee, as a dict.
    #[new]
    fn new(scene: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scene = Scene::from_dict(&to_json(scene)?, None, None)
            .map_err(scene_fault)
            .map_err(fault)?;
        let index = Index::of(&scene);
        Ok(LayerWire {
            scene,
            index,
            loaded: None,
        })
    }

    /// The width and height in pixels of the block that `words` lay out in
    /// on the scene's last pass in a frame of `size` pixels, as
    /// Scene::measure() gives them, the scene's files found in `base_dir`.
    #[pyo3(signature = (words, size, base_dir, font_size=None, font_family=None))]
    fn measure(
        &mut self,
        words: &str,
        size: (f64, f64),
        base_dir: &str,
        font_size: Option<f64>,
        font_family: Option<&str>,
    ) -> PyResult<(f64, f64)> {
        if self.loaded.is_none() {
            let mut loaded = self.scene.clone();
            loaded
                .load_assets(base_dir)
                .map_err(scene_fault)
                .map_err(fault)?;
            self.loaded = Some(loaded);
        }
        let scene = self.loaded.as_ref().expect("loaded just now");
        let [w, h] = scene
            .measure([size.0, size.1], words, font_size, font_family)
            .map_err(scene_fault)
            .map_err(fault)?;
        Ok((w, h))
    }

    /// The message of a live set of `field` of layer `name`, or of its
    /// effect `effect`: its kind, its wire name, and its floats for kind 0
    /// or 1, or its text for kind 3. Raises ValueError as Handle.set_layer
    /// does.
    #[pyo3(signature = (name, field, value, effect=None))]
    fn message<'py>(
        &mut self,
        py: Python<'py>,
        name: &str,
        field: &str,
        value: &Bound<'py, PyAny>,
        effect: Option<&str>,
    ) -> PyResult<(u8, String, Bound<'py, PyAny>)> {
        let value = to_json(value)?;
        let (wire, body) = self.set(name, effect, field, &value).map_err(fault)?;
        let (kind, body) = match body {
            Body::Floats(floats) => (1, PyList::new(py, floats)?.into_any()),
            Body::Text(text) => (3, PyString::new(py, &text).into_any()),
            Body::Array(floats) => (0, PyList::new(py, floats)?.into_any()),
        };
        Ok((kind, wire, body))
    }
}
