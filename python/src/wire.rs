//! The live layer sets of a scene process that ghee runs, as the shim writes
//! them on the wire.
//!
//! The shim holds a LayerWire for the scene that it hands ghee. Each set goes
//! through the scene's own checks there, so a set that ghee would refuse
//! raises in the scene process, as it does on a handle.

use ghee::console::server::coerce;
use ghee::layer::{Address, Color, Form, Rgb};
use ghee::scene::{ParamValue, Scene};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use serde_json::Value;

use crate::session::{Fault, Index, Outcome};
use crate::{fault, to_json};

/// A scene that a scene process runs under ghee, to check each live layer set
/// against and to give the set's message.
#[pyclass(module = "ghee", name = "_LayerWire", unsendable)]
pub struct LayerWire {
    scene: Scene,
    index: Index,
}

/// The body of a message: the floats of kind 1, or the text of kind 3.
enum Body {
    Floats(Vec<f64>),
    Text(String),
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
            (Form::Image, Value::String(name)) => return Ok((wire, Body::Text(name.clone()))),
            (_, Value::Null) => vec![],
            (Form::Flag, v) => vec![if v.as_bool() == Some(true) { 1.0 } else { 0.0 }],
            (Form::Colour { .. }, v) => match (Rgb::from_json(v), set.as_text()) {
                (Some(c), _) => c.rgb.to_vec(),
                (None, Some(text)) => match text.color {
                    Color::Gradient { from, to, angle } => {
                        [&from.rgb[..], &to.rgb, &[angle]].concat()
                    }
                    Color::Solid(c) => c.rgb.to_vec(),
                },
                (None, None) => unreachable!("an image layer took its tint as one colour"),
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
        Ok(LayerWire { scene, index })
    }

    /// The message of a live set of `field` of layer `name`, or of its
    /// effect `effect`: its wire name, and its floats for kind 1 or its
    /// text for kind 3. Raises ValueError as Handle.set_layer does.
    #[pyo3(signature = (name, field, value, effect=None))]
    fn message<'py>(
        &mut self,
        py: Python<'py>,
        name: &str,
        field: &str,
        value: &Bound<'py, PyAny>,
        effect: Option<&str>,
    ) -> PyResult<(String, Bound<'py, PyAny>)> {
        let value = to_json(value)?;
        let (wire, body) = self.set(name, effect, field, &value).map_err(fault)?;
        let body = match body {
            Body::Floats(floats) => PyList::new(py, floats)?.into_any(),
            Body::Text(text) => PyString::new(py, &text).into_any(),
        };
        Ok((wire, body))
    }
}
