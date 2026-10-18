//! The live layer sets and the drawing of a scene process that ghee runs, as
//! the shim writes them on the wire.
//!
//! The shim holds a LayerWire for the scene that it hands ghee. Each set and
//! each layer that the drawing adds or gives a new look goes through the
//! scene's own checks there, so one that ghee would refuse raises in the
//! scene process, as it does on a handle, and the scene there grows as the
//! scene in ghee does.

use std::collections::HashMap;

use ghee::draw::{self, Call};
use ghee::layer::{Address, Form, Layer, Rgb};
use ghee::live::Index;
use ghee::scene::Scene;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use serde_json::Value;

use crate::session::{drawing, Fault, Outcome};
use crate::{drawing_dict, fault, to_json};

/// A scene that a scene process runs under ghee, to check each live layer set
/// against and to give the set's message, and to lay words out in.
#[pyclass(module = "ghee", name = "_LayerWire", unsendable)]
pub struct LayerWire {
    scene: Scene,
    index: Index,
    /// The scene as the process handed it, for measure(). The drawing grows
    /// `scene` alone, so a file of a drawn layer never loads here.
    handed: Scene,
    /// `handed` with its files loaded, once measure() first asks.
    loaded: Option<Scene>,
    /// The pass and the place among its layers of each layer of `scene`.
    places: HashMap<String, (usize, usize)>,
}

/// The kind of the message that gives several layers their looks.
const UPDATE_LAYERS: u8 = 8;

/// The pass and the place among its layers of each layer of `scene`.
fn places(scene: &Scene) -> HashMap<String, (usize, usize)> {
    let mut at = HashMap::new();
    for (p, pass) in scene.passes.iter().enumerate() {
        for (i, layer) in pass.layers.iter().enumerate() {
            at.insert(layer.name().to_string(), (p, i));
        }
    }
    at
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
        self.index.layer(layer)?;
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
            (Form::Image | Form::Colour { .. } | Form::Words, Value::String(text)) => {
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

    /// Index the scene again when layer `name` joined it or changed between
    /// a text layer and another kind.
    fn took(&mut self, name: &str) {
        let text = self.scene.layer(name).map(|l| l.as_text().is_some());
        if self.index.text(name) != text {
            self.index = Index::of(&self.scene);
        }
    }

    /// The kind 8 message that gives the layers of `calls` the looks of the
    /// calls whose looks differ in live fields alone, as Session gives them
    /// to the renderer: the kind, no name and the JSON text of the looks,
    /// None when no such call changes its layer. The scene takes those
    /// looks. Returns the indexes of the other calls too, for update(),
    /// which says why the scene refuses one.
    pub(crate) fn draw(&mut self, calls: &[Call]) -> (Option<(u8, String, String)>, Vec<usize>) {
        let mut looks = Vec::new();
        let mut left = Vec::new();
        for (k, call) in calls.iter().enumerate() {
            let Some(&(p, i)) = self.places.get(&call.layer) else {
                left.push(k);
                continue;
            };
            match draw::live_of(call, &self.scene.passes[p].layers[i]) {
                Some(None) => {}
                Some(Some(new)) => {
                    self.scene.passes[p].layers[i] = new;
                    looks.push(draw::look_of(call));
                }
                None => left.push(k),
            }
        }
        let message = (!looks.is_empty()).then(|| {
            (
                UPDATE_LAYERS,
                String::new(),
                Value::Array(looks).to_string(),
            )
        });
        (message, left)
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
            handed: scene.clone(),
            places: places(&scene),
            scene,
            index,
            loaded: None,
        })
    }

    /// The names of the scene's layers, those that the drawing added
    /// included.
    fn names(&self) -> Vec<String> {
        self.index.layers.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The wire name and the type of each param that a set can name, as
    /// ghee addresses them.
    fn params(&self) -> Vec<(String, String)> {
        self.index
            .params
            .iter()
            .map(|r| (r.wire.clone(), r.ty.clone()))
            .collect()
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
        font_family: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(f64, f64)> {
        let family = crate::draw::family(font_family.as_ref())?;
        let family: Vec<&str> = family.iter().map(String::as_str).collect();
        if self.loaded.is_none() {
            let mut loaded = self.handed.clone();
            loaded
                .load_assets(base_dir)
                .map_err(scene_fault)
                .map_err(fault)?;
            self.loaded = Some(loaded);
        }
        let scene = self.loaded.as_ref().expect("loaded just now");
        let [w, h] = scene
            .measure([size.0, size.1], words, font_size, &family)
            .map_err(scene_fault)
            .map_err(fault)?;
        Ok((w, h))
    }

    /// The pass that a drawing paints on and its layers, every name that the
    /// scene holds, and every layer of the scene, as Handle._drawing() gives
    /// them.
    fn drawing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        drawing_dict(py, drawing(&self.scene))
    }

    /// The message that adds `layer`, a layer as a scene declares it, at
    /// `index` among the layers of pass `pass`: kind 6, the pass, and its
    /// JSON text. Raises ValueError as Handle._add_layer() does.
    fn add(
        &mut self,
        pass: &str,
        index: usize,
        layer: &Bound<'_, PyAny>,
    ) -> PyResult<(u8, String, String)> {
        let layer = to_json(layer)?;
        let at = format!("the layer added at {index} to pass '{pass}'");
        let new = Layer::from_dict(&at, &layer)
            .map_err(scene_fault)
            .map_err(fault)?;
        let name = new.name().to_string();
        self.scene
            .insert_layer(pass, index, new)
            .map_err(scene_fault)
            .map_err(fault)?;
        self.places = places(&self.scene);
        self.took(&name);
        let text = serde_json::json!({"index": index, "layer": layer}).to_string();
        Ok((6, pass.to_string(), text))
    }

    /// The message that gives the layer of the name of `layer` the look of
    /// `layer`, a layer as a scene declares it: kind 7, the layer's name,
    /// and its JSON text, None when the layer has that look already. Raises
    /// ValueError as Handle._update_layer() does.
    fn update(&mut self, layer: &Bound<'_, PyAny>) -> PyResult<Option<(u8, String, String)>> {
        let look = to_json(layer)?;
        let changed = self
            .scene
            .update_layer(&look)
            .map_err(scene_fault)
            .map_err(fault)?;
        if !changed {
            return Ok(None);
        }
        let name = look["name"].as_str().unwrap_or_default().to_string();
        self.took(&name);
        Ok(Some((7, name, look.to_string())))
    }

    /// The message that takes layer `name` out of the scene: kind 9, the
    /// layer's name, and no text. Raises ValueError for a name that names
    /// no layer.
    fn remove(&mut self, name: &str) -> PyResult<(u8, String, String)> {
        self.scene
            .remove_layer(name)
            .map_err(scene_fault)
            .map_err(fault)?;
        self.places = places(&self.scene);
        self.index = Index::of(&self.scene);
        Ok((9, name.to_string(), String::new()))
    }

    /// The message that puts layer `name` at `index` among the layers of
    /// its pass: kind 10, the layer's name, and the JSON text of its place.
    /// Raises ValueError as Handle._move_layer() does.
    #[pyo3(name = "move")]
    fn move_layer(&mut self, name: &str, index: usize) -> PyResult<(u8, String, String)> {
        self.scene
            .move_layer(name, index)
            .map_err(scene_fault)
            .map_err(fault)?;
        self.places = places(&self.scene);
        let text = serde_json::json!({ "index": index }).to_string();
        Ok((10, name.to_string(), text))
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
