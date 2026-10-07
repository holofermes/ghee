//! A screen's drawing: one frame's calls, each a layer, kept from frame to
//! frame.
//!
//! A Drawing records the calls of one frame, the block of `with
//! screen.draw() as d:`. A call keeps its layer from frame to frame by the
//! line of code that makes it and its count from that line in the frame, or
//! by its name. A Screen holds what the drawings of one screen keep between
//! frames: the layer of each call, and the values that each layer shows. A
//! call whose values are those that its layer shows holds nothing more.
//!
//! Leaving the block sends ghee what changed through the handle that the
//! drawing draws on. A call new to the screen adds its layer, written as the
//! scene format writes it by the shim's `_call_look`. The changed calls of
//! the layers that the screen holds go out at once, typed: to the session
//! itself on a native handle, through `_draw()` on a handle that has one, and
//! as looks through `_update_layer()` on any other, which also takes the
//! calls that the typed path leaves. A layer whose call did not run hides
//! through its live eye.

use std::collections::{HashMap, HashSet};

use ghee::draw::{Arg, Call, CallKind};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::Handle;

/// `v` as an arg. None for a value of a type that a look writes in another
/// form, such as a path or a numpy array, or that it refuses.
pub fn arg(v: &Bound<'_, PyAny>) -> Option<Arg> {
    if v.is_none() {
        return Some(Arg::None);
    }
    // a Python bool is an int, so it answers first
    if let Ok(b) = v.cast::<PyBool>() {
        return Some(Arg::Bool(b.is_true()));
    }
    if let Ok(f) = v.cast::<PyFloat>() {
        return Some(Arg::num(f.value()));
    }
    if v.cast::<PyInt>().is_ok() {
        return v.extract::<f64>().ok().map(Arg::num);
    }
    if let Ok(s) = v.cast::<PyString>() {
        return Some(Arg::Str(s.to_str().ok()?.to_string()));
    }
    if let Ok(t) = v.cast::<PyTuple>() {
        return t
            .iter()
            .map(|x| arg(&x))
            .collect::<Option<_>>()
            .map(Arg::List);
    }
    if let Ok(l) = v.cast::<PyList>() {
        return l
            .iter()
            .map(|x| arg(&x))
            .collect::<Option<_>>()
            .map(Arg::List);
    }
    if let Ok(d) = v.cast::<PyDict>() {
        let pairs = d.iter().map(|(k, x)| {
            let k = k.cast::<PyString>().ok()?.to_str().ok()?.to_string();
            Some((k, arg(&x)?))
        });
        return pairs.collect::<Option<_>>().map(Arg::Map);
    }
    None
}

/// True when arg() makes `a` of `v`, with no arg made.
fn same(a: &Arg, v: &Bound<'_, PyAny>) -> bool {
    match a {
        Arg::None => v.is_none() || v.cast::<PyFloat>().is_ok_and(|f| !f.value().is_finite()),
        Arg::Bool(b) => v.cast::<PyBool>().is_ok_and(|x| x.is_true() == *b),
        Arg::Num(n) => {
            if let Ok(f) = v.cast::<PyFloat>() {
                return f.value() == *n;
            }
            v.cast::<PyBool>().is_err()
                && v.cast::<PyInt>().is_ok()
                && v.extract::<f64>().is_ok_and(|f| f == *n)
        }
        Arg::Str(s) => v
            .cast::<PyString>()
            .is_ok_and(|x| x.to_str().is_ok_and(|t| t == s)),
        Arg::List(items) => {
            if let Ok(t) = v.cast::<PyTuple>() {
                return t.len() == items.len()
                    && items.iter().zip(t.iter()).all(|(a, x)| same(a, &x));
            }
            if let Ok(l) = v.cast::<PyList>() {
                return l.len() == items.len()
                    && items.iter().zip(l.iter()).all(|(a, x)| same(a, &x));
            }
            false
        }
        Arg::Map(pairs) => {
            let Ok(d) = v.cast::<PyDict>() else {
                return false;
            };
            d.len() == pairs.len()
                && d.iter().zip(pairs).all(|((k, x), (key, a))| {
                    k.cast::<PyString>()
                        .is_ok_and(|k| k.to_str().is_ok_and(|k| k == key))
                        && same(a, &x)
                })
        }
    }
}

/// The shim's number for a kind of call.
fn number_of(kind: CallKind) -> i64 {
    match kind {
        CallKind::Text => 0,
        CallKind::Rect => 1,
        CallKind::Image => 2,
        CallKind::Line => 3,
    }
}

/// The name of a new layer of a call that names none, by its kind.
fn base_of(kind: CallKind) -> &'static str {
    match kind {
        CallKind::Text => "text",
        CallKind::Rect => "rect",
        CallKind::Image => "image",
        CallKind::Line => "line",
    }
}

/// The place of `visible` among the values of a call of `kind`.
fn visible_at(kind: CallKind) -> usize {
    kind.keys()
        .iter()
        .position(|k| *k == "visible")
        .expect("every kind of call takes visible")
}

/// A call by its line and its count from that line in the frame.
type Key = (u32, u32);

/// The values that a layer shows from the call that drew it: None when a
/// later frame cannot compare its own with them.
struct Last {
    layer: u32,
    args: Option<Vec<Arg>>,
}

/// What the screen holds for a call, by its line and its count.
#[derive(Default)]
struct Slot {
    /// The layer of a call that names none, once added.
    named: Option<u32>,
    last: Option<Last>,
}

/// What the screen holds for a layer, by its number.
#[derive(Default)]
struct Held {
    name: String,
    /// The call whose values the layer shows.
    owner: Option<Key>,
    /// True once the drawing drew the layer.
    drawn: bool,
    /// True while the layer shows, as the last call that drew it says.
    visible: bool,
    /// The number of the last frame that drew the layer.
    frame: u64,
    /// The layer's place: the drawing's pass holds it, the scene holds it.
    placed: bool,
    in_scene: bool,
    /// The look last given to the layer through a handle with no _draw().
    look: Option<Py<PyAny>>,
}

/// What a screen's drawing keeps from one frame to the next.
#[pyclass(module = "ghee", name = "_Screen")]
#[derive(Default)]
pub struct Screen {
    /// The line of each call, by its caller's code object and the offset of
    /// the call's instruction in it. Every code object stays, so no other
    /// takes its address.
    sites: HashMap<(usize, i32), u32>,
    lines: HashMap<(String, i32), u32>,
    codes: Vec<Py<PyAny>>,
    /// Each call's slot, by its line, then by its count.
    slots: Vec<Vec<Slot>>,
    /// Each layer by its number, and the number of each name.
    held: Vec<Held>,
    numbers: HashMap<String, u32>,
    /// Every layer that the drawing drew, in the order of its first frame.
    order: Vec<u32>,
    /// The count of the drawings made.
    frame: u64,
    /// The pass that the drawing paints on and its layers in their order,
    /// the scene's own count of them, and every name that the scene holds.
    /// The first drawing reads them.
    pass: Option<String>,
    placed: Vec<u32>,
    own: usize,
    taken: HashSet<String>,
}

#[pymethods]
impl Screen {
    #[new]
    fn new() -> Screen {
        Screen::default()
    }
}

impl Screen {
    /// The number of layer `name`.
    fn number(&mut self, name: &str) -> u32 {
        if let Some(&n) = self.numbers.get(name) {
            return n;
        }
        let n = self.held.len() as u32;
        self.held.push(Held {
            name: name.to_string(),
            ..Held::default()
        });
        self.numbers.insert(name.to_string(), n);
        n
    }

    fn slot(&mut self, (site, count): Key) -> &mut Slot {
        let site = &mut self.slots[site as usize];
        if site.len() <= count as usize {
            site.resize_with(count as usize + 1, Slot::default);
        }
        &mut site[count as usize]
    }

    /// The number of the line of the call that the running Python code
    /// makes.
    fn site(&mut self, py: Python<'_>) -> PyResult<u32> {
        // SAFETY: the frame is the interpreter's own while this thread holds
        // the GIL, and PyFrame_GetCode gives a new reference.
        let (frame, code) = unsafe {
            let f = ffi::PyEval_GetFrame();
            if f.is_null() {
                return Ok(self.line_of(String::new(), 0));
            }
            (
                Bound::from_borrowed_ptr(py, f.cast::<ffi::PyObject>()),
                Bound::from_owned_ptr(py, ffi::PyFrame_GetCode(f).cast::<ffi::PyObject>()),
            )
        };
        // PyFrame_GetLineNumber walks the code's line table from its start,
        // so a known instruction skips it.
        let offset: i32 = frame.getattr(pyo3::intern!(py, "f_lasti"))?.extract()?;
        let at = (code.as_ptr() as usize, offset);
        if let Some(&site) = self.sites.get(&at) {
            return Ok(site);
        }
        // SAFETY: as above, `frame` holds the frame alive.
        let line = unsafe { ffi::PyFrame_GetLineNumber(frame.as_ptr().cast()) };
        let file: String = code.getattr("co_filename")?.extract()?;
        let site = self.line_of(file, line);
        self.sites.insert(at, site);
        self.codes.push(code.unbind());
        Ok(site)
    }

    fn line_of(&mut self, file: String, line: i32) -> u32 {
        let next = self.lines.len() as u32;
        let site = *self.lines.entry((file, line)).or_insert(next);
        if self.slots.len() <= site as usize {
            self.slots.resize_with(site as usize + 1, Vec::new);
        }
        site
    }
}

/// A call that changed in this frame.
struct Changed {
    /// Its place among the frame's calls.
    pos: usize,
    key: Key,
    kind: CallKind,
    /// Its layer, None for a call that names none and is new.
    layer: Option<u32>,
    args: Option<Vec<Arg>>,
    values: Py<PyTuple>,
}

/// One frame of a screen, the block of `with screen.draw() as d:`, or of
/// the scene that run() runs, the block of `with ghee.draw() as d:` in a
/// tick or a feed.
///
/// Each call writes a layer. Leaving a screen's block draws the frame, and
/// leaving the block of ghee.draw() hands its layers to the tick's frame.
/// A call takes the keys of its layer as keywords, the effects included, as
/// ghee.Layer takes them, and `name`. Positions and box sizes are pixels
/// from the top left, y down, on the screen's last pass before its filters,
/// and the drawing paints over the scene's own layers there, in the order
/// of the calls.
///
/// Each frame states the whole look of each call: a keyword left out takes
/// its default again. A call keeps its layer from frame to frame by the line
/// of code that makes it and its count from that line in the frame, or by
/// its `name`, which names the layer too. A call with no name names its
/// layer after its kind: text, text2, image, rect, line. Two calls that draw
/// one layer in a frame raise ValueError.
///
/// Leaving the block sends ghee what changed: a live field changes live,
/// any other change builds that layer alone again, a call that runs for the
/// first time adds its layer after the layer of the call before it, a layer
/// whose call did not run hides, and an effect that a call stops naming is
/// bypassed. A frame whose calls changed nothing sends nothing, and draws.
#[pyclass(module = "ghee")]
pub struct Drawing {
    handle: Py<PyAny>,
    screen: Py<Screen>,
    ends_frame: bool,
    /// Each line's count of calls in the frame.
    counts: Vec<u32>,
    /// The layer of each call, in order.
    layers: Vec<Option<u32>>,
    changed: Vec<Changed>,
}

impl Drawing {
    pub fn of(py: Python<'_>, handle: Py<PyAny>, screen: Py<Screen>, ends_frame: bool) -> Drawing {
        screen.bind(py).borrow_mut().frame += 1;
        Drawing {
            handle,
            screen,
            ends_frame,
            counts: Vec::new(),
            layers: Vec::new(),
            changed: Vec::new(),
        }
    }

    /// Hold one call of `kind` for the frame, made by the running Python
    /// code with `values`, in the order of the kind's keywords.
    fn put(
        &mut self,
        py: Python<'_>,
        kind: CallKind,
        name: Option<&str>,
        values: &[Option<Bound<'_, PyAny>>],
    ) -> PyResult<()> {
        let site = self.screen.bind(py).borrow_mut().site(py)?;
        self.put_at(py, site, kind, name, values)
    }

    /// put() of a call made on line `site`.
    fn put_at(
        &mut self,
        py: Python<'_>,
        site: u32,
        kind: CallKind,
        name: Option<&str>,
        values: &[Option<Bound<'_, PyAny>>],
    ) -> PyResult<()> {
        let screen = self.screen.bind(py);
        let mut s = screen.borrow_mut();
        if self.counts.len() <= site as usize {
            self.counts.resize(site as usize + 1, 0);
        }
        let key = (site, self.counts[site as usize]);
        self.counts[site as usize] += 1;
        let layer = match name {
            Some(name) => Some(s.number(name)),
            None => s.slot(key).named,
        };
        let frame = s.frame;
        if let Some(n) = layer {
            let held = &mut s.held[n as usize];
            if held.frame == frame {
                return Err(PyValueError::new_err(format!(
                    "two calls in one frame draw layer '{}'",
                    held.name
                )));
            }
            held.frame = frame;
        }
        self.layers.push(layer);
        let last = s.slot(key).last.as_ref();
        let unchanged = last.is_some_and(|l| {
            Some(l.layer) == layer
                && l.args.as_ref().is_some_and(|args| {
                    args.iter().zip(values).all(|(a, v)| match v {
                        Some(v) => same(a, v),
                        None => *a == Arg::None,
                    })
                })
        });
        if unchanged {
            return Ok(());
        }
        let mut all = Vec::with_capacity(values.len());
        let args = values
            .iter()
            .all(|v| match v.as_ref().map_or(Some(Arg::None), arg) {
                Some(a) => {
                    all.push(a);
                    true
                }
                None => false,
            })
            .then_some(all);
        let values = PyTuple::new(py, values.iter().cloned())?;
        self.changed.push(Changed {
            pos: self.layers.len() - 1,
            key,
            kind,
            layer,
            args,
            values: values.unbind(),
        });
        Ok(())
    }

    /// Hand ghee what the frame changed.
    fn send(&mut self, py: Python<'_>) -> PyResult<()> {
        let handle = self.handle.bind(py).clone();
        if self.screen.bind(py).borrow().pass.is_none() {
            let at = handle.call_method0("_drawing")?;
            let pass: String = at.get_item("pass")?.extract()?;
            let layers: Vec<String> = at.get_item("layers")?.extract()?;
            let taken: Vec<String> = at.get_item("taken")?.extract()?;
            let all: Vec<String> = at.get_item("all")?.extract()?;
            let mut s = self.screen.bind(py).borrow_mut();
            s.own = layers.len();
            let placed: Vec<u32> = layers.iter().map(|n| s.number(n)).collect();
            for &n in &placed {
                s.held[n as usize].placed = true;
            }
            s.placed = placed;
            for name in &all {
                let n = s.number(name);
                s.held[n as usize].in_scene = true;
            }
            s.taken = taken.into_iter().collect();
            s.pass = Some(pass);
        }
        let native = handle.cast::<Handle>().is_ok();
        let draw = match native {
            true => None,
            false => handle.getattr_opt("_draw")?,
        };
        let typed = native || draw.is_some();
        let changed = std::mem::take(&mut self.changed);
        let mut batch = Vec::new();
        for c in changed {
            let known = c.layer.is_some_and(|n| {
                let s = self.screen.bind(py).borrow();
                let held = &s.held[n as usize];
                held.placed || held.in_scene
            });
            if !known {
                self.add(py, &handle, c)?;
            } else if typed {
                batch.push(c);
            } else {
                self.update(py, &handle, c, false)?;
            }
        }
        if !batch.is_empty() {
            self.flush(py, &handle, draw.as_ref(), batch)?;
        }
        self.hide(py, &handle)
    }

    /// Add the layer of `c`, a call new to the screen, after the layer of
    /// the call before it.
    fn add(&mut self, py: Python<'_>, handle: &Bound<'_, PyAny>, c: Changed) -> PyResult<()> {
        let shim = py.import("ghee._shim")?;
        let look =
            shim.getattr("_call_look")?
                .call1((number_of(c.kind), c.values.bind(py), py.None()))?;
        let (name, n, pass, index) = {
            let mut s = self.screen.bind(py).borrow_mut();
            let frame = s.frame;
            let n = match c.layer {
                Some(n) => n,
                None => {
                    let base = base_of(c.kind);
                    let free = |name: &str| {
                        !s.taken.contains(name)
                            && s.numbers
                                .get(name)
                                .is_none_or(|&n| s.held[n as usize].frame != frame)
                    };
                    let mut name = base.to_string();
                    let mut k = 2;
                    while !free(&name) {
                        name = format!("{base}{k}");
                        k += 1;
                    }
                    let n = s.number(&name);
                    s.held[n as usize].frame = frame;
                    n
                }
            };
            let after = self.layers[..c.pos]
                .iter()
                .rev()
                .flatten()
                .find(|&&m| s.held[m as usize].placed);
            let index = match after {
                None => s.own,
                Some(m) => s
                    .placed
                    .iter()
                    .position(|p| p == m)
                    .map_or(s.own, |i| i + 1),
            };
            let pass = s.pass.clone().unwrap_or_default();
            (s.held[n as usize].name.clone(), n, pass, index)
        };
        let named = PyDict::new(py);
        named.set_item("name", &name)?;
        named.update(look.cast::<PyDict>()?.as_mapping())?;
        handle.call_method1("_add_layer", (pass, index, &named))?;
        let mut s = self.screen.bind(py).borrow_mut();
        if c.layer.is_none() {
            s.slot(c.key).named = Some(n);
            self.layers[c.pos] = Some(n);
        }
        s.placed.insert(index, n);
        s.taken.insert(name);
        let held = &mut s.held[n as usize];
        held.placed = true;
        held.look = Some(named.into_any().unbind());
        drop(s);
        self.took(
            py,
            Changed {
                layer: Some(n),
                ..c
            },
        )
    }

    /// Give the layer of `c` its look, written as the scene format writes
    /// it. With `typed`, the handle took the frame's other calls typed, and
    /// the layer's last look stands for nothing.
    fn update(
        &mut self,
        py: Python<'_>,
        handle: &Bound<'_, PyAny>,
        c: Changed,
        typed: bool,
    ) -> PyResult<()> {
        let n = c.layer.expect("a known call has its layer");
        let name = self.screen.bind(py).borrow().held[n as usize].name.clone();
        let look = py.import("ghee._shim")?.getattr("_call_look")?.call1((
            number_of(c.kind),
            c.values.bind(py),
            &name,
        ))?;
        let last = self.screen.bind(py).borrow().held[n as usize]
            .look
            .as_ref()
            .map(|l| l.clone_ref(py));
        let same_look = match (&last, typed) {
            (Some(last), false) => last.bind(py).eq(&look)?,
            _ => false,
        };
        if !same_look {
            handle.call_method1("_update_layer", (&look,))?;
        }
        self.screen.bind(py).borrow_mut().held[n as usize].look = Some(look.unbind());
        self.took(py, c)
    }

    /// Give the layers of the calls of `batch` their looks typed, through
    /// the session of a native handle or the handle's `draw`, and through
    /// update() the calls that the typed path leaves.
    fn flush(
        &mut self,
        py: Python<'_>,
        handle: &Bound<'_, PyAny>,
        draw: Option<&Bound<'_, PyAny>>,
        mut batch: Vec<Changed>,
    ) -> PyResult<()> {
        let left = match self.take_typed(py, handle, draw, &mut batch) {
            Ok(left) => left,
            Err(e) => {
                // The handle may have taken a call before it failed, so each
                // call of the batch goes out again.
                let mut s = self.screen.bind(py).borrow_mut();
                for c in &batch {
                    s.slot(c.key).last = None;
                }
                return Err(e);
            }
        };
        let mut slow = Vec::new();
        for (k, c) in batch.into_iter().enumerate() {
            match left.binary_search(&k) {
                Ok(_) => slow.push(c),
                Err(_) => self.took(py, c)?,
            }
        }
        for c in slow {
            self.update(py, handle, c, true)?;
        }
        Ok(())
    }

    /// The indexes of the calls of `batch` that the typed path leaves, in
    /// order.
    fn take_typed(
        &self,
        py: Python<'_>,
        handle: &Bound<'_, PyAny>,
        draw: Option<&Bound<'_, PyAny>>,
        batch: &mut [Changed],
    ) -> PyResult<Vec<usize>> {
        let s = self.screen.bind(py).borrow();
        let name = |c: &Changed| s.held[c.layer.expect("a known call") as usize].name.clone();
        if let Some(draw) = draw {
            let items = batch
                .iter()
                .map(|c| (number_of(c.kind), name(c), c.values.bind(py)))
                .collect::<Vec<_>>();
            drop(s);
            let mut left: Vec<usize> = draw.call1((PyList::new(py, items)?,))?.extract()?;
            left.sort_unstable();
            return Ok(left);
        }
        let mut calls = Vec::with_capacity(batch.len());
        let mut at = Vec::with_capacity(batch.len());
        let mut left = Vec::new();
        for (k, c) in batch.iter_mut().enumerate() {
            let layer = name(c);
            match c.args.take() {
                Some(args) => {
                    calls.push(Call {
                        kind: c.kind,
                        layer,
                        args,
                    });
                    at.push(k);
                }
                None => left.push(k),
            }
        }
        drop(s);
        let rest = handle.cast::<Handle>()?.borrow_mut().update_calls(&calls);
        for (call, &k) in calls.into_iter().zip(&at) {
            batch[k].args = Some(call.args);
        }
        left.extend(rest?.into_iter().map(|i| at[i]));
        left.sort_unstable();
        Ok(left)
    }

    /// Hold that the layer of `c` shows its values.
    fn took(&mut self, py: Python<'_>, c: Changed) -> PyResult<()> {
        let n = c.layer.expect("a call that went out has its layer");
        let hidden = {
            let v = c.values.bind(py).get_item(visible_at(c.kind))?;
            match v.cast::<PyBool>() {
                Ok(b) => !b.is_true(),
                Err(_) if v.is_none() => false,
                Err(_) => py
                    .import("ghee._shim")?
                    .getattr("_plain")?
                    .call1((&v,))?
                    .is(&*PyBool::new(py, false)),
            }
        };
        let mut s = self.screen.bind(py).borrow_mut();
        if let Some(owner) = s.held[n as usize].owner.filter(|o| *o != c.key) {
            let slot = s.slot(owner);
            if slot.last.as_ref().is_some_and(|l| l.layer == n) {
                slot.last = None;
            }
        }
        s.slot(c.key).last = Some(Last {
            layer: n,
            args: c.args,
        });
        let held = &mut s.held[n as usize];
        held.owner = Some(c.key);
        held.visible = !hidden;
        if !held.drawn {
            held.drawn = true;
            s.order.push(n);
        }
        Ok(())
    }

    /// Hide each layer that shows and whose call did not run, through its
    /// live eye.
    fn hide(&mut self, py: Python<'_>, handle: &Bound<'_, PyAny>) -> PyResult<()> {
        let gone: Vec<(u32, String)> = {
            let s = self.screen.bind(py).borrow();
            s.order
                .iter()
                .map(|&n| (n, &s.held[n as usize]))
                .filter(|(_, h)| h.visible && h.frame != s.frame)
                .map(|(n, h)| (n, h.name.clone()))
                .collect()
        };
        for (n, name) in gone {
            handle.call_method1("set_layer", (&name, "visible", false))?;
            let look = self.screen.bind(py).borrow().held[n as usize]
                .look
                .as_ref()
                .map(|l| l.clone_ref(py));
            let hidden = match look {
                Some(look) => {
                    let copy = look.bind(py).cast::<PyDict>()?.copy()?;
                    copy.set_item("visible", false)?;
                    Some(copy.into_any().unbind())
                }
                None => None,
            };
            let mut s = self.screen.bind(py).borrow_mut();
            let held = &mut s.held[n as usize];
            held.visible = false;
            if hidden.is_some() {
                held.look = hidden;
            }
            if let Some(owner) = held.owner {
                let slot = s.slot(owner);
                if slot.last.as_ref().is_some_and(|l| l.layer == n) {
                    slot.last = None;
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl Drawing {
    #[new]
    #[pyo3(signature = (handle, screen, ends_frame=true))]
    fn new(py: Python<'_>, handle: Py<PyAny>, screen: Py<Screen>, ends_frame: bool) -> Drawing {
        Drawing::of(py, handle, screen, ends_frame)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (kind, _value, _traceback))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if kind.is_none() {
            self.send(py)?;
            if self.ends_frame {
                self.handle.bind(py).call_method0("frame")?;
            }
        }
        Ok(false)
    }

    /// Hand ghee what the frame changed, as leaving the block does before
    /// it draws the frame.
    fn _send(&mut self, py: Python<'_>) -> PyResult<()> {
        self.send(py)
    }

    /// Draw `words`, a text layer. With no `font_size` the words fit the
    /// screen, and an axis with no position centres them.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (words, *, name=None, x=None, y=None, color=None, opacity=None,
                        blend=None, visible=None, font_size=None, font_family=None,
                        text_align=None, render=None, text_shadow=None, glow=None,
                        text_stroke=None, karaoke=None, wave=None, scroll=None,
                        blink=None))]
    fn text<'py>(
        &mut self,
        py: Python<'py>,
        words: Bound<'py, PyAny>,
        name: Option<&str>,
        x: Option<Bound<'py, PyAny>>,
        y: Option<Bound<'py, PyAny>>,
        color: Option<Bound<'py, PyAny>>,
        opacity: Option<Bound<'py, PyAny>>,
        blend: Option<Bound<'py, PyAny>>,
        visible: Option<Bound<'py, PyAny>>,
        font_size: Option<Bound<'py, PyAny>>,
        font_family: Option<Bound<'py, PyAny>>,
        text_align: Option<Bound<'py, PyAny>>,
        render: Option<Bound<'py, PyAny>>,
        text_shadow: Option<Bound<'py, PyAny>>,
        glow: Option<Bound<'py, PyAny>>,
        text_stroke: Option<Bound<'py, PyAny>>,
        karaoke: Option<Bound<'py, PyAny>>,
        wave: Option<Bound<'py, PyAny>>,
        scroll: Option<Bound<'py, PyAny>>,
        blink: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let words = Some(words).filter(|w| !w.is_none());
        self.put(
            py,
            CallKind::Text,
            name,
            &[
                words,
                x,
                y,
                color,
                opacity,
                blend,
                visible,
                font_size,
                font_family,
                text_align,
                render,
                text_shadow,
                glow,
                text_stroke,
                karaoke,
                wave,
                scroll,
                blink,
            ],
        )
    }

    /// Draw `image`, an image layer of a .png or .jpg file or an image asset
    /// of the scene. A file loads once and stays with the screen.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (image, *, name=None, x=None, y=None, width=None, height=None,
                        color=None, opacity=None, blend=None, visible=None))]
    fn image<'py>(
        &mut self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        name: Option<&str>,
        x: Option<Bound<'py, PyAny>>,
        y: Option<Bound<'py, PyAny>>,
        width: Option<Bound<'py, PyAny>>,
        height: Option<Bound<'py, PyAny>>,
        color: Option<Bound<'py, PyAny>>,
        opacity: Option<Bound<'py, PyAny>>,
        blend: Option<Bound<'py, PyAny>>,
        visible: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let image = Some(image).filter(|i| !i.is_none());
        self.put(
            py,
            CallKind::Image,
            name,
            &[image, x, y, width, height, color, opacity, blend, visible],
        )
    }

    /// Draw a box, a rect shape layer, the whole screen on a size left out.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, name=None, x=None, y=None, width=None, height=None, color=None,
                        opacity=None, blend=None, visible=None, border=None,
                        border_radius=None))]
    fn rect<'py>(
        &mut self,
        py: Python<'py>,
        name: Option<&str>,
        x: Option<Bound<'py, PyAny>>,
        y: Option<Bound<'py, PyAny>>,
        width: Option<Bound<'py, PyAny>>,
        height: Option<Bound<'py, PyAny>>,
        color: Option<Bound<'py, PyAny>>,
        opacity: Option<Bound<'py, PyAny>>,
        blend: Option<Bound<'py, PyAny>>,
        visible: Option<Bound<'py, PyAny>>,
        border: Option<Bound<'py, PyAny>>,
        border_radius: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        self.put(
            py,
            CallKind::Rect,
            name,
            &[
                x,
                y,
                width,
                height,
                color,
                opacity,
                blend,
                visible,
                border,
                border_radius,
            ],
        )
    }

    /// Draw a circle of `radius` pixels around `centre`, (x, y): a rect of
    /// `2 * radius` with `border_radius` at `radius`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (centre, radius, *, name=None, color=None, opacity=None, blend=None,
                        visible=None, border=None))]
    fn circle<'py>(
        &mut self,
        py: Python<'py>,
        centre: Bound<'py, PyAny>,
        radius: Bound<'py, PyAny>,
        name: Option<&str>,
        color: Option<Bound<'py, PyAny>>,
        opacity: Option<Bound<'py, PyAny>>,
        blend: Option<Bound<'py, PyAny>>,
        visible: Option<Bound<'py, PyAny>>,
        border: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        // The line of the call goes first: the shim's arithmetic runs
        // Python code of its own.
        let site = self.screen.bind(py).borrow_mut().site(py)?;
        let boxed = py
            .import("ghee._shim")?
            .getattr("_circle")?
            .call1((centre, radius))?;
        let [x, y, w, h, r] = [0, 1, 2, 3, 4].map(|k| boxed.get_item(k).ok());
        self.put_at(
            py,
            site,
            CallKind::Rect,
            name,
            &[x, y, w, h, color, opacity, blend, visible, border, r],
        )
    }

    /// Draw a line from `a` to `b`, each (x, y): `lines([a, b])`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, b, *, name=None, x=None, y=None, color=None, opacity=None,
                        blend=None, visible=None, stroke_width=None))]
    fn line<'py>(
        &mut self,
        py: Python<'py>,
        a: Bound<'py, PyAny>,
        b: Bound<'py, PyAny>,
        name: Option<&str>,
        x: Option<Bound<'py, PyAny>>,
        y: Option<Bound<'py, PyAny>>,
        color: Option<Bound<'py, PyAny>>,
        opacity: Option<Bound<'py, PyAny>>,
        blend: Option<Bound<'py, PyAny>>,
        visible: Option<Bound<'py, PyAny>>,
        stroke_width: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let points = PyTuple::new(py, [a, b])?.into_any();
        self.put(
            py,
            CallKind::Line,
            name,
            &[
                Some(points),
                x,
                y,
                color,
                opacity,
                blend,
                visible,
                stroke_width,
            ],
        )
    }

    /// Draw a line through `points`, each (x, y), two or more. `x` and `y`
    /// shift every point.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (points, *, name=None, x=None, y=None, color=None, opacity=None,
                        blend=None, visible=None, stroke_width=None))]
    fn lines<'py>(
        &mut self,
        py: Python<'py>,
        points: Bound<'py, PyAny>,
        name: Option<&str>,
        x: Option<Bound<'py, PyAny>>,
        y: Option<Bound<'py, PyAny>>,
        color: Option<Bound<'py, PyAny>>,
        opacity: Option<Bound<'py, PyAny>>,
        blend: Option<Bound<'py, PyAny>>,
        visible: Option<Bound<'py, PyAny>>,
        stroke_width: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let points = Some(points).filter(|p| !p.is_none());
        self.put(
            py,
            CallKind::Line,
            name,
            &[points, x, y, color, opacity, blend, visible, stroke_width],
        )
    }
}
