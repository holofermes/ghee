//! A drawing's calls, as the shim's drawing hands them over: each a tuple of
//! its kind, the name of its layer, and the values of its keywords in the
//! order of ghee::draw's args of that kind. A value takes the form that the
//! shim's look gives it, so a call here makes the layer that the same look
//! written as JSON makes.

use ghee::draw::{Arg, Call, CallKind};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

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
        return Some(Arg::Str(s.to_cow().ok()?.into_owned()));
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
            let k = k.cast::<PyString>().ok()?.to_cow().ok()?.into_owned();
            Some((k, arg(&x)?))
        });
        return pairs.collect::<Option<_>>().map(Arg::Map);
    }
    None
}

/// The kind of a call by the shim's number for it.
pub fn kind_of(n: i64) -> Option<CallKind> {
    match n {
        0 => Some(CallKind::Text),
        1 => Some(CallKind::Rect),
        2 => Some(CallKind::Image),
        3 => Some(CallKind::Line),
        _ => None,
    }
}

/// The call of `item`, a tuple (kind, layer, values). None for anything
/// that arg() takes no value of.
pub fn call_of(item: &Bound<'_, PyAny>) -> Option<Call> {
    let t = item.cast::<PyTuple>().ok()?;
    let kind = kind_of(t.get_item(0).ok()?.extract::<i64>().ok()?)?;
    let layer = t.get_item(1).ok()?.extract::<String>().ok()?;
    let values = t.get_item(2).ok()?;
    let args = values
        .cast::<PyTuple>()
        .ok()?
        .iter()
        .map(|v| arg(&v))
        .collect::<Option<Vec<_>>>()?;
    Some(Call { kind, layer, args })
}
