//! Kind 6 of the wire: an RFC 6902 JSON Patch over the scene's layers.
//!
//! The patch's document is the scene's layers, an object of each pass's name
//! and the list of its layers in the order that they draw. The pointer
//! `/main/2` names the third layer of pass `main`, `/main/-` the end of that
//! list, `/main/2/x` a key of the layer, and `/main/2/glow/strength` a field
//! of its effect (RFC 6901).

use serde_json::Value;

/// A place that a pointer names.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub pass: String,
    /// The layer's place among the pass's layers, None for `-`, past the
    /// last.
    pub index: Option<usize>,
    /// A key of the layer, then a field of its effect. None names the layer.
    pub keys: Vec<String>,
}

impl Place {
    /// The pointer, as the patch wrote it.
    pub fn pointer(&self) -> String {
        let index = self.index.map_or("-".to_string(), |i| i.to_string());
        let mut out = format!("/{}/{index}", escape(&self.pass));
        for k in &self.keys {
            out.push('/');
            out.push_str(&escape(k));
        }
        out
    }
}

/// One operation of a patch, as RFC 6902 section 4 names it.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Add(Place, Value),
    Remove(Place),
    Replace(Place, Value),
    Move { from: Place, path: Place },
    Copy { from: Place, path: Place },
    Test(Place, Value),
}

/// The operations of the patch `text`, or why it holds none.
pub fn ops_of(text: &str) -> Result<Vec<Op>, String> {
    let doc: Value =
        serde_json::from_str(text).map_err(|e| format!("kind 6 holds no JSON: {e}"))?;
    let Value::Array(items) = doc else {
        return Err("kind 6 needs a JSON Patch, a list of operations".to_string());
    };
    items.iter().enumerate().map(|(k, o)| op_of(k, o)).collect()
}

fn op_of(k: usize, o: &Value) -> Result<Op, String> {
    let at = |what: &str| format!("kind 6 operation {k}: {what}");
    let Value::Object(o) = o else {
        return Err(at("needs an object"));
    };
    let text = |key: &str| {
        o.get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| at(&format!("needs '{key}'")))
    };
    let place = |key: &str, end: bool| -> Result<Place, String> {
        let place = place_of(text(key)?).map_err(|why| at(&why))?;
        if place.index.is_none() && (!end || !place.keys.is_empty()) {
            return Err(at(&format!(
                "'{key}' takes '-' only as the end of a pass's layers"
            )));
        }
        Ok(place)
    };
    let value = || o.get("value").cloned().ok_or_else(|| at("needs 'value'"));
    Ok(match text("op")? {
        "add" => Op::Add(place("path", true)?, value()?),
        "remove" => Op::Remove(place("path", false)?),
        "replace" => Op::Replace(place("path", false)?, value()?),
        "move" => Op::Move {
            from: place("from", false)?,
            path: place("path", true)?,
        },
        "copy" => Op::Copy {
            from: place("from", false)?,
            path: place("path", true)?,
        },
        "test" => Op::Test(place("path", false)?, value()?),
        op => return Err(at(&format!("has no op '{op}'"))),
    })
}

/// The place that `pointer` names: a pass, the place of a layer, and up to
/// two keys.
fn place_of(pointer: &str) -> Result<Place, String> {
    let Some(rest) = pointer.strip_prefix('/') else {
        return Err(format!("pointer '{pointer}' needs to start with '/'"));
    };
    let tokens: Vec<String> = rest.split('/').map(unescape).collect();
    let (pass, index, keys) = match tokens.as_slice() {
        [pass, index, keys @ ..] if keys.len() <= 2 => (pass, index, keys),
        _ => {
            return Err(format!(
                "pointer '{pointer}' needs a pass, a layer's place, and at most a key and a \
                 field: /main/0/glow/strength"
            ))
        }
    };
    let index = match index.as_str() {
        "-" => None,
        i if i == "0" || (!i.starts_with('0') && i.bytes().all(|b| b.is_ascii_digit())) => Some(
            i.parse()
                .map_err(|_| format!("pointer '{pointer}': no place {i}"))?,
        ),
        i => return Err(format!("pointer '{pointer}': '{i}' is no layer's place")),
    };
    Ok(Place {
        pass: pass.clone(),
        index,
        keys: keys.to_vec(),
    })
}

/// A reference token as RFC 6901 section 4 reads it: `~1` is `/`, then `~0`
/// is `~`.
fn unescape(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn escape(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// True when `a` and `b` are equal as RFC 6902 section 4.6 compares JSON:
/// numbers by their value, objects by their members in any order.
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn at(pass: &str, index: Option<usize>, keys: &[&str]) -> Place {
        Place {
            pass: pass.to_string(),
            index,
            keys: keys.iter().map(|k| (*k).to_string()).collect(),
        }
    }

    #[test]
    fn a_patch_reads_each_operation_of_rfc_6902() {
        let text = json!([
            {"op": "add", "path": "/main/-", "value": {"name": "t", "text": "hi"}},
            {"op": "replace", "path": "/main/0/x", "value": null},
            {"op": "remove", "path": "/main/1/glow/strength"},
            {"op": "move", "from": "/main/1", "path": "/main/0"},
            {"op": "copy", "from": "/main/0/x", "path": "/main/1/y"},
            {"op": "test", "path": "/m~1a~0in/10", "value": {"name": "t"}},
        ])
        .to_string();
        assert_eq!(
            ops_of(&text).unwrap(),
            vec![
                Op::Add(at("main", None, &[]), json!({"name": "t", "text": "hi"})),
                Op::Replace(at("main", Some(0), &["x"]), Value::Null),
                Op::Remove(at("main", Some(1), &["glow", "strength"])),
                Op::Move {
                    from: at("main", Some(1), &[]),
                    path: at("main", Some(0), &[]),
                },
                Op::Copy {
                    from: at("main", Some(0), &["x"]),
                    path: at("main", Some(1), &["y"]),
                },
                Op::Test(at("m/a~in", Some(10), &[]), json!({"name": "t"})),
            ]
        );
        assert_eq!(at("m/a~in", Some(10), &[]).pointer(), "/m~1a~0in/10");
    }

    #[test]
    fn a_patch_that_rfc_6902_refuses_says_why() {
        for (patch, why) in [
            (
                json!({"op": "add"}),
                "kind 6 needs a JSON Patch, a list of operations",
            ),
            (
                json!([{"path": "/main/0"}]),
                "kind 6 operation 0: needs 'op'",
            ),
            (
                json!([{"op": "add", "path": "/main/0"}]),
                "kind 6 operation 0: needs 'value'",
            ),
            (
                json!([{"op": "remove", "path": "/main/-"}]),
                "kind 6 operation 0: 'path' takes '-' only as the end of a pass's layers",
            ),
            (
                json!([{"op": "replace", "path": "/main/01", "value": 1}]),
                "kind 6 operation 0: pointer '/main/01': '01' is no layer's place",
            ),
            (
                json!([{"op": "remove", "path": "main/0"}]),
                "kind 6 operation 0: pointer 'main/0' needs to start with '/'",
            ),
            (
                json!([{"op": "remove", "path": "/main/0/a/b/c"}]),
                "kind 6 operation 0: pointer '/main/0/a/b/c' needs a pass, a layer's place, \
                 and at most a key and a field: /main/0/glow/strength",
            ),
            (
                json!([{"op": "merge", "path": "/main/0", "value": {}}]),
                "kind 6 operation 0: has no op 'merge'",
            ),
        ] {
            assert_eq!(ops_of(&patch.to_string()), Err(why.to_string()), "{patch}");
        }
    }

    #[test]
    fn json_compares_as_rfc_6902_compares_it() {
        assert!(same(
            &json!({"a": 1, "b": [2.0]}),
            &json!({"b": [2], "a": 1.0})
        ));
        assert!(!same(&json!({"a": 1}), &json!({"a": 1, "b": null})));
        assert!(!same(&json!("1"), &json!(1)));
    }
}
