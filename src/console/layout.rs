//! A JSON text rewritten to hold a new value, with its layout kept.
//!
//! rewrite() reads the text into nodes, each with the byte span of its value,
//! and compares each node with the new value at its place. A node whose value
//! is unchanged keeps its text byte for byte. An object keeps the order of
//! its keys: a removed key goes with its comma, and a new key follows the
//! last kept one, on a line of its own when its siblings take a line each.
//! An array of the same length changes element by element. An array that
//! grew or shrank is written again, keeping the text of each element that it
//! still holds, matched by its `name` when its elements have one. A value
//! written new takes one line when the text that it replaces took one line,
//! and else the indent of the text around it.

use serde_json::Value;

/// rewrite(text, new): `text`, a JSON document, rewritten to hold `new`.
pub fn rewrite(text: &str, new: &Value) -> Result<String, String> {
    let mut p = Parser { src: text, at: 0 };
    p.ws();
    let root = p.value()?;
    p.ws();
    if p.at != text.len() {
        return Err(format!("trailing text at byte {}", p.at));
    }
    let doc = Doc {
        src: text,
        lined_lists: lined_lists(text, &root),
    };
    let body = node_text(&doc, &root, new);
    Ok(format!(
        "{}{body}{}",
        &text[..root.start],
        &text[root.end..]
    ))
}

/// The text being rewritten, and whether it writes a list of plain values
/// across lines, as ghee does.
struct Doc<'a> {
    src: &'a str,
    lined_lists: bool,
}

/// Whether some array of `node` holds plain values only and spans lines.
fn lined_lists(src: &str, node: &Node) -> bool {
    match &node.kind {
        Kind::Scalar => false,
        Kind::Object(m) => m.iter().any(|(_, _, v)| lined_lists(src, v)),
        Kind::Array(items) => {
            let plain = !items.is_empty() && items.iter().all(|i| matches!(i.kind, Kind::Scalar));
            (plain && src[node.start..node.end].contains('\n'))
                || items.iter().any(|i| lined_lists(src, i))
        }
    }
}

struct Node {
    start: usize,
    end: usize,
    kind: Kind,
}

enum Kind {
    Scalar,
    /// Each member: the start of its key, its key, and its value.
    Object(Vec<(usize, String, Node)>),
    Array(Vec<Node>),
}

impl Node {
    fn value(&self, src: &str) -> Value {
        serde_json::from_str(&src[self.start..self.end]).unwrap_or(Value::Null)
    }
}

struct Parser<'a> {
    src: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.at).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn eat(&mut self, b: u8) -> Result<(), String> {
        self.ws();
        match self.peek() {
            Some(c) if c == b => {
                self.at += 1;
                Ok(())
            }
            _ => Err(format!("expected '{}' at byte {}", b as char, self.at)),
        }
    }

    fn value(&mut self) -> Result<Node, String> {
        self.ws();
        let start = self.at;
        let kind = match self.peek() {
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                } else {
                    loop {
                        self.ws();
                        let key_at = self.at;
                        let key = self.string()?;
                        self.eat(b':')?;
                        let v = self.value()?;
                        members.push((key_at, key, v));
                        self.ws();
                        match self.peek() {
                            Some(b',') => self.at += 1,
                            Some(b'}') => {
                                self.at += 1;
                                break;
                            }
                            _ => return Err(format!("expected ',' or '}}' at byte {}", self.at)),
                        }
                    }
                }
                Kind::Object(members)
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.at += 1;
                } else {
                    loop {
                        items.push(self.value()?);
                        self.ws();
                        match self.peek() {
                            Some(b',') => self.at += 1,
                            Some(b']') => {
                                self.at += 1;
                                break;
                            }
                            _ => return Err(format!("expected ',' or ']' at byte {}", self.at)),
                        }
                    }
                }
                Kind::Array(items)
            }
            Some(b'"') => {
                self.string()?;
                Kind::Scalar
            }
            Some(_) => {
                while matches!(self.peek(), Some(c) if !b",]} \t\n\r".contains(&c)) {
                    self.at += 1;
                }
                if self.at == start {
                    return Err(format!("expected a value at byte {start}"));
                }
                Kind::Scalar
            }
            None => return Err("the text ends before its value".to_string()),
        };
        let node = Node {
            start,
            end: self.at,
            kind,
        };
        if matches!(node.kind, Kind::Scalar)
            && serde_json::from_str::<Value>(&self.src[start..self.at]).is_err()
        {
            return Err(format!("no JSON value at byte {start}"));
        }
        Ok(node)
    }

    fn string(&mut self) -> Result<String, String> {
        let start = self.at;
        if self.peek() != Some(b'"') {
            return Err(format!("expected a string at byte {start}"));
        }
        self.at += 1;
        loop {
            match self.peek() {
                Some(b'\\') => self.at += 2,
                Some(b'"') => {
                    self.at += 1;
                    break;
                }
                Some(_) => self.at += 1,
                None => return Err(format!("a string at byte {start} never ends")),
            }
        }
        serde_json::from_str(&self.src[start..self.at])
            .map_err(|e| format!("string at byte {start}: {e}"))
    }
}

/// The text of `node`, rewritten to hold `new`.
fn node_text(doc: &Doc, node: &Node, new: &Value) -> String {
    let src = doc.src;
    let old = &src[node.start..node.end];
    if node.value(src) == *new {
        return old.to_string();
    }
    match (&node.kind, new) {
        (Kind::Object(members), Value::Object(m))
            if members.iter().any(|(_, k, _)| m.contains_key(k)) =>
        {
            object_text(doc, node, members, m)
        }
        (Kind::Array(items), Value::Array(a))
            if items.len() == a.len() && aligned(src, items, a) =>
        {
            let edits = items
                .iter()
                .zip(a)
                .map(|(n, v)| (n.start, n.end, node_text(doc, n, v)))
                .collect();
            splice(src, node.start, node.end, edits)
        }
        (Kind::Array(items), Value::Array(a)) if !items.is_empty() && !a.is_empty() => {
            array_text(doc, node, items, a)
        }
        _ => fresh(doc, node, new),
    }
}

fn object_text(
    doc: &Doc,
    node: &Node,
    members: &[(usize, String, Node)],
    new: &serde_json::Map<String, Value>,
) -> String {
    let src = doc.src;
    let kept: Vec<bool> = members
        .iter()
        .map(|(_, k, _)| new.contains_key(k))
        .collect();
    let last_kept = kept.iter().rposition(|k| *k).unwrap_or_default();
    let mut edits = Vec::new();
    let mut gone: Vec<(usize, usize)> = Vec::new();
    for (i, (key_at, key, v)) in members.iter().enumerate() {
        match new.get(key) {
            Some(nv) => edits.push((v.start, v.end, node_text(doc, v, nv))),
            None if i < last_kept => {
                let next = members[i + 1].0;
                match i > 0 && src[v.end..next].contains('\n') {
                    true => gone.push((members[i - 1].2.end, v.end)),
                    false => gone.push((*key_at, next)),
                }
            }
            None => {}
        }
    }
    let tail = members[last_kept].2.end;
    let end = members.last().map_or(tail, |m| m.2.end);
    gone.push((tail, end));
    gone.sort();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (from, to) in gone {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    let one_line = !src[node.start..node.end].contains('\n');
    let lined = !one_line && starts_line(src, members[last_kept].0);
    let indent = line_indent(src, members[last_kept].0);
    let sibling_one_line = !src[members[last_kept].0..tail].contains('\n');
    let mut added = String::new();
    for (k, v) in new
        .iter()
        .filter(|(k, _)| !members.iter().any(|(_, m, _)| m == *k))
    {
        let key = serde_json::to_string(k).unwrap_or_default();
        let value = match one_line || sibling_one_line || plain_list(v) && !doc.lined_lists {
            true => compact(v),
            false => pretty(v, &indent, &unit(src, node)),
        };
        match lined {
            true => added.push_str(&format!(",\n{indent}{key}: {value}")),
            false => added.push_str(&format!(", {key}: {value}")),
        }
    }
    for (from, to) in merged {
        let text = match from == tail {
            true => std::mem::take(&mut added),
            false => String::new(),
        };
        if from != to || !text.is_empty() {
            edits.push((from, to, text));
        }
    }
    splice(src, node.start, node.end, edits)
}

/// Whether the elements of `items` and `new` pair by position: every pair
/// that both have a `name` has the same one.
fn aligned(src: &str, items: &[Node], new: &[Value]) -> bool {
    items
        .iter()
        .zip(new)
        .all(|(n, v)| match (name_of(&n.value(src)), name_of(v)) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        })
}

/// Whether `v` is a list of plain values, such as a vec2 or a list of
/// filter names.
fn plain_list(v: &Value) -> bool {
    v.as_array()
        .is_some_and(|a| a.iter().all(|i| !i.is_array() && !i.is_object()))
}

fn name_of(v: &Value) -> Option<String> {
    v.get("name").and_then(Value::as_str).map(str::to_string)
}

fn array_text(doc: &Doc, node: &Node, items: &[Node], new: &[Value]) -> String {
    let src = doc.src;
    let olds: Vec<Value> = items.iter().map(|n| n.value(src)).collect();
    let mut used = vec![false; items.len()];
    let one_line = !src[node.start..node.end].contains('\n');
    let first = &items[0];
    let indent = match starts_line(src, first.start) {
        true => line_indent(src, first.start),
        false => format!("{}{}", line_indent(src, node.start), unit(src, node)),
    };
    let texts: Vec<String> = new
        .iter()
        .map(|v| {
            let found = (0..items.len()).find(|&i| {
                !used[i]
                    && match name_of(v) {
                        Some(n) => name_of(&olds[i]).as_deref() == Some(n.as_str()),
                        None => olds[i] == *v,
                    }
            });
            match found {
                Some(i) => {
                    used[i] = true;
                    node_text(doc, &items[i], v)
                }
                None if one_line => compact(v),
                None => pretty(v, &indent, &unit(src, node)),
            }
        })
        .collect();
    if one_line {
        return format!("[{}]", texts.join(", "));
    }
    let close = line_indent(src, node.start);
    let open = match starts_line(src, first.start) {
        true => format!("[\n{indent}"),
        false => "[".to_string(),
    };
    let last = items.last().map_or(node.end, |n| n.end);
    let closing = match src[last..node.end].contains('\n') {
        true => format!("\n{close}]"),
        false => "]".to_string(),
    };
    format!("{open}{}{closing}", texts.join(&format!(",\n{indent}")))
}

/// `new` written where `node` stood: on one line when the node took one,
/// else as an indented block.
fn fresh(doc: &Doc, node: &Node, new: &Value) -> String {
    let src = doc.src;
    match src[node.start..node.end].contains('\n') && !(plain_list(new) && !doc.lined_lists) {
        false => compact(new),
        true => pretty(new, &line_indent(src, node.start), &unit(src, node)),
    }
}

/// `src[start..end]` with each edit `(from, to, text)` in place of its span.
fn splice(src: &str, start: usize, end: usize, mut edits: Vec<(usize, usize, String)>) -> String {
    edits.sort_by_key(|e| e.0);
    let mut out = String::new();
    let mut at = start;
    for (from, to, text) in edits {
        if from < at {
            continue;
        }
        out.push_str(&src[at..from]);
        out.push_str(&text);
        at = to;
    }
    out.push_str(&src[at..end]);
    out
}

fn starts_line(src: &str, at: usize) -> bool {
    src[..at]
        .bytes()
        .rev()
        .take_while(|b| *b != b'\n')
        .all(|b| b == b' ' || b == b'\t')
}

fn line_indent(src: &str, at: usize) -> String {
    let line = src[..at].rfind('\n').map_or(0, |i| i + 1);
    src[line..]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

/// The indent that one level adds inside `node`: the indent of the first
/// line inside it past the node's own, else two spaces, as ghee writes.
fn unit(src: &str, node: &Node) -> String {
    let base = line_indent(src, node.start);
    src[node.start..node.end]
        .split('\n')
        .skip(1)
        .map(|l| {
            l.chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect::<String>()
        })
        .find(|i| i.len() > base.len() && i.starts_with(&base))
        .map_or("  ".to_string(), |i| i[base.len()..].to_string())
}

/// `v` on one line, with a space after each `,` and `:`.
fn compact(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(compact).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    serde_json::to_string(k).unwrap_or_default(),
                    compact(v)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        v => serde_json::to_string(v).unwrap_or_default(),
    }
}

/// `v` as an indented block that opens on the current line, at `indent`,
/// each level `unit` deeper.
fn pretty(v: &Value, indent: &str, unit: &str) -> String {
    use serde::Serialize;
    let mut buf = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(unit.as_bytes());
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
    if v.serialize(&mut ser).is_err() {
        return compact(v);
    }
    String::from_utf8_lossy(&buf).replace('\n', &format!("\n{indent}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn edit(text: &str, f: impl Fn(&mut Value)) -> String {
        let mut v: Value = serde_json::from_str(text).unwrap();
        f(&mut v);
        let out = rewrite(text, &v).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap(), v, "{out}");
        out
    }

    const SCENE: &str = r#"{
  "version": 1,
  "params": {
    "period":   {"type": "float", "value": 8.0, "min": 2.0},
    "rate": {"type": "float", "value": 30.0}
  },
  "passes": [
    {"name": "flow", "fragment": {"file": "a-flow.glsl"}, "scale": 0.25,
     "params": {"size": {"type": "vec2", "value": [0.5, 0.5]}}},
    {"name": "image", "fragment": {"file": "a-image.glsl"}}
  ]
}
"#;

    #[test]
    fn an_unchanged_value_keeps_its_text() {
        assert_eq!(edit(SCENE, |_| {}), SCENE);
    }

    #[test]
    fn a_changed_value_changes_alone() {
        let out = edit(SCENE, |v| {
            v["params"]["period"]["value"] = json!(9.5);
            v["passes"][0]["params"]["size"]["value"] = json!([0.5, 0.25]);
        });
        assert_eq!(
            out,
            SCENE
                .replace("\"value\": 8.0", "\"value\": 9.5")
                .replace("[0.5, 0.5]", "[0.5, 0.25]")
        );
    }

    #[test]
    fn a_new_key_follows_the_last_on_its_own_line_and_a_removed_key_takes_its_comma() {
        let out = edit(SCENE, |v| {
            v["params"]["hot"] = json!({"type": "color", "value": "#ffffff"});
            v["params"].as_object_mut().unwrap().shift_remove("period");
            v["passes"][1]["visible"] = json!(false);
            v["passes"][0]
                .as_object_mut()
                .unwrap()
                .shift_remove("scale");
        });
        assert_eq!(
            out,
            SCENE
                .replace(
                    "    \"period\":   {\"type\": \"float\", \"value\": 8.0, \"min\": 2.0},\n",
                    ""
                )
                .replace(
                    "\"value\": 30.0}\n",
                    "\"value\": 30.0},\n    \"hot\": {\"type\": \"color\", \"value\": \"#ffffff\"}\n"
                )
                .replace(", \"scale\": 0.25,\n", ",\n")
                .replace("a-image.glsl\"}}", "a-image.glsl\"}, \"visible\": false}")
        );
    }

    #[test]
    fn a_grown_array_keeps_the_text_of_each_element_that_it_holds() {
        let out = edit(SCENE, |v| {
            let passes = v["passes"].as_array_mut().unwrap();
            passes.swap(0, 1);
            passes.push(json!({"name": "glow", "fragment": {"file": "glow.glsl"}}));
        });
        assert_eq!(
            out,
            r#"{
  "version": 1,
  "params": {
    "period":   {"type": "float", "value": 8.0, "min": 2.0},
    "rate": {"type": "float", "value": 30.0}
  },
  "passes": [
    {"name": "image", "fragment": {"file": "a-image.glsl"}},
    {"name": "flow", "fragment": {"file": "a-flow.glsl"}, "scale": 0.25,
     "params": {"size": {"type": "vec2", "value": [0.5, 0.5]}}},
    {
      "name": "glow",
      "fragment": {
        "file": "glow.glsl"
      }
    }
  ]
}
"#
        );
    }

    #[test]
    fn a_text_that_ghee_wrote_stays_as_ghee_writes_it() {
        let text = serde_json::to_string_pretty(&json!({"version": 1,
            "params": {"a": {"type": "float", "value": 1.0}}}))
        .unwrap()
            + "\n";
        let out = edit(&text, |v| {
            v["params"]["b"] = json!({"type": "int", "value": 2});
        });
        let mut want: Value = serde_json::from_str(&text).unwrap();
        want["params"]["b"] = json!({"type": "int", "value": 2});
        assert_eq!(out, serde_json::to_string_pretty(&want).unwrap() + "\n");
    }
}
