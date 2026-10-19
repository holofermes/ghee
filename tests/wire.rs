//! A scene process that grows its running scene over the wire, as `ghee
//! render` runs it: a layer added mid-run, a layer built again in a new
//! look, and a layer hidden, each frame against the frame of a process that
//! held that layer from its start.
//!
//! The process feeds its array once, on its first tick. A rebuild of the
//! whole scene would leave the array empty, so a frame that matches the
//! process that held the layer from its start also shows that the rest of
//! the scene carried on.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};

/// The frames that a test renders, at FRAME_W x FRAME_H, each tick 0.05 s.
const FRAMES: usize = 8;
const FRAME_W: usize = 48;
const FRAME_H: usize = 24;

/// A directory of one test's own under the temp directory. The drop removes
/// it, with everything that the test wrote in it.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ghee-wire-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The printf line that writes a message of `kind` for `name` with the
/// payload `payload`, its count `count`, as the wire frames it.
fn message(kind: u8, name: &str, count: usize, payload: &[u8]) -> String {
    let mut bytes = vec![kind, name.len() as u8];
    bytes.extend(name.as_bytes());
    bytes.extend((count as u32).to_le_bytes());
    bytes.extend(payload);
    let octal: String = bytes.iter().map(|b| format!("\\{b:03o}")).collect();
    format!("printf '{octal}'\n")
}

/// A kind 6 message: the JSON Patch `ops` over the scene's layers.
fn patch(ops: &Value) -> String {
    let body = ops.to_string();
    message(6, "", body.len(), body.as_bytes())
}

/// The bar that the tests add, at index 1 of pass `main`, over the words.
fn bar() -> Value {
    json!({"name": "bar", "shape": "rect", "x": 4, "y": 12, "width": 30, "height": 8,
           "color": "lime"})
}

/// A lockstep process whose scene is one pass, `main`, red at the level of
/// the array `level`, under the text layer `title`, scrolling, and the
/// layers `held`. It feeds `level` on its first tick, and on tick N it
/// writes the messages `at[N]`.
fn process(held: &[Value], at: &[(u64, String)]) -> String {
    let mut layers = vec![json!({"name": "title", "text": "hi", "x": 0, "y": 0,
                                  "font_size": 12, "scroll": {"speed": 2}})];
    layers.extend(held.iter().cloned());
    let doc = json!({
        "version": 1,
        "lockstep": true,
        "passes": [{"name": "main",
                    "fragment": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }",
                    "arrays": [{"name": "level", "size": 1}],
                    "layers": layers}],
    });
    let mut cases = format!("    0) {}", message(0, "level", 1, &0.5f32.to_le_bytes()));
    for (n, line) in at {
        cases.push_str(&format!("    ;;\n    {n}) {line}"));
    }
    format!(
        "#!/bin/sh\n\
         printf '%s\\n' '{doc}'\n\
         while IFS= read -r line; do\n\
         n=${{line#*\\\"frame\\\":}}; n=${{n%%,*}}\n\
         case $n in\n{cases}    ;;\nesac\n\
         printf '\\005'\n\
         done\n"
    )
}

/// The frames of `ghee render` of the process `body`, each FRAME_W x FRAME_H
/// RGBA, with the messages that ghee refused, one a line.
fn render(tag: &str, body: &str) -> (Vec<Vec<u8>>, String) {
    let dir = Dir::new(tag);
    let path = dir.0.join("grow.sh");
    fs::write(&path, body).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ghee"))
        .arg("render")
        .arg(&path)
        .args(["-o", "-", "--timestep", "0.05"])
        .args(["--frames", &FRAMES.to_string()])
        .args(["--size", &format!("{FRAME_W}x{FRAME_H}")])
        .env("GHEE_ASSETS", "")
        .env("XDG_CACHE_HOME", &dir.0)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{err}");
    let refused: Vec<&str> = err
        .lines()
        .filter(|l| l.ends_with(" changes nothing"))
        .collect();
    assert_eq!(out.stdout.len(), FRAMES * FRAME_W * FRAME_H * 4, "{err}");
    let frames = out
        .stdout
        .as_chunks::<{ FRAME_W * FRAME_H * 4 }>()
        .0
        .iter()
        .map(|f| f.to_vec())
        .collect();
    (frames, refused.join("\n"))
}

/// The count of pixels of `frame` that hold the bar's lime.
fn lime(frame: &[u8]) -> usize {
    frame
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] < 100 && px[1] > 200)
        .count()
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_added_mid_run_draws_as_if_it_was_there_from_the_start() {
    let (plain, _) = render("plain", &process(&[], &[]));
    let (held, _) = render("held", &process(&[bar()], &[]));
    let add = patch(&json!([{"op": "add", "path": "/main/1", "value": bar()}]));
    let (grown, err) = render("grown", &process(&[], &[(3, add)]));
    assert_eq!(err, "");
    assert!(lime(&held[3]) > 0);
    assert_eq!(grown[..3], plain[..3]);
    assert_eq!(grown[3..], held[3..]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_given_a_new_look_mid_run_is_built_again_alone() {
    // blend changes with a rebuild
    let mut added = bar();
    added["blend"] = json!("add");
    let (held, _) = render("held-add", &process(&[added.clone()], &[]));
    let look = patch(&json!([{"op": "replace", "path": "/main/1", "value": added}]));
    let (rebuilt, err) = render("rebuilt", &process(&[bar()], &[(3, look)]));
    assert_eq!(err, "");
    assert_ne!(rebuilt[2], held[2]);
    assert_eq!(rebuilt[3..], held[3..]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_added_layer_hidden_mid_run_draws_nothing() {
    let (plain, _) = render("plain-hide", &process(&[], &[]));
    let add = patch(&json!([{"op": "add", "path": "/main/1", "value": bar()}]));
    let hide = message(1, "bar.visible", 1, &0f32.to_le_bytes());
    let (hidden, err) = render("hidden", &process(&[], &[(1, add), (4, hide)]));
    assert_eq!(err, "");
    assert!(lime(&hidden[3]) > 0);
    assert_eq!(hidden[4..], plain[4..]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_that_the_scene_refuses_is_logged_and_the_run_goes_on() {
    let (plain, _) = render("plain-refused", &process(&[], &[]));
    let mut broken = bar();
    broken["color"] = json!("nosuch");
    let add = patch(&json!([{"op": "add", "path": "/main/-", "value": broken}]));
    let (refused, err) = render("refused", &process(&[], &[(2, add)]));
    assert_eq!(refused, plain);
    assert_eq!(err.lines().count(), 1, "{err}");
    assert!(err.contains("layer 'bar' color"), "{err}");
}
