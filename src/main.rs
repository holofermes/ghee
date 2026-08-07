use std::process::exit;

use anyhow::{bail, Result};

use ghee::engine::Renderer;
use ghee::scene::{Pass, Scene};
use ghee::{cli, golden};

/// Two checks on the engine itself. They stay out of `ghee --help`.
const EXTRAS: &str =
    "usage: ghee smoke [dri] [components] | ghee golden --manifest PATH [--dump DIR | --pin]";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();
    let code = match args.first().map(String::as_str) {
        Some("smoke") => smoke(&rest).map(|()| 0),
        Some("golden") => golden(&rest),
        _ => exit(cli::main(args)),
    };
    match code {
        Ok(code) => exit(code),
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}

/// Render one flat pass at a known colour and read it back. One command covers
/// the engine, the scene assembly and the readback.
fn smoke(args: &[&str]) -> Result<()> {
    let dri = args.first().copied().unwrap_or("/dev/dri/renderD128");
    let components: i32 = match args.get(1) {
        Some(c) => c.parse()?,
        None => 3,
    };
    let mut r = Renderer::new(64, 64, components, Some(dri), false, false)?;
    let scene = Scene::of(vec![Pass::of(
        "main",
        "void main(){ gl_FragColor = vec4(0.25,0.5,0.75,1.0); }",
    )?])?;
    let err = r.setup(&scene)?;
    if !err.is_empty() {
        bail!("scene build failed: {err}");
    }
    let frame = r.render(None)?;
    println!("RUST_SMOKE_OK pixel={},{},{}", frame[0], frame[1], frame[2]);
    Ok(())
}

fn golden(args: &[&str]) -> Result<i32> {
    match args {
        ["--manifest", path] => golden::run(path, None, false),
        ["--manifest", path, "--dump", dir] => golden::run(path, Some(dir), false),
        ["--manifest", path, "--pin"] => golden::run(path, None, true),
        _ => bail!("{EXTRAS}"),
    }
}
