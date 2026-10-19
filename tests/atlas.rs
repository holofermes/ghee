//! The MTSDF render against exact coverage: the words drawn by an MTSDF layer,
//! against what `<name>_cover` draws on this host's GPU.

use serde_json::json;
use ttf_parser::Face;

use ghee::engine::{probe_dri_devices, Renderer};
use ghee::scene::Scene;

const FONTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
/// DejaVu Sans cut to printable ASCII, TrueType outlines, a kern table.
const SANS: &str = "DejaVuSans-ASCII.ttf";
const LYRICS: &str = "the radio hums a borrowed tune\n\
                      we drove until the map ran out\n\
                      and every streetlight knew our names\n\
                      so sing it back the way it was\n\
                      a little louder, little slower\n\
                      the night is long and so are we\n\
                      hold the note until it breaks";
const W: usize = 1280;
const H: usize = 800;

/// A W x H RGBA renderer on the first GL device that opens.
fn gpu() -> Renderer {
    let mut errs = Vec::new();
    for dev in probe_dri_devices() {
        match Renderer::new(W as i32, H as i32, 4, Some(&dev), false, false) {
            Ok(r) => return r,
            Err(e) => errs.push(format!("{dev}: {e}")),
        }
    }
    panic!(
        "no GL device opens ({}). A host without one runs the tests under \
         RUSTFLAGS=\"--cfg ghee_no_gpu\", which reports them as ignored",
        errs.join("; ")
    );
}

/// The pixels that a layer in SANS at `px` pixels per em draws its first
/// baseline below the em of its layout's ascent: the ascender as FreeType's
/// size metrics round it for Pillow's default anchor, from Pillow's 26.6
/// size request, FT_DivFix, FT_MulFix and the round up to a whole pixel,
/// less the exact one.
fn rounding(px: f64) -> f64 {
    let data = std::fs::read(format!("{FONTS}/{SANS}")).unwrap();
    let face = Face::parse(&data, 0).unwrap();
    let (a, u) = (i64::from(face.ascender()), i64::from(face.units_per_em()));
    let size = (px as f32 * 64.0) as i64;
    let scale = ((size << 16) + (u >> 1)) / u;
    let ascender = (a * scale + 0x8000) >> 16;
    ((ascender + 63) >> 6) as f64 - a as f64 / u as f64 * px
}

/// The lyrics centred at 46 px/em in layer `t`, drawn from the MTSDF atlas,
/// shown or hidden, in a pass of `code`.
fn frame(r: &mut Renderer, code: &str, visible: bool) -> Vec<u8> {
    let doc = json!({
        "version": 1,
        "assets": {"font": {"file": SANS}},
        "passes": [{"name": "main", "fragment": code,
                    "layers": [{"name": "t", "visible": visible, "text": LYRICS,
                                "font_size": 46, "font_family": "font",
                                "text_align": "center", "render": "mtsdf"}]}],
    });
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    sc.load_assets(FONTS).unwrap();
    assert_eq!(r.setup(&sc).unwrap(), "");
    r.set_time(Some(0.0));
    r.render(None).unwrap().to_vec()
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn the_mtsdf_fill_at_46_px_em_matches_exact_coverage() {
    std::env::set_var(
        "XDG_CACHE_HOME",
        std::env::temp_dir().join("ghee-text-cache"),
    );
    let mut r = gpu();
    let cover = frame(
        &mut r,
        &format!(
            "void main() {{\n\
             vec2 px = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);\n\
             vec2 off = (iResolution.xy - t_size * 46.0) * 0.5 + vec2(0.0, {:?});\n\
             float c = t_cover((px - off) / 46.0, 46.0);\n\
             gl_FragColor = vec4(vec3(c), 1.0);\n\
             }}",
            rounding(46.0)
        ),
        false,
    );
    let fill = frame(
        &mut r,
        "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
        true,
    );
    let errs: Vec<f64> = cover
        .iter()
        .zip(&fill)
        .step_by(4)
        .filter(|(c, f)| **c > 0 || **f > 0)
        .map(|(c, f)| f64::from(c.abs_diff(*f)))
        .collect();
    let mean = errs.iter().sum::<f64>() / errs.len() as f64;
    let max = errs.iter().copied().fold(0.0, f64::max);
    eprintln!(
        "MTSDF render against t_cover at 46 px/em: {} px, mean {mean:.2}/255, max {max:.0}/255",
        errs.len()
    );
    assert!(errs.len() > 50_000, "{} pixels compared", errs.len());
    assert!(mean < 4.0, "mean {mean:.2}/255");
}
