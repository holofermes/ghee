//! `ghee bench`: the benchmark suite, run from the binary.
//!
//! The timing comes from the engine's own fenced profiling.
//! draw_ms is the GPU render of every pass behind a glFinish.
//! read_ms is the glReadPixels readback.
//! host_ms is the rest of the frame.
//!
//! No case is sized for a particular machine. A run warms each case until its
//! frame time stops falling. It then measures the frame cost, and it derives
//! the frame count from that cost.
//!
//! The result file is schema 2. The harness compares that shape.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::console::examples_dir;
use crate::engine::{probe_dri_devices, Feed, Renderer};
use crate::filters::apply_filters;
use crate::procedural::load_scene;
use crate::scene::{ArraySize, Scene};

const SCHEMA: i64 = 2;
const PROBE_ROUNDS: usize = 3;
const PROBE_S: f64 = 0.05;

/// A derived frame count snaps to one of these rungs.
/// The spacing is a factor of two, so a cost measurement moves 41% to cross a
/// boundary. The same case then lands on the same rung run after run.
const FRAME_RUNGS: [i64; 4] = [60, 120, 240, 480];

/// The smallest useful frame count pooled across the kept repetitions.
/// Below it the 1st percentile degrades into the minimum of the run.
const MIN_POOLED: i64 = 200;

/// A run with no --suite measures this suite.
const SUITE: &str = r#"{
  "reps": 10,
  "rep_ms": 250,
  "warmup_min_s": 0.25,
  "warmup_max_s": 3.0,
  "components": 4,
  "timestep": 0.0166667,
  "cases": [
    { "scene": "bars", "size": [128, 32] },
    { "scene": "bars", "size": [320, 240] },
    { "scene": "bars", "size": [320, 240], "filters": ["dither"] },
    { "scene": "twopass", "size": [320, 240] },
    { "scene": "trail", "size": [320, 240] },
    { "scene": "blur", "size": [320, 240] },
    { "scene": "mix", "size": [320, 240] },
    { "scene": "trail", "size": [800, 480] }
  ]
}
"#;

#[derive(Args)]
pub struct BenchArgs {
    #[arg(
        long,
        value_name = "PATH",
        help = "suite JSON (default: the built-in suite)"
    )]
    suite: Option<String>,

    #[arg(long, help = "result file, or - for stdout")]
    out: Option<String>,

    #[arg(
        long,
        value_name = "DIR",
        help = "directory the default result file lands in (default: the working \
                directory)"
    )]
    results_dir: Option<String>,

    #[arg(long = "case", value_name = "NAME", help = "run only these case names")]
    case: Vec<String>,

    #[arg(
        long,
        help = "same frame count for every case, instead of one derived per case"
    )]
    frames: Option<i64>,

    #[arg(
        long,
        value_name = "RESULT.json",
        help = "take each case's frame count from an earlier result, so the two are \
                dimensioned identically whatever this machine measures today"
    )]
    like: Option<String>,

    #[arg(long, help = "wall clock a derived repetition aims at")]
    rep_ms: Option<f64>,

    #[arg(long)]
    warmup_min_s: Option<f64>,

    #[arg(long)]
    warmup_max_s: Option<f64>,

    #[arg(long)]
    reps: Option<i64>,

    #[arg(long)]
    components: Option<i32>,

    #[arg(long)]
    timestep: Option<f64>,

    #[arg(long, help = "GPU node, ghee's own probe order by default")]
    dri: Option<String>,

    #[arg(long, help = "pin this process to one CPU")]
    pin: Option<i64>,

    #[arg(long, default_value = "", help = "free text stored in the result")]
    note: String,

    #[arg(
        long,
        help = "warm up, print the plan the run would use, measure nothing"
    )]
    plan_only: bool,
}

// -------------------------------------------------------------- statistics --

fn sorted(xs: &[f64]) -> Vec<f64> {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

fn median(xs: &[f64]) -> f64 {
    let v = sorted(xs);
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The q-th percentile of a sorted sample, by index.
/// A pooled sample of 200 frames then still has a 1st percentile.
fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let i = (q * v.len() as f64) as usize;
    v[i.min(v.len() - 1)]
}

/// Sample standard deviation as a percentage of the mean.
fn rel_sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    100.0 * var.sqrt() / mean
}

fn stats(xs: &[f64]) -> Map<String, Value> {
    let v = sorted(xs);
    let mut m = Map::new();
    m.insert("median".into(), json!(median(&v)));
    m.insert("min".into(), json!(v.first().copied().unwrap_or(0.0)));
    m.insert("p1".into(), json!(pct(&v, 0.01)));
    m.insert("p95".into(), json!(pct(&v, 0.95)));
    m.insert("n".into(), json!(v.len()));
    m
}

/// Reduce one list of frame times per repetition to the numbers behind a
/// comparison.
///
/// `stable` is the comparison statistic. It is the 1st percentile of every
/// measured frame. Interference only adds time, so the bottom of the
/// distribution is the frame that met no interference.
///
/// `spread_pct` is the gap in that statistic between the odd and the even
/// repetitions. Drift over the run hits both halves alike, so `spread_pct`
/// hides it. `rep_spread_pct` gives the other view: the gap between the
/// individual repetitions.
///
/// The summary drops the first repetition. A machine settles further after the
/// warmup.
fn summarise(per_rep: &[Vec<f64>]) -> Map<String, Value> {
    let kept: &[Vec<f64>] = if per_rep.len() > 2 {
        &per_rep[1..]
    } else {
        per_rep
    };
    let pooled: Vec<f64> = kept.iter().flatten().copied().collect();
    let mut out = stats(&pooled);
    let rep_medians: Vec<f64> = kept.iter().map(|r| median(r)).collect();
    let rep_mins: Vec<f64> = kept
        .iter()
        .map(|r| sorted(r).first().copied().unwrap_or(0.0))
        .collect();
    let stable = out["p1"].as_f64().unwrap_or(0.0);
    let halves: Vec<f64> = (0..2)
        .map(|side| {
            let v: Vec<f64> = kept
                .iter()
                .skip(side)
                .step_by(2)
                .flatten()
                .copied()
                .collect();
            pct(&sorted(&v), 0.01)
        })
        .collect();
    let lo = halves[0].min(halves[1]);
    let hi = halves[0].max(halves[1]);
    out.insert("rep_medians".into(), json!(rep_medians));
    out.insert("rep_mins".into(), json!(rep_mins));
    out.insert("stable".into(), json!(stable));
    out.insert(
        "spread_pct".into(),
        json!(if lo > 0.0 {
            100.0 * (hi - lo) / lo
        } else {
            0.0
        }),
    );
    out.insert("rep_spread_pct".into(), json!(rel_sd(&rep_mins)));
    out
}

/// The cost of a frame measurement on an empty frame.
/// A delta below this value comes from the clock.
fn timer_floor(samples: usize) -> Value {
    let mut xs = Vec::with_capacity(samples);
    for _ in 0..samples {
        let a = Instant::now();
        let b = Instant::now();
        xs.push(b.duration_since(a).as_secs_f64() * 1000.0);
    }
    let v = sorted(&xs);
    json!({
        "median_ms": v[v.len() / 2],
        "p99_ms": v[(0.99 * v.len() as f64) as usize],
        "max_ms": v[v.len() - 1],
    })
}

// -------------------------------------------------------------------- plan --

fn nearest_rung(x: f64) -> i64 {
    if x <= 0.0 {
        return FRAME_RUNGS[FRAME_RUNGS.len() - 1];
    }
    let mut best = FRAME_RUNGS[0];
    let mut score = (x / best as f64).ln().abs();
    for &r in &FRAME_RUNGS[1..] {
        let s = (x / r as f64).ln().abs();
        if s < score {
            best = r;
            score = s;
        }
    }
    best
}

/// At this cost, a rep_ms repetition buys this many frames.
/// The second value names which bound the count ran into, or None.
fn wanted_frames(ms_per_frame: f64, rep_ms: f64) -> (f64, Option<&'static str>) {
    let last = FRAME_RUNGS[FRAME_RUNGS.len() - 1] as f64;
    let wanted = if ms_per_frame > 0.0 {
        rep_ms / ms_per_frame
    } else {
        last
    };
    let bound = if wanted < FRAME_RUNGS[0] as f64 {
        Some("floor")
    } else if wanted > last {
        Some("cap")
    } else {
        None
    };
    (wanted, bound)
}

struct Warm {
    frames: usize,
    seconds: f64,
    settled: bool,
}

struct PlanRow {
    name: String,
    ms_per_frame: f64,
    warm: Warm,
    frames: i64,
    wanted: f64,
    bound: Option<&'static str>,
    rep_s: f64,
    total_s: f64,
    pooled: i64,
    thin: bool,
}

impl PlanRow {
    fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "ms_per_frame": self.ms_per_frame,
            "warm_s": self.warm.seconds,
            "warm_frames": self.warm.frames,
            "settled": self.warm.settled,
            "frames": self.frames,
            "wanted": self.wanted,
            "bound": self.bound,
            "rep_s": self.rep_s,
            "total_s": self.total_s,
            "pooled": self.pooled,
            "thin": self.thin,
        })
    }
}

fn dimension(
    warm: Vec<(String, Warm)>,
    cost: &HashMap<String, Vec<f64>>,
    reps: i64,
    rep_ms: f64,
    fixed_frames: Option<i64>,
    like: Option<&HashMap<String, i64>>,
) -> Vec<PlanRow> {
    let mut rows = Vec::new();
    for (name, w) in warm {
        let ms = median(&cost[&name]);
        let (wanted, mut bound) = wanted_frames(ms, rep_ms);
        let frames = if let Some(f) = fixed_frames {
            bound = None;
            f
        } else if let Some(f) = like.and_then(|l| l.get(&name)) {
            bound = None;
            *f
        } else {
            nearest_rung(wanted)
        };
        let pooled = frames * (reps - 1).max(1);
        rows.push(PlanRow {
            name,
            ms_per_frame: ms,
            warm: w,
            frames,
            wanted,
            bound,
            rep_s: frames as f64 * ms / 1000.0,
            total_s: reps as f64 * frames as f64 * ms / 1000.0,
            pooled,
            thin: pooled < MIN_POOLED,
        });
    }
    rows
}

fn format_plan(
    rows: &[PlanRow],
    reps: i64,
    rep_ms: f64,
    fixed_frames: Option<i64>,
    like: Option<&str>,
) -> String {
    let how = if let Some(f) = fixed_frames {
        format!("frames fixed at {f}")
    } else if let Some(l) = like {
        format!("frames taken from {l}")
    } else {
        format!("frames per case for a {} ms repetition", num(rep_ms))
    };
    let mut out = vec![format!("bench: {how}, {reps} reps"), String::new()];
    out.push(format!(
        "{:<24} {:>9} {:>9} {:>7} {:>7} {:>11}",
        "case", "ms/frame", "warmup", "frames", "rep", "case total"
    ));
    for r in rows {
        let mut note = String::new();
        if r.bound == Some("floor") {
            note = format!("  floor (a {} ms rep wants {:.0})", num(rep_ms), r.wanted);
        } else if r.bound == Some("cap") {
            note = format!("  cap (a {} ms rep wants {:.0})", num(rep_ms), r.wanted);
        }
        if !r.warm.settled {
            note += "  still warming at the ceiling";
        }
        if r.thin {
            note += &format!("  THIN: {} frames pooled, under {MIN_POOLED}", r.pooled);
        }
        out.push(format!(
            "{:<24} {:>9.4} {:>8.2}s {:>7} {:>6.2}s {:>10.1}s{note}",
            r.name, r.ms_per_frame, r.warm.seconds, r.frames, r.rep_s, r.total_s
        ));
    }
    out.push(String::new());
    out.push(format!(
        "bench: {} case{}, about {:.0}s of measured frames",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" },
        rows.iter().map(|r| r.total_s).sum::<f64>()
    ));
    out.join("\n")
}

/// A float in `%g` form.
/// A whole number loses its trailing zeros.
fn num(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e15 {
        format!("{:.0}", x)
    } else {
        format!("{x}")
    }
}

// ------------------------------------------------------------------- cases --

struct CaseSpec {
    scene: String,
    size: (i32, i32),
    filters: Vec<String>,
}

fn case_name(c: &CaseSpec) -> String {
    let tag = c.filters.join("+");
    let base = format!("{}@{}x{}", c.scene, c.size.0, c.size.1);
    if tag.is_empty() {
        base
    } else {
        format!("{base}+{tag}")
    }
}

/// Precompute every array the suite feeds.
/// The provider then indexes a list, and its arithmetic stays out of host_ms.
fn feed_table(n: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|f| {
            (0..n)
                .map(|k| (0.5 + 0.5 * (f as f64 / 15.0 + k as f64).sin()) as f32)
                .collect()
        })
        .collect()
}

fn array_len(size: ArraySize) -> usize {
    match size {
        ArraySize::Linear(n) => n as usize,
        ArraySize::Grid(w, h) => (w * h) as usize,
    }
}

/// Resolve a scene name.
/// The search order is the name itself, then $GHEE_SCENES, then the bundled
/// examples.
fn find_scene(name: &str) -> Result<String> {
    let mut dirs: Vec<String> = std::env::var("GHEE_SCENES")
        .unwrap_or_default()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(String::from)
        .collect();
    dirs.extend(examples_dir());
    for stem in std::iter::once(name.to_string()).chain(
        dirs.iter()
            .map(|d| Path::new(d).join(name).to_string_lossy().into_owned()),
    ) {
        for p in [
            stem.clone(),
            format!("{stem}.scene.json"),
            format!("{stem}.py"),
        ] {
            if Path::new(&p).exists() {
                return Ok(p);
            }
        }
    }
    bail!("no scene '{name}' (searched: cwd, {})", dirs.join(", "))
}

/// One scene at one size, held open for the whole run.
struct Case {
    spec: CaseSpec,
    name: String,
    w: i32,
    h: i32,
    passes: usize,
    ts: f64,
    frames: i64,
    r: Renderer,
    /// array name -> element count, in declaration order.
    declared: Vec<(String, usize)>,
    tables: HashMap<String, Vec<Vec<f32>>>,
    draw: Vec<Vec<f64>>,
    read: Vec<Vec<f64>>,
    host: Vec<Vec<f64>>,
    whole: Vec<Vec<f64>>,
}

impl Case {
    fn open(spec: CaseSpec, components: i32, timestep: f64, dri: &str) -> Result<Case> {
        let name = case_name(&spec);
        let (w, h) = spec.size;
        let path = find_scene(&spec.scene)?;
        let mut scene: Scene = load_scene(&path, &[], false)?;
        apply_filters(&mut scene, &spec.filters, None)?;
        let mut declared: Vec<(String, usize)> = Vec::new();
        for p in &scene.passes {
            for a in &p.arrays {
                let n = array_len(a.size);
                match declared.iter_mut().find(|(name, _)| *name == a.name) {
                    Some(slot) => slot.1 = n,
                    None => declared.push((a.name.clone(), n)),
                }
            }
        }
        let mut r = Renderer::new(w, h, components, Some(dri), false, false)?;
        let err = r.setup(&scene)?;
        if !err.is_empty() {
            bail!("{err}");
        }
        r.set_profiling(true);
        Ok(Case {
            spec,
            name,
            w,
            h,
            passes: scene.passes.len(),
            ts: timestep,
            frames: 0,
            r,
            declared,
            tables: HashMap::new(),
            draw: Vec::new(),
            read: Vec::new(),
            host: Vec::new(),
            whole: Vec::new(),
        })
    }

    fn load_tables(&mut self, count: usize) {
        self.tables = self
            .declared
            .iter()
            .map(|(name, n)| (name.clone(), feed_table(*n, count)))
            .collect();
    }

    /// Render frame i at time i*timestep.
    /// Every repetition of every run then renders the same work.
    fn render_at(&mut self, i: usize) -> Result<()> {
        let Case {
            r,
            tables,
            declared,
            ts,
            ..
        } = self;
        r.set_time(Some(i as f64 * *ts));
        if declared.is_empty() {
            r.render(None)?;
        } else {
            let mut provide = |name: &str, _size: ArraySize| {
                let t = &tables[name];
                Feed::Data(t[i % t.len()].clone())
            };
            r.render(Some(&mut provide))?;
        }
        Ok(())
    }

    /// Warm one case until its frame time stops falling.
    ///
    /// The warmup cycles a table of more than one frame. The engine skips the
    /// texture upload for an unchanged feed. A warmup on constant data then
    /// measures a cheaper frame than the run.
    fn warm(&mut self, floor_s: f64, ceiling_s: f64) -> Result<Warm> {
        const WINDOW_S: f64 = 0.08;
        const TOL: f64 = 0.02;
        self.load_tables(64);
        let start = Instant::now();
        let (mut stamps, mut times): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut settled = false;
        let mut checked_at = 0.0;
        loop {
            let t0 = start.elapsed().as_secs_f64();
            self.render_at(times.len())?;
            let t1 = start.elapsed().as_secs_f64();
            stamps.push(t1);
            times.push((t1 - t0) * 1000.0);
            if t1 >= ceiling_s {
                break;
            }
            if t1 >= floor_s && t1 - checked_at >= WINDOW_S {
                checked_at = t1;
                let last = window(&stamps, &times, t1 - WINDOW_S, t1);
                let before = window(&stamps, &times, t1 - 2.0 * WINDOW_S, t1 - WINDOW_S);
                if let (Some(a), Some(b)) = (least(&last), least(&before)) {
                    if a >= b * (1.0 - TOL) {
                        settled = true;
                        break;
                    }
                }
            }
        }
        self.tables.clear();
        Ok(Warm {
            frames: times.len(),
            seconds: (start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
            settled,
        })
    }

    /// Frame times for a short slice, which only size the run.
    /// The probe interleaves the cases like the measurement, after every case
    /// is warm. A GPU that clocks up then hits every case alike.
    fn probe(&mut self, seconds: f64) -> Result<Vec<f64>> {
        self.load_tables(64);
        let start = Instant::now();
        let mut out = Vec::new();
        let mut n = 0;
        while start.elapsed().as_secs_f64() < seconds {
            let t0 = Instant::now();
            self.render_at(n)?;
            out.push(t0.elapsed().as_secs_f64() * 1000.0);
            n += 1;
        }
        self.tables.clear();
        Ok(out)
    }

    fn size_to(&mut self, frames: i64) {
        self.frames = frames;
        self.load_tables(frames as usize);
    }

    /// One repetition. It always renders frames 0 to frames-1.
    fn block(&mut self) -> Result<()> {
        let n = self.frames as usize;
        let (mut d, mut rd, mut hs, mut wh) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for i in 0..n {
            let t0 = Instant::now();
            self.render_at(i)?;
            let wall = t0.elapsed().as_secs_f64() * 1000.0;
            let (dm, rm) = self.r.timings();
            d.push(dm);
            rd.push(rm);
            hs.push((wall - dm - rm).max(0.0));
            wh.push(wall);
        }
        self.draw.push(d);
        self.read.push(rd);
        self.host.push(hs);
        self.whole.push(wh);
        Ok(())
    }

    fn result(&self) -> Value {
        let frame_ms = summarise(&self.whole);
        let stable = frame_ms["stable"].as_f64().unwrap_or(0.0);
        json!({
            "name": self.name,
            "scene": self.spec.scene,
            "width": self.w,
            "height": self.h,
            "filters": self.spec.filters,
            "passes": self.passes,
            "frames": self.frames,
            "draw_ms": summarise(&self.draw),
            "read_ms": summarise(&self.read),
            "host_ms": summarise(&self.host),
            "frame_ms": frame_ms,
            "fps": if stable > 0.0 { 1000.0 / stable } else { 0.0 },
        })
    }
}

fn window(stamps: &[f64], times: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    stamps
        .iter()
        .zip(times)
        .filter(|(t, _)| **t > lo && **t <= hi)
        .map(|(_, ms)| *ms)
        .collect()
}

fn least(xs: &[f64]) -> Option<f64> {
    xs.iter().copied().reduce(f64::min)
}

// -------------------------------------------------------------------- host --

fn read_text(path: &str) -> Option<String> {
    let raw = fs::read(path).ok()?;
    Some(
        String::from_utf8_lossy(&raw)
            .trim_matches(|c| c == '\0' || c == '\n' || c == ' ')
            .to_string(),
    )
}

/// A command's stdout.
/// None means the command is missing, or it fails, or it prints nothing.
fn run_cmd(argv: &[&str]) -> Option<String> {
    let out = Command::new(argv[0]).args(&argv[1..]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn field(text: &str, key: &str, sep: char) -> Option<String> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(v) = rest.strip_prefix(sep) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn os_release() -> Option<String> {
    let txt = read_text("/etc/os-release")?;
    let v = field(&txt, "PRETTY_NAME", '=')?;
    Some(v.trim_matches('"').to_string())
}

fn cpu_model() -> Option<String> {
    let txt = read_text("/proc/cpuinfo")?;
    field(&txt, "model name", ':').or_else(|| field(&txt, "Model", ':'))
}

/// Every /sys CPU directory, in lexicographic order.
fn cpu_dirs() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            n.strip_prefix("cpu")
                .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(|n| format!("/sys/devices/system/cpu/{n}"))
        .collect();
    out.sort();
    out
}

fn cpu_ids() -> Vec<i64> {
    let mut ids: Vec<i64> = cpu_dirs()
        .iter()
        .filter_map(|d| d.rsplit("/cpu").next().and_then(|n| n.parse().ok()))
        .collect();
    ids.sort_unstable();
    ids
}

fn dedup_push(seen: &mut Vec<String>, value: &str) {
    if !value.is_empty() && !seen.iter().any(|s| s == value) {
        seen.push(value.to_string());
    }
}

/// Every core's cpufreq governor, deduplicated.
fn governors() -> Vec<String> {
    let mut seen = Vec::new();
    for d in cpu_dirs() {
        if let Some(g) = read_text(&format!("{d}/cpufreq/scaling_governor")) {
            dedup_push(&mut seen, &g);
        }
    }
    seen
}

/// The kernel's own list of governors.
fn governors_available() -> Vec<String> {
    let mut seen = Vec::new();
    for d in cpu_dirs() {
        if let Some(line) = read_text(&format!("{d}/cpufreq/scaling_available_governors")) {
            for g in line.split_whitespace() {
                dedup_push(&mut seen, g);
            }
        }
    }
    seen
}

/// The kernel driver behind a /dev/dri node, such as i915, v3d or vc4.
fn drm_driver(node: &str) -> Option<String> {
    let name = Path::new(node).file_name()?.to_string_lossy().into_owned();
    let txt = read_text(&format!("/sys/class/drm/{name}/device/uevent"))?;
    field(&txt, "DRIVER", '=')
}

fn mesa_version() -> Option<String> {
    if let Some(v) = run_cmd(&["dpkg-query", "-W", "-f=${Version}", "libgl1-mesa-dri"]) {
        return Some(v);
    }
    let info = run_cmd(&["glxinfo", "-B"])?;
    let at = info.find("Mesa ")? + "Mesa ".len();
    let v: String = info[at..]
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != ')')
        .collect();
    v.starts_with(|c: char| c.is_ascii_digit()).then_some(v)
}

/// Per-core thermal counts and the package count, each read once.
///
/// These files repeat one another. package_throttle_count is one chip-wide
/// number, and the kernel exposes it through every CPU directory.
/// core_throttle_count is per physical core, so both hyperthreads report it.
fn x86_throttle(cpus: &[i64]) -> Option<Value> {
    let (mut cores, mut package) = (Map::new(), Map::new());
    for cpu in cpus {
        let d = format!("/sys/devices/system/cpu/cpu{cpu}");
        let pkg = read_text(&format!("{d}/topology/physical_package_id"));
        let Some(core) = read_text(&format!("{d}/topology/core_id")) else {
            continue;
        };
        let pkg = pkg.unwrap_or_else(|| "None".to_string());
        if let Some(c) = read_text(&format!("{d}/thermal_throttle/core_throttle_count"))
            .and_then(|v| v.parse::<i64>().ok())
        {
            cores.insert(format!("{pkg}:{core}"), json!(c));
        }
        if let Some(p) = read_text(&format!("{d}/thermal_throttle/package_throttle_count"))
            .and_then(|v| v.parse::<i64>().ok())
        {
            package.insert(pkg, json!(p));
        }
    }
    if cores.is_empty() && package.is_empty() {
        return None;
    }
    Some(json!({ "cores": cores, "package": package }))
}

const VCGENCMD_BITS: [(u32, &str); 8] = [
    (0, "under-voltage now"),
    (1, "arm frequency capped now"),
    (2, "throttled now"),
    (3, "soft temperature limit now"),
    (16, "under-voltage since boot"),
    (17, "arm frequency capped since boot"),
    (18, "throttled since boot"),
    (19, "soft temperature limit since boot"),
];

/// A throttle reading, taken before and after a run.
///
/// `gating` states whether a change invalidates the run.
/// On a Pi a change gates: get_throttled is the SoC's verdict on itself.
/// On x86 the package counter is chip-wide, and other tenants move it too.
/// That reading reports only, and it does not gate.
fn throttle_counter() -> Option<Value> {
    if let Some(v) = run_cmd(&["vcgencmd", "get_throttled"]) {
        if let Some(at) = v.find("0x") {
            let hex: String = v[at + 2..]
                .chars()
                .take_while(char::is_ascii_hexdigit)
                .collect();
            if let Ok(bits) = u64::from_str_radix(&hex, 16) {
                let flags: Vec<&str> = VCGENCMD_BITS
                    .iter()
                    .filter(|(bit, _)| bits & (1 << bit) != 0)
                    .map(|(_, name)| *name)
                    .collect();
                return Some(json!({
                    "source": "vcgencmd",
                    "gating": true,
                    "value": bits,
                    "flags": flags,
                }));
            }
        }
    }
    let cpus = cpu_ids();
    let counts = x86_throttle(&cpus)?;
    Some(json!({
        "source": "thermal_throttle",
        "gating": false,
        "cpus": cpus,
        "value": counts,
    }))
}

fn git_state(src: Option<&str>) -> Value {
    let null = json!({"commit": Value::Null, "dirty": Value::Null, "describe": Value::Null});
    let Some(src) = src else { return null };
    let Some(head) = run_cmd(&["git", "-C", src, "rev-parse", "HEAD"]) else {
        return null;
    };
    let status = run_cmd(&["git", "-C", src, "status", "--porcelain"]);
    json!({
        "commit": head,
        "dirty": status.is_some(),
        "describe": run_cmd(&["git", "-C", src, "describe", "--always", "--dirty"]),
    })
}

fn sha256_file(path: &Path) -> Option<String> {
    let blob = fs::read(path).ok()?;
    Some(
        Sha256::digest(&blob)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()[..12]
            .to_string(),
    )
}

/// A benchmark result is comparable within these facts.
/// A fact is null when this call cannot derive it.
fn facts(dri: &str) -> Value {
    let exe = std::env::current_exe().unwrap_or_default();
    let src = std::env::var("GHEE_SRC")
        .ok()
        .filter(|s| !s.is_empty() && Path::new(s).is_dir());
    let govs = governors();
    let pinned = if govs.is_empty() {
        Value::Null
    } else {
        json!(govs == ["performance"])
    };
    let in_repo = src.as_ref().map(|s| exe.starts_with(s));
    let git = git_state(src.as_deref());
    json!({
        "host": read_text("/proc/sys/kernel/hostname"),
        "arch": run_cmd(&["uname", "-m"]),
        "kernel": read_text("/proc/sys/kernel/osrelease"),
        "os": os_release(),
        "model": read_text("/proc/device-tree/model")
            .or_else(|| read_text("/sys/devices/virtual/dmi/id/product_name")),
        "cpu": cpu_model(),
        "cpus": cpu_dirs().len(),
        "virt": run_cmd(&["systemd-detect-virt"]),
        "governors": govs,
        "governors_available": governors_available(),
        "governor_pinned": pinned,
        "dri": dri,
        "drm_driver": drm_driver(dri),
        "mesa": mesa_version(),
        "software_rasteriser": std::env::var("LIBGL_ALWAYS_SOFTWARE").as_deref() == Ok("1"),
        "ghee": env!("CARGO_PKG_VERSION"),
        "ghee_path": exe.to_string_lossy(),
        "ghee_src": src,
        "ghee_origin": "ghee binary",
        "ghee_in_repo": in_repo,
        "git": git,
        // The sources of this binary are absent from the host that runs it.
        "source_sha": "rust-binary",
        "engine": {"path": exe.to_string_lossy(), "sha": sha256_file(&exe)},
    })
}

fn load_avg() -> Value {
    let txt = read_text("/proc/loadavg").unwrap_or_default();
    let xs: Vec<f64> = txt
        .split_whitespace()
        .take(3)
        .filter_map(|s| s.parse().ok())
        .collect();
    json!(xs)
}

// glibc's cpu_set_t is 1024 bits of CPU mask.
#[repr(C)]
struct CpuSet([u64; 16]);

extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
}

fn pin_to(cpu: i64) -> Result<()> {
    if !(0..1024).contains(&cpu) {
        bail!("--pin {cpu}: the affinity mask holds CPUs 0 to 1023");
    }
    let mut set = CpuSet([0; 16]);
    set.0[(cpu / 64) as usize] = 1 << (cpu % 64);
    if unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set) } != 0 {
        bail!("--pin {cpu}: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

// -------------------------------------------------------------------- time --

/// (year, month, day) for a count of days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn epoch_secs(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// UTC as "%Y-%m-%dT%H:%M:%SZ" when iso is true.
/// Otherwise UTC as "%Y%m%d-%H%M%S", for a file name.
fn utc(secs: i64, iso: bool) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    if iso {
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
    } else {
        format!("{y:04}{m:02}{d:02}-{hh:02}{mm:02}{ss:02}")
    }
}

// --------------------------------------------------------------------- run --

fn suite_cases(suite: &Value) -> Result<Vec<CaseSpec>> {
    let cases = suite
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("suite: 'cases' must be a list"))?;
    cases
        .iter()
        .map(|c| {
            let scene = c
                .get("scene")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("suite: a case needs a 'scene' name"))?;
            let size = c
                .get("size")
                .and_then(Value::as_array)
                .filter(|s| s.len() == 2)
                .ok_or_else(|| anyhow!("suite: case '{scene}' needs a 'size' [w, h]"))?;
            let dim = |i: usize| -> Result<i32> {
                size[i]
                    .as_i64()
                    .and_then(|v| i32::try_from(v).ok())
                    .ok_or_else(|| anyhow!("suite: case '{scene}' size is not two ints"))
            };
            Ok(CaseSpec {
                scene: scene.to_string(),
                size: (dim(0)?, dim(1)?),
                filters: c
                    .get("filters")
                    .and_then(Value::as_array)
                    .map(|f| {
                        f.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        })
        .collect()
}

/// The suite's settings with the command-line overrides applied.
/// The keys come in the result file's own order.
fn config(suite: &Value, a: &BenchArgs) -> Result<Map<String, Value>> {
    let mut cfg = Map::new();
    for k in [
        "reps",
        "rep_ms",
        "warmup_min_s",
        "warmup_max_s",
        "components",
        "timestep",
    ] {
        let v = suite
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow!("suite: missing '{k}'"))?;
        cfg.insert(k.to_string(), v);
    }
    cfg.insert(
        "frames".to_string(),
        suite.get("frames").cloned().unwrap_or(Value::Null),
    );
    let overrides: [(&str, Option<Value>); 7] = [
        ("reps", a.reps.map(Value::from)),
        ("rep_ms", a.rep_ms.map(Value::from)),
        ("warmup_min_s", a.warmup_min_s.map(Value::from)),
        ("warmup_max_s", a.warmup_max_s.map(Value::from)),
        ("components", a.components.map(Value::from)),
        ("timestep", a.timestep.map(Value::from)),
        ("frames", a.frames.map(Value::from)),
    ];
    for (k, v) in overrides {
        if let Some(v) = v {
            cfg.insert(k.to_string(), v);
        }
    }
    cfg.insert("pin".to_string(), json!(a.pin));
    Ok(cfg)
}

fn number(cfg: &Map<String, Value>, key: &str) -> Result<f64> {
    cfg[key]
        .as_f64()
        .ok_or_else(|| anyhow!("suite: '{key}' must be a number"))
}

/// Every case that opens. A case that fails to open drops out of the run.
/// Its name and its error land in the result file, and the comparison reports
/// them.
fn open_cases(
    specs: Vec<CaseSpec>,
    components: i32,
    timestep: f64,
    dri: &str,
    skipped: &mut Vec<Value>,
) -> Vec<Case> {
    let mut out = Vec::new();
    for spec in specs {
        let name = case_name(&spec);
        match Case::open(spec, components, timestep, dri) {
            Ok(c) => out.push(c),
            Err(e) => {
                skipped.push(json!({"name": name, "error": format!("{e:#}")}));
                eprintln!("bench: SKIPPED {name}: {e:#}");
            }
        }
    }
    out
}

pub fn run(a: &BenchArgs) -> Result<()> {
    let exe = std::env::current_exe().unwrap_or_default();
    eprintln!(
        "bench: ghee {} from {} (ghee binary)",
        env!("CARGO_PKG_VERSION"),
        exe.display()
    );
    if let Some(cpu) = a.pin {
        pin_to(cpu)?;
    }

    let text = match &a.suite {
        Some(p) => fs::read_to_string(p).with_context(|| format!("cannot read suite {p}"))?,
        None => SUITE.to_string(),
    };
    let suite: Value = serde_json::from_str(&text).context("suite: not JSON")?;
    let cfg = config(&suite, a)?;
    let reps = cfg["reps"]
        .as_i64()
        .ok_or_else(|| anyhow!("suite: 'reps' must be an integer"))?;
    let rep_ms = number(&cfg, "rep_ms")?;
    let timestep = number(&cfg, "timestep")?;
    // The engine reads back GL_RGB for 3 components and GL_RGBA otherwise.
    let components = match cfg["components"].as_i64() {
        Some(c @ (3 | 4)) => c as i32,
        _ => bail!(
            "suite: 'components' is 3 (RGB) or 4 (RGBA), got {}",
            cfg["components"]
        ),
    };
    let fixed_frames = cfg["frames"].as_i64();

    let dri = match &a.dri {
        Some(d) => d.clone(),
        None => probe_dri_devices()
            .into_iter()
            .find(|d| Path::new(d).exists())
            .ok_or_else(|| anyhow!("no GPU node: pass --dri or set $GHEE_DRI"))?,
    };

    let like: Option<HashMap<String, i64>> = match &a.like {
        None => None,
        Some(p) => {
            let doc: Value = serde_json::from_str(&fs::read_to_string(p)?)
                .with_context(|| format!("{p}: not a result file"))?;
            Some(
                doc["results"]
                    .as_array()
                    .ok_or_else(|| anyhow!("{p}: no results"))?
                    .iter()
                    .filter_map(|r| Some((r["name"].as_str()?.to_string(), r["frames"].as_i64()?)))
                    .collect(),
            )
        }
    };

    let mut specs = suite_cases(&suite)?;
    if !a.case.is_empty() {
        specs.retain(|c| a.case.contains(&case_name(c)));
        let have: Vec<String> = specs.iter().map(case_name).collect();
        let missing: Vec<&String> = a.case.iter().filter(|w| !have.contains(w)).collect();
        if !missing.is_empty() {
            let mut names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            names.sort_unstable();
            bail!("no such case: {}", names.join(", "));
        }
    }

    let mut machine = facts(&dri);
    machine["timer_floor"] = timer_floor(4000);
    let throttle_before = throttle_counter();
    let load_before = load_avg();
    let govs = machine["governors"].as_array().cloned().unwrap_or_default();
    let available = machine["governors_available"].to_string();
    if machine["governor_pinned"] == json!(false) && available.contains("\"performance\"") {
        let names: Vec<&str> = govs.iter().filter_map(Value::as_str).collect();
        eprintln!(
            "bench: cpu governor is {} and this machine offers performance; the spread \
             column says whether it mattered",
            names.join("/")
        );
    }
    if let Some(flags) = throttle_before
        .as_ref()
        .and_then(|t| t.get("flags"))
        .and_then(Value::as_array)
        .filter(|f| !f.is_empty())
    {
        let names: Vec<&str> = flags.iter().filter_map(Value::as_str).collect();
        eprintln!(
            "bench: this board reports {}. A board that has throttled once will do it \
             again under a benchmark; the run is discarded if the counter moves",
            names.join(", ")
        );
    }

    let started = SystemTime::now();
    let clock = Instant::now();
    let mut skipped: Vec<Value> = Vec::new();
    let mut live = open_cases(specs, components, timestep, &dri, &mut skipped);
    if live.is_empty() {
        bail!("no case opened");
    }

    let floor_s = number(&cfg, "warmup_min_s")?;
    let ceiling_s = number(&cfg, "warmup_max_s")?;
    let mut warm = Vec::new();
    for c in &mut live {
        warm.push((c.name.clone(), c.warm(floor_s, ceiling_s)?));
    }
    let mut cost: HashMap<String, Vec<f64>> =
        live.iter().map(|c| (c.name.clone(), Vec::new())).collect();
    for _ in 0..PROBE_ROUNDS {
        for c in &mut live {
            let slice = c.probe(PROBE_S)?;
            cost.get_mut(&c.name).unwrap().extend(slice);
        }
    }
    let rows = dimension(warm, &cost, reps, rep_ms, fixed_frames, like.as_ref());
    println!(
        "{}",
        format_plan(&rows, reps, rep_ms, fixed_frames, a.like.as_deref())
    );
    for r in &rows {
        if r.thin {
            eprintln!(
                "bench: {} pools only {} frames; its stable statistic is the run \
                 minimum, not a percentile",
                r.name, r.pooled
            );
        }
    }
    if a.plan_only {
        return Ok(());
    }

    let sized: HashMap<&str, i64> = rows.iter().map(|r| (r.name.as_str(), r.frames)).collect();
    for c in &mut live {
        c.size_to(sized[c.name.as_str()]);
    }

    // The run interleaves the cases, one repetition at a time.
    // A wandering GPU clock corrupts the case that runs at that moment.
    // Interleaving spreads that cost across every case.
    for rep in 0..reps {
        for c in &mut live {
            c.block()?;
        }
        println!("bench: rep {}/{reps}", rep + 1);
    }

    let mut results = Vec::new();
    for c in &mut live {
        let mut r = c.result();
        let row = rows.iter().find(|p| p.name == c.name).unwrap();
        r["warm_frames"] = json!(row.warm.frames);
        r["warm_s"] = json!(row.warm.seconds);
        r["warm_settled"] = json!(row.warm.settled);
        results.push(r);
        c.r.close();
    }
    let throttle_after = throttle_counter();
    let elapsed = clock.elapsed().as_secs_f64();

    let gating = throttle_before
        .as_ref()
        .is_some_and(|t| t["gating"] == json!(true));
    let moved = match (&throttle_before, &throttle_after) {
        (Some(b), Some(n)) => b["value"] != n["value"],
        _ => false,
    };
    if let (true, Some(b)) = (moved, throttle_before.as_ref()) {
        let source = b["source"].as_str().unwrap_or("");
        if gating {
            eprintln!(
                "bench: THROTTLED during the run ({source}: {} -> {}): these numbers \
                 are the heatsink, not the change",
                b["value"],
                throttle_after.as_ref().unwrap()["value"]
            );
        } else {
            eprintln!(
                "bench: {source} counters moved ({} -> {}). Chip wide on a shared \
                 host, so advisory only: judge the run by its spread column",
                b["value"],
                throttle_after.as_ref().unwrap()["value"]
            );
        }
    }

    for r in &results {
        let ms = |k: &str| r[k]["stable"].as_f64().unwrap_or(0.0);
        println!(
            "{:<24} draw {:7.4} read {:7.4} host {:7.4} ms  {:7.1} fps  median {:7.4} \
             spread {:5.1}% rep {:5.1}%",
            r["name"].as_str().unwrap_or(""),
            ms("draw_ms"),
            ms("read_ms"),
            ms("host_ms"),
            r["fps"].as_f64().unwrap_or(0.0),
            r["frame_ms"]["median"].as_f64().unwrap_or(0.0),
            r["frame_ms"]["spread_pct"].as_f64().unwrap_or(0.0),
            r["frame_ms"]["rep_spread_pct"].as_f64().unwrap_or(0.0),
        );
    }
    if !skipped.is_empty() {
        println!(
            "\nSKIPPED {} case(s), this run does not cover them:",
            skipped.len()
        );
        for s in &skipped {
            println!(
                "  {}: {}",
                s["name"].as_str().unwrap_or(""),
                s["error"].as_str().unwrap_or("")
            );
        }
    }

    let value_of = |t: &Option<Value>, k: &str| {
        t.as_ref()
            .map_or(Value::Null, |v| v.get(k).cloned().unwrap_or(Value::Null))
    };
    let doc = json!({
        "schema": SCHEMA,
        "started": utc(epoch_secs(started), true),
        "duration_s": (elapsed * 10.0).round() / 10.0,
        "note": a.note,
        "config": cfg,
        "plan": rows.iter().map(PlanRow::to_json).collect::<Vec<_>>(),
        "skipped": skipped,
        "machine": machine,
        "load": {"before": load_before, "after": load_avg()},
        "throttle": {
            "source": value_of(&throttle_before, "source"),
            "gating": gating,
            "cpus": value_of(&throttle_before, "cpus"),
            "before": value_of(&throttle_before, "value"),
            "after": value_of(&throttle_after, "value"),
            "throttled": if throttle_before.is_some() && throttle_after.is_some() {
                json!(moved)
            } else {
                Value::Null
            },
        },
        "results": results,
    });

    let text = serde_json::to_string_pretty(&doc)? + "\n";
    let out = match &a.out {
        Some(o) => o.clone(),
        None => {
            let host = doc["machine"]["host"].as_str().unwrap_or("nohost");
            let commit = doc["machine"]["git"]["commit"].as_str().unwrap_or("nogit");
            let commit: String = commit.chars().take(8).collect();
            let dir = a.results_dir.clone().unwrap_or_else(|| ".".to_string());
            fs::create_dir_all(&dir)?;
            format!(
                "{dir}/{host}-{commit}-{}.json",
                utc(epoch_secs(started), false)
            )
        }
    };
    if out == "-" {
        print!("{text}");
    } else {
        fs::write(&out, &text).with_context(|| format!("cannot write {out}"))?;
        println!("bench: wrote {out} in {elapsed:.0}s");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_case_names_a_shipped_scene_once_and_the_suite_holds_every_setting_key() {
        let _g = crate::filters::env_guard();
        let suite: Value = serde_json::from_str(SUITE).unwrap();
        let cases = suite_cases(&suite).unwrap();
        assert!(!cases.is_empty());
        let names: std::collections::HashSet<String> = cases.iter().map(case_name).collect();
        assert_eq!(names.len(), cases.len());
        let shipped = crate::filters::filter_defs();
        for c in &cases {
            let scene = format!(
                "{}/ghee/examples/{}.scene.json",
                env!("CARGO_MANIFEST_DIR"),
                c.scene
            );
            assert!(Path::new(&scene).is_file(), "{scene}");
            assert!(c.filters.iter().all(|f| shipped.contains_key(f)), "{scene}");
        }
        let dithered = CaseSpec {
            scene: "bars".to_string(),
            size: (320, 240),
            filters: vec!["dither".to_string()],
        };
        assert_eq!(case_name(&dithered), "bars@320x240+dither");
        for k in [
            "reps",
            "rep_ms",
            "warmup_min_s",
            "warmup_max_s",
            "components",
        ] {
            assert!(suite.get(k).is_some(), "suite: missing {k}");
        }
    }

    #[test]
    fn the_config_carries_the_eight_settings_in_one_fixed_order() {
        let suite: Value = serde_json::from_str(SUITE).unwrap();
        let a = BenchArgs {
            suite: None,
            out: None,
            results_dir: None,
            case: Vec::new(),
            frames: None,
            like: None,
            rep_ms: None,
            warmup_min_s: None,
            warmup_max_s: None,
            reps: None,
            components: None,
            timestep: None,
            dri: None,
            pin: None,
            note: String::new(),
            plan_only: false,
        };
        let cfg = config(&suite, &a).unwrap();
        assert_eq!(
            cfg.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "reps",
                "rep_ms",
                "warmup_min_s",
                "warmup_max_s",
                "components",
                "timestep",
                "frames",
                "pin"
            ]
        );
        assert_eq!(cfg["reps"], json!(10));
        assert_eq!(cfg["frames"], Value::Null);
        assert_eq!(cfg["pin"], Value::Null);
    }

    #[test]
    fn an_override_replaces_the_suite_setting() {
        let suite: Value = serde_json::from_str(SUITE).unwrap();
        let mut a = BenchArgs {
            suite: None,
            out: None,
            results_dir: None,
            case: Vec::new(),
            frames: Some(60),
            like: None,
            rep_ms: None,
            warmup_min_s: None,
            warmup_max_s: None,
            reps: Some(3),
            components: None,
            timestep: None,
            dri: None,
            pin: Some(2),
            note: String::new(),
            plan_only: false,
        };
        let cfg = config(&suite, &a).unwrap();
        assert_eq!(cfg["reps"], json!(3));
        assert_eq!(cfg["frames"], json!(60));
        assert_eq!(cfg["pin"], json!(2));
        a.reps = None;
        assert_eq!(config(&suite, &a).unwrap()["reps"], json!(10));
    }

    #[test]
    fn a_frame_count_snaps_to_the_nearest_rung_in_log_space() {
        assert_eq!(nearest_rung(0.0), 480);
        assert_eq!(nearest_rung(-3.0), 480);
        assert_eq!(nearest_rung(1.0), 60);
        assert_eq!(nearest_rung(59.0), 60);
        // The boundary is the geometric midpoint, 84.85.
        assert_eq!(nearest_rung(84.0), 60);
        assert_eq!(nearest_rung(86.0), 120);
        assert_eq!(nearest_rung(90.0), 120);
        assert_eq!(nearest_rung(10_000.0), 480);
    }

    #[test]
    fn a_count_outside_the_rungs_names_its_bound() {
        // A case too cheap to time falls to the top rung, and reports no bound.
        assert_eq!(wanted_frames(0.0, 250.0), (480.0, None));
        let (wanted, bound) = wanted_frames(0.5, 250.0);
        assert!((wanted - 500.0).abs() < 1e-9);
        assert_eq!(bound, Some("cap"));
        assert_eq!(wanted_frames(10.0, 250.0), (25.0, Some("floor")));
        assert_eq!(wanted_frames(2.5, 250.0), (100.0, None));
    }

    #[test]
    fn median_percentile_and_rel_sd_match_their_definitions() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(pct(&v, 0.01), 2.0);
        assert_eq!(pct(&v, 0.95), 96.0);
        assert_eq!(pct(&[], 0.01), 0.0);
        assert!((rel_sd(&[10.0, 12.0]) - 100.0 * (2.0f64 / (2.0f64).sqrt()) / 11.0).abs() < 1e-9);
        assert_eq!(rel_sd(&[5.0]), 0.0);
    }

    /// A run of more than two repetitions drops the first one.
    #[test]
    fn the_summary_drops_the_first_repetition() {
        let reps: Vec<Vec<f64>> =
            vec![vec![100.0; 10], vec![1.0; 10], vec![1.0; 10], vec![1.0; 10]];
        let s = summarise(&reps);
        assert_eq!(s["n"], json!(30));
        assert_eq!(s["stable"], json!(1.0));
        assert_eq!(s["rep_mins"].as_array().unwrap().len(), 3);
        let two = summarise(&[vec![2.0; 4], vec![2.0; 4]]);
        assert_eq!(two["n"], json!(8));
    }

    #[test]
    fn the_split_half_spread_is_the_gap_between_the_odd_and_even_repetitions() {
        let reps = vec![
            vec![1.0; 100],
            vec![2.0; 100],
            vec![1.0; 100],
            vec![2.0; 100],
        ];
        // Rep 0 drops, so the halves are [2.0] and [1.0, 2.0].
        let s = summarise(&reps);
        assert_eq!(s["spread_pct"], json!(100.0));
        let flat = summarise(&[vec![1.0; 50], vec![1.0; 50], vec![1.0; 50]]);
        assert_eq!(flat["spread_pct"], json!(0.0));
    }

    #[test]
    fn the_feed_table_is_one_row_per_frame() {
        let t = feed_table(4, 3);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 4);
        assert!((f64::from(t[0][0]) - 0.5).abs() < 1e-6);
        assert!(t.iter().flatten().all(|v| (0.0..=1.0).contains(v)));
        assert_ne!(t[0], t[1]);
    }

    #[test]
    fn a_case_name_carries_its_size_and_filters() {
        let c = CaseSpec {
            scene: "bars".into(),
            size: (128, 32),
            filters: vec!["dither".into(), "invert".into()],
        };
        assert_eq!(case_name(&c), "bars@128x32+dither+invert");
    }

    #[test]
    fn utc_prints_both_stamp_forms_including_a_leap_day() {
        assert_eq!(utc(0, true), "1970-01-01T00:00:00Z");
        assert_eq!(utc(0, false), "19700101-000000");
        assert_eq!(utc(1_700_000_000, true), "2023-11-14T22:13:20Z");
        assert_eq!(utc(1_700_000_000, false), "20231114-221320");
        // A leap day, and the last second before the next leap day.
        assert_eq!(utc(1_709_164_800, true), "2024-02-29T00:00:00Z");
        assert_eq!(utc(951_868_799, true), "2000-02-29T23:59:59Z");
    }

    #[test]
    fn a_key_value_line_is_read_off_its_separator() {
        assert_eq!(
            field("DRIVER=i915\nOF_NAME=gpu", "DRIVER", '='),
            Some("i915".to_string())
        );
        assert_eq!(
            field("model name\t: Ryzen", "model name", ':'),
            Some("Ryzen".into())
        );
        assert_eq!(field("DRIVERX=v3d", "DRIVER", '='), None);
        assert_eq!(field("", "DRIVER", '='), None);
    }

    #[test]
    fn the_plan_names_the_bound_beside_the_wanted_count() {
        let rows = vec![PlanRow {
            name: "bars@128x32".into(),
            ms_per_frame: 0.384,
            warm: Warm {
                frames: 700,
                seconds: 0.25,
                settled: true,
            },
            frames: 480,
            wanted: 651.0,
            bound: Some("cap"),
            rep_s: 0.18,
            total_s: 1.8,
            pooled: 4320,
            thin: false,
        }];
        let out = format_plan(&rows, 10, 250.0, None, None);
        assert!(out.starts_with("bench: frames per case for a 250 ms repetition, 10 reps\n"));
        assert!(out.contains("cap (a 250 ms rep wants 651)"), "{out}");
        assert!(
            out.contains("bench: 1 case, about 2s of measured frames"),
            "{out}"
        );
        assert!(format_plan(&rows, 10, 250.0, Some(60), None).contains("frames fixed at 60"));
        assert!(format_plan(&rows, 10, 250.0, None, Some("A.json"))
            .contains("frames taken from A.json"));
    }
}
