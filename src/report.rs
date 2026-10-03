//! What ghee says while it works.
//!
//! Three levels: `--quiet` leaves errors alone, the default names each step,
//! `--verbose` adds the paths and the formats behind them. Every word here
//! goes to stderr, because `-o -` pipes its frames to stdout, and
//! `ghee displays` writes its listing there.
//!
//! A terminal gets a progress line that rewrites itself. A pipe gets one line
//! per event and no escape codes, so a log stays a log.

use std::io::{stderr, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const QUIET: u8 = 0;
pub const NORMAL: u8 = 1;
pub const VERBOSE: u8 = 2;

static LEVEL: AtomicU8 = AtomicU8::new(NORMAL);
static TTY: AtomicBool = AtomicBool::new(false);
/// A progress line is on screen and owns the cursor until something clears it.
static OPEN: AtomicBool = AtomicBool::new(false);

/// Every line lines up on this label column, so a run reads as one block.
const LABEL: usize = 9;

/// Take the level from the command line, and hand it to the engine as well:
/// most of what a run says about itself comes from the C side.
pub fn apply(a: &crate::cli::Ns) {
    let level = match (a.quiet, a.verbose) {
        (true, _) => QUIET,
        (_, true) => VERBOSE,
        _ => NORMAL,
    };
    set_level(level);
    unsafe { crate::ffi::ghee_set_verbosity(level as i32) };
}

pub fn set_level(level: u8) {
    LEVEL.store(level, Ordering::Relaxed);
    TTY.store(stderr().is_terminal(), Ordering::Relaxed);
}

pub fn level() -> u8 {
    LEVEL.load(Ordering::Relaxed)
}

pub fn tty() -> bool {
    TTY.load(Ordering::Relaxed)
}

/// Clear a progress line before anything else writes, so the two never share
/// a row.
fn clear() {
    if OPEN.swap(false, Ordering::Relaxed) {
        let _ = write!(stderr(), "\r\x1b[K");
    }
}

fn say(label: &str, text: &str) {
    clear();
    let _ = writeln!(stderr(), "{label:>LABEL$}  {text}");
}

/// A step of the run: a found display, a loaded scene.
pub fn step(label: &str, text: impl AsRef<str>) {
    if level() >= NORMAL {
        say(label, text.as_ref());
    }
}

/// The detail behind a step: a path, a pixel format, a device.
pub fn detail(label: &str, text: impl AsRef<str>) {
    if level() >= VERBOSE {
        say(label, text.as_ref());
    }
}

/// Something is wrong, and the run carries on: a text that draws error
/// boxes. Every level says it, `--quiet` too.
pub fn error(text: impl AsRef<str>) {
    say("error", text.as_ref());
}

/// The run carried on through this, and the reader would want to know.
pub fn warn(text: impl AsRef<str>) {
    if level() >= NORMAL {
        say("careful", text.as_ref());
    }
}

/// The last line of a run.
pub fn summary(text: impl AsRef<str>) {
    if level() >= NORMAL {
        say("done", text.as_ref());
    }
}

/// A line that rewrites itself while work runs. A pipe gets nothing: the
/// finished step says the same thing once, in `step`.
pub fn progress(label: &str, text: impl AsRef<str>) {
    if level() < NORMAL || !tty() {
        return;
    }
    let mut err = stderr();
    let _ = write!(err, "\r\x1b[K{label:>LABEL$}  {}", text.as_ref());
    let _ = err.flush();
    OPEN.store(true, Ordering::Relaxed);
}

/// Take the progress line off the screen, leaving the row to whatever writes
/// next.
pub fn progress_done() {
    clear();
    let _ = stderr().flush();
}

/// A running render's own numbers, redrawn a few times a second.
struct Live {
    start: Instant,
    drawn: Instant,
    frames: u64,
    ms: f64,
}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);
const REDRAW: Duration = Duration::from_millis(200);

/// One frame went by, costing `ms`. `budget` 0 is a run with no end in sight.
/// `sinks` is read only when the line redraws, since a run can gain one.
/// Only a terminal sees this.
pub fn frame(ms: f64, drawn: u64, budget: u64, dropped: u64, sinks: impl FnOnce() -> String) {
    if level() < NORMAL || !tty() {
        return;
    }
    let now = Instant::now();
    let Ok(mut held) = LIVE.lock() else {
        return;
    };
    let live = held.get_or_insert(Live {
        start: now,
        drawn: now,
        frames: 0,
        ms: 0.0,
    });
    live.frames += 1;
    // an average that leans on the recent frames, so a stall shows and fades
    live.ms = match live.frames {
        1 => ms,
        _ => live.ms * 0.9 + ms * 0.1,
    };
    if now - live.drawn < REDRAW {
        return;
    }
    live.drawn = now;
    let secs = (now - live.start).as_secs_f64();
    let fps = match secs > 0.0 {
        true => live.frames as f64 / secs,
        false => 0.0,
    };
    let ms = live.ms * 1000.0;
    drop(held);
    progress("render", running(drawn, budget, fps, ms, dropped, &sinks()));
}

/// A budget makes it a fraction, and a run with no end reports its rate
/// alone. Dropped frames appear once there are any, so a paced run that keeps
/// up says nothing about it.
fn running(drawn: u64, budget: u64, fps: f64, ms: f64, dropped: u64, sinks: &str) -> String {
    let mut out = match budget {
        0 => String::new(),
        n => format!("{drawn}/{n}  "),
    };
    out.push_str(&format!("{fps:.1} fps  {ms:.1} ms"));
    if dropped > 0 {
        out.push_str(&format!("  {dropped} dropped"));
    }
    if !sinks.is_empty() {
        out.push_str(&format!("  {sinks}"));
    }
    out
}

/// End the live line and forget the run behind it.
pub fn frames_done() {
    if let Ok(mut held) = LIVE.lock() {
        *held = None;
    }
    progress_done();
}

/// The greeting on a bare `ghee` and on `--version`, and nowhere else: a
/// render prints frames, not a logo. Colour only where a terminal reads it.
pub fn splash(version: &str) -> String {
    let art = [
        r"  ____ _   _ _____ _____ ",
        r" / ___| | | | ____| ____|",
        r"| |  _| |_| |  _| |  _|  ",
        r"| |_| |  _  | |___| |___ ",
        r" \____|_| |_|_____|_____|",
    ];
    let (dim, off) = match stderr().is_terminal() {
        true => ("\x1b[2m", "\x1b[0m"),
        false => ("", ""),
    };
    let mut out = String::new();
    for line in art {
        out.push_str(&format!("{dim}{line}{off}\n"));
    }
    out.push_str(&format!("\n {version}\n"));
    out.push_str(" ghee --help\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_holds_its_tongue_and_verbose_does_not() {
        set_level(QUIET);
        assert_eq!(level(), QUIET);
        set_level(VERBOSE);
        assert_eq!(level(), VERBOSE);
        set_level(NORMAL);
        assert_eq!(level(), NORMAL);
    }

    #[test]
    fn a_budget_reads_as_a_fraction_and_a_run_without_one_does_not() {
        assert_eq!(
            running(240, 900, 33.06, 30.21, 0, "kms"),
            "240/900  33.1 fps  30.2 ms  kms"
        );
        assert_eq!(
            running(240, 0, 33.06, 30.21, 0, "kms"),
            "33.1 fps  30.2 ms  kms"
        );
    }

    /// Dropped frames stay off the line until a run drops one, and a sink
    /// gained mid-run shows up beside the rest.
    #[test]
    fn dropped_frames_appear_only_once_there_are_any() {
        assert!(!running(240, 900, 33.0, 30.0, 0, "kms").contains("dropped"));
        assert_eq!(
            running(240, 900, 33.0, 30.0, 7, "kms+h264"),
            "240/900  33.0 fps  30.0 ms  7 dropped  kms+h264"
        );
    }

    /// Nothing here writes to a pipe's progress row, whatever the level.
    #[test]
    fn a_pipe_takes_no_progress() {
        set_level(NORMAL);
        TTY.store(false, Ordering::Relaxed);
        progress("scenes", "asking hero");
        assert!(!OPEN.load(Ordering::Relaxed));
    }
}
