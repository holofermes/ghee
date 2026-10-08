//! The facts about the host that GET /hwinfo serves to the console's render
//! info.
//!
//! The device tree names the board, and a Raspberry Pi's board name gives its
//! GPU. /proc/meminfo gives the memory, and on a Raspberry Pi vcgencmd gives
//! the GPU's share of it and the SoC's temperature. Any other board gives n/a
//! for the GPU, and a host with no device tree gives its kernel's name and
//! release in place of the board.

use std::fs;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

/// vcgencmd talks to the VideoCore mailbox. A busy mailbox answers late, so the
/// call gets a deadline.
const DEADLINE: Duration = Duration::from_secs(1);
const POLL: Duration = Duration::from_millis(10);

/// The GPU each Raspberry Pi generation carries, by a word of its board name.
const GPUS: [(&str, &str); 5] = [
    ("pi 5", "VideoCore VII"),
    ("pi 4", "VideoCore VI"),
    ("pi 3", "VideoCore IV"),
    ("pi 2", "VideoCore IV"),
    ("zero 2", "VideoCore IV"),
];

/// The facts, as GET /hwinfo serves them: model, gpu, ram, ram_free,
/// gpu_mem, temp. A fact that this host cannot read is absent.
pub fn hwinfo() -> Value {
    let mut d = Map::new();
    let board = read_model();
    let gpu = gpu_name(board.as_deref());
    put(&mut d, "model", Some(board.unwrap_or_else(platform)));
    put(&mut d, "gpu", Some(gpu.to_string()));
    if let Ok(text) = fs::read_to_string("/proc/meminfo") {
        put(&mut d, "ram", meminfo(&text, "MemTotal").map(mb));
        put(&mut d, "ram_free", meminfo(&text, "MemAvailable").map(mb));
    }
    if gpu != NO_GPU {
        put(
            &mut d,
            "gpu_mem",
            vcgencmd(&["get_mem", "gpu"]).and_then(value_of),
        );
        put(
            &mut d,
            "temp",
            vcgencmd(&["measure_temp"]).and_then(value_of),
        );
    }
    Value::Object(d)
}

/// The GPU of a host whose board names none.
const NO_GPU: &str = "n/a";

fn put(d: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|v| !v.is_empty()) {
        d.insert(key.to_string(), Value::from(value));
    }
}

/// The board, as the device tree names it.
fn read_model() -> Option<String> {
    let raw = fs::read("/proc/device-tree/model").ok()?;
    let text = String::from_utf8_lossy(&raw);
    Some(text.trim_matches('\0').trim().to_string()).filter(|m| !m.is_empty())
}

/// The GPU of `board`: its generation's for a Raspberry Pi, VideoCore for
/// one this table misses, and n/a for any other board.
fn gpu_name(board: Option<&str>) -> &'static str {
    let board = board.unwrap_or_default().to_lowercase();
    if !board.contains("raspberry pi") {
        return NO_GPU;
    }
    GPUS.iter()
        .find(|(word, _)| board.contains(word))
        .map_or("VideoCore", |(_, gpu)| *gpu)
}

/// The kernel's own description of the machine.
fn platform() -> String {
    let os = fs::read_to_string("/proc/sys/kernel/ostype").unwrap_or_default();
    let release = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
    let parts: Vec<&str> = [os.trim(), release.trim()]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return "unknown".to_string();
    }
    parts.join("-")
}

/// The kB figure /proc/meminfo names.
fn meminfo(text: &str, key: &str) -> Option<u64> {
    for line in text.lines() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        if name == key {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn mb(kb: u64) -> String {
    format!("{} MB", kb / 1024)
}

/// vcgencmd answers `name=value`.
fn value_of(out: String) -> Option<String> {
    out.rsplit_once('=').map(|(_, value)| value.to_string())
}

/// vcgencmd's stdout, or None when the host has no vcgencmd and when the
/// deadline runs out.
fn vcgencmd(args: &[&str]) -> Option<String> {
    let child = Command::new("vcgencmd")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let child = wait_for(child)?;
    let out = child.wait_with_output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The exited child. A child still running at the deadline is killed, and this
/// answers None.
fn wait_for(mut child: Child) -> Option<Child> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Some(child),
            Ok(None) if start.elapsed() < DEADLINE => thread::sleep(POLL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Err(_) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The board names that each Pi's /proc/device-tree/model holds.
    #[test]
    fn each_board_names_its_gpu() {
        for (board, gpu) in [
            ("Raspberry Pi 5 Model B Rev 1.0", "VideoCore VII"),
            ("Raspberry Pi 4 Model B Rev 1.5", "VideoCore VI"),
            ("Raspberry Pi 3 Model B Plus Rev 1.3", "VideoCore IV"),
            ("Raspberry Pi Zero 2 W Rev 1.0", "VideoCore IV"),
            ("Raspberry Pi Zero W Rev 1.1", "VideoCore"),
            ("Pine64 RockPro64 v2.1", "n/a"),
        ] {
            assert_eq!(gpu_name(Some(board)), gpu, "{board}");
        }
        assert_eq!(gpu_name(None), "n/a");
    }

    /// Lines as a Pi 3's /proc/meminfo holds them.
    #[test]
    fn a_meminfo_line_reads_in_kb() {
        let text =
            "MemTotal:         418176 kB\nMemAvailable:      1024 kB\nHugepagesize: 2048 kB\n";
        assert_eq!(meminfo(text, "MemTotal"), Some(418_176));
        assert_eq!(mb(meminfo(text, "MemTotal").unwrap()), "408 MB");
        assert_eq!(meminfo(text, "MemFree"), None);
        assert_eq!(meminfo("no colon here\n", "MemTotal"), None);
    }

    #[test]
    fn vcgencmd_answers_after_the_equals_sign() {
        assert_eq!(
            value_of("temp=45.6'C".to_string()).as_deref(),
            Some("45.6'C")
        );
        assert_eq!(value_of("gpu=76M".to_string()).as_deref(), Some("76M"));
        assert_eq!(value_of("no equals sign".to_string()), None);
    }
}
