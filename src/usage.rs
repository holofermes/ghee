//! What this process uses: its resident set, the buffers and the GPU time of
//! its DRM clients, and beside them the machine's free memory and, on a
//! Raspberry Pi, the SoC's temperature. Every number comes from a file that
//! the process itself may read, with no root, at the moment it asks.
//!
//! The kernel reports a DRM client in the fdinfo of its descriptor
//! (Documentation/gpu/drm-usage-stats.rst): its buffers as
//! `drm-total-<region>` keys, and the time its jobs ran on each queue of the
//! GPU as `drm-engine-<queue>`. On a Pi 4 and a Pi 5 (v3d) the buffers are
//! system RAM that the resident set leaves out. A driver that writes no such
//! key leaves that number unknown: vc4, on a Pi 1 to a Pi 3 and a Zero 2 W,
//! writes none, and NVIDIA's writes none.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::time::Instant;

/// A size as fdinfo and /proc/self/status write it: a count, then no unit for
/// bytes, or `KiB`, `kB`, `MiB` or `GiB`.
fn size(v: &str) -> Option<u64> {
    let mut words = v.split_whitespace();
    let n: u64 = words.next()?.parse().ok()?;
    let unit = match words.next() {
        None => 1,
        Some("KiB" | "kB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some(_) => return None,
    };
    n.checked_mul(unit)
}

/// One DRM client, as its fdinfo reports it.
#[derive(Debug, Default, PartialEq)]
pub struct Client<'a> {
    /// `drm-client-id`, which every descriptor of one open file shares.
    pub id: Option<&'a str>,
    /// The `drm-total-<region>` keys summed, in bytes.
    pub memory: Option<u64>,
    /// `drm-engine-<queue>`: the nanoseconds that the client's jobs ran on
    /// each queue since it opened.
    pub engines: Vec<(&'a str, u64)>,
}

pub fn client(fdinfo: &str) -> Client<'_> {
    let mut c = Client::default();
    for line in fdinfo.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if key == "drm-client-id" {
            c.id = Some(value);
        } else if key.starts_with("drm-total-") {
            if let Some(bytes) = size(value) {
                c.memory = Some(c.memory.unwrap_or(0) + bytes);
            }
        } else if let Some(queue) = key.strip_prefix("drm-engine-") {
            let ns = value.strip_suffix("ns").map(str::trim);
            if let Some(ns) = ns.and_then(|n| n.parse().ok()) {
                c.engines.push((queue, ns));
            }
        }
    }
    c
}

/// The fdinfo of each DRM client of this process, once however many
/// descriptors it has.
fn clients() -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let Ok(fds) = fs::read_dir("/proc/self/fd") else {
        return out;
    };
    for entry in fds.flatten() {
        let Ok(link) = fs::read_link(entry.path()) else {
            continue;
        };
        if !link.starts_with("/dev/dri/") {
            continue;
        }
        let name = entry.file_name();
        let path = format!("/proc/self/fdinfo/{}", name.to_string_lossy());
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let fresh = match client(&text).id {
            Some(id) => seen.insert(id.to_string()),
            None => true,
        };
        if fresh {
            out.push(text);
        }
    }
    out
}

/// The bytes of buffers that this process's DRM clients hold.
pub fn gpu() -> Option<u64> {
    clients()
        .iter()
        .filter_map(|text| client(text).memory)
        .reduce(|a, b| a + b)
}

/// The nanoseconds that this process's jobs ran on each queue of its GPU
/// since its DRM clients opened, read at one moment.
pub struct Engines {
    at: Instant,
    ns: BTreeMap<String, u64>,
}

impl Engines {
    /// The reading now. None where the driver reports no engine time.
    pub fn read() -> Option<Engines> {
        let mut ns = BTreeMap::new();
        for text in clients() {
            for (queue, n) in client(&text).engines {
                *ns.entry(queue.to_string()).or_insert(0) += n;
            }
        }
        let at = Instant::now();
        (!ns.is_empty()).then_some(Engines { at, ns })
    }

    /// The reading now of a process whose DRM clients open later, with no
    /// time on any queue.
    pub fn none_yet() -> Engines {
        Engines {
            at: Instant::now(),
            ns: BTreeMap::new(),
        }
    }

    /// The queues that ran by this reading.
    pub fn ran(&self) -> impl Iterator<Item = &str> {
        self.ns
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(q, _)| q.as_str())
    }

    /// The percent of the time from `before` to this reading that each
    /// queue that ran by now was busy.
    pub fn busy_since(&self, before: &Engines) -> BTreeMap<String, f64> {
        let elapsed = self.at.saturating_duration_since(before.at).as_nanos();
        busy(
            &before.ns,
            &self.ns,
            u64::try_from(elapsed).unwrap_or(u64::MAX),
        )
    }
}

/// The percent of `elapsed_ns` that each queue that ran by `now` was busy,
/// from two readings of its nanoseconds. A queue new in `now` started at 0.
fn busy(
    before: &BTreeMap<String, u64>,
    now: &BTreeMap<String, u64>,
    elapsed_ns: u64,
) -> BTreeMap<String, f64> {
    now.iter()
        .filter(|(_, ns)| **ns > 0)
        .map(|(queue, ns)| {
            let ran = ns.saturating_sub(before.get(queue).copied().unwrap_or(0));
            (queue.clone(), 100.0 * ran as f64 / elapsed_ns.max(1) as f64)
        })
        .collect()
}

/// The SoC's temperature in °C, from the kernel's `cpu-thermal` zone, a
/// Raspberry Pi's one sensor for its whole SoC. None where zone 0 is another.
pub fn soc_celsius() -> Option<f64> {
    let zone = "/sys/class/thermal/thermal_zone0";
    if fs::read_to_string(format!("{zone}/type")).ok()?.trim() != "cpu-thermal" {
        return None;
    }
    let milli: i64 = fs::read_to_string(format!("{zone}/temp"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(milli as f64 / 1000.0)
}

/// The machine's free memory, the kernel's MemAvailable, in bytes.
pub fn machine_free() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(size)
}

/// The resident set of this process, in bytes.
pub fn rss() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(size)
}

/// Bytes in MiB.
pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fdinfo of a ghee process's v3d descriptor on a Pi 5, kernel
    /// 6.18.39+rpt-rpi-2712, cut to two of its six engines.
    const V3D: &str = "pos:\t0\nflags:\t02400002\nmnt_id:\t22\nino:\t520\n\
                       drm-driver:\tv3d\ndrm-client-id:\t1038\n\
                       drm-engine-bin: \t2140185 ns\nv3d-jobs-bin: \t2 jobs\n\
                       drm-engine-render: \t81414852 ns\nv3d-jobs-render: \t1 jobs\n\
                       drm-total-memory:\t72832 KiB\ndrm-shared-memory:\t0\n\
                       drm-active-memory:\t36928 KiB\ndrm-resident-memory:\t72832 KiB\n";

    /// Two regions and a size in bytes, as drm-usage-stats.rst lets a driver
    /// write them.
    const TWO_REGIONS: &str = "drm-driver:\ti915\ndrm-client-id:\t7\n\
                               drm-total-system0:\t2 MiB\ndrm-resident-system0:\t2 MiB\n\
                               drm-total-stolen-system0:\t512\n";

    #[test]
    fn a_client_holds_the_sum_of_its_total_keys_and_the_time_of_each_engine() {
        assert_eq!(
            client(V3D),
            Client {
                id: Some("1038"),
                memory: Some(72832 * 1024),
                engines: vec![("bin", 2140185), ("render", 81414852)],
            }
        );
        assert_eq!(client(TWO_REGIONS).memory, Some(2 * 1024 * 1024 + 512));
    }

    #[test]
    fn a_driver_that_reports_nothing_leaves_every_number_unknown() {
        assert_eq!(
            client("pos:\t0\nflags:\t02\nmnt_id:\t22\nino:\t9\n"),
            Client::default()
        );
    }

    #[test]
    fn sizes_read_in_every_unit_that_the_kernel_writes() {
        for (text, bytes) in [
            ("512", Some(512)),
            ("4 KiB", Some(4096)),
            ("4 kB", Some(4096)),
            ("3 MiB", Some(3 << 20)),
            ("1 GiB", Some(1 << 30)),
            ("4 KB", None),
            ("", None),
        ] {
            assert_eq!(size(text), bytes, "{text:?}");
        }
    }

    #[test]
    fn a_gpu_that_opens_after_the_first_reading_is_busy_from_that_reading() {
        let start = Engines::none_yet();
        let now = Engines {
            at: start.at + std::time::Duration::from_millis(1),
            ns: BTreeMap::from([("render".to_string(), 250_000)]),
        };
        assert_eq!(
            now.busy_since(&start),
            BTreeMap::from([("render".to_string(), 25.0)])
        );
    }

    #[test]
    fn a_queue_is_busy_for_the_share_of_the_time_that_its_jobs_ran() {
        let before = BTreeMap::from([("render".to_string(), 1_000), ("bin".to_string(), 500)]);
        let now = BTreeMap::from([
            ("render".to_string(), 251_000),
            ("bin".to_string(), 500),
            ("tfu".to_string(), 10_000),
            ("csd".to_string(), 0),
        ]);
        assert_eq!(
            busy(&before, &now, 1_000_000),
            BTreeMap::from([
                ("bin".to_string(), 0.0),
                ("render".to_string(), 25.0),
                ("tfu".to_string(), 1.0),
            ])
        );
    }
}
