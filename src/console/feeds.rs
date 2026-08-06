//! The console fills in this host info for itself. crate::feeds::demo_feed does
//! the same for array data.

use serde_json::{Map, Value};

/// Host info for a host that injects no provider of its own.
pub fn default_hwinfo() -> Value {
    let mut d = Map::new();
    d.insert("model".to_string(), Value::from(platform()));
    d.insert("gpu".to_string(), Value::from("n/a"));
    Value::Object(d)
}

/// The kernel's own description of the machine.
fn platform() -> String {
    let os = std::fs::read_to_string("/proc/sys/kernel/ostype").unwrap_or_default();
    let release = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
    let parts: Vec<&str> = [os.trim(), release.trim()]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return "unknown".to_string();
    }
    parts.join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hwinfo_names_the_host_and_no_gpu() {
        let info = default_hwinfo();
        assert_eq!(info["gpu"], "n/a");
        assert!(info["model"].as_str().is_some_and(|m| !m.is_empty()));
    }
}
