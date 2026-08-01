//! Port of ghee/console/feeds.py.
//!
//! What the console fills in for itself: host info. The array placeholder is
//! feeds::demo_feed.

use serde_json::{Map, Value};

/// Generic host info when no real provider is injected.
pub fn default_hwinfo() -> Value {
    let mut d = Map::new();
    d.insert("model".to_string(), Value::from(platform()));
    d.insert("gpu".to_string(), Value::from("n/a"));
    Value::Object(d)
}

/// platform.platform(): the kernel's own description of the machine.
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
