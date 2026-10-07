//! What the kernel tells about the display behind a KMS card, read from
//! sysfs under the card's device: its backlight.
//!
//! Every reader takes the sysfs root, `/sys` on a running system, so a test
//! hands it a directory tree of its own.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

/// The sysfs root of a running system.
pub const SYS: &str = "/sys";

/// The name of the card node `card`, such as `card1`: a /dev/dri path, its
/// /dev/dri/by-path link resolved.
fn card_name(card: &str) -> String {
    let path = Path::new(card);
    let real = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    real.file_name()
        .map_or_else(|| card.to_string(), |n| n.to_string_lossy().into_owned())
}

/// The sysfs directory of the device behind the card node `card`.
fn card_device(sys: &Path, card: &str) -> PathBuf {
    sys.join("class/drm").join(card_name(card)).join("device")
}

/// A backlight device that the kernel registered for a display. A panel
/// driver such as ssd130x registers one under the panel's own device, which
/// sets the panel's contrast.
#[derive(Debug, PartialEq)]
pub struct Backlight {
    dir: PathBuf,
}

impl Backlight {
    /// The backlight device under the device of the card node `card`, the
    /// first by name when it holds several. None when it holds none.
    pub fn of_card(sys: &Path, card: &str) -> Option<Backlight> {
        let entries = fs::read_dir(card_device(sys, card).join("backlight")).ok()?;
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        dirs.sort();
        dirs.into_iter()
            .find(|d| d.join("brightness").exists())
            .map(|dir| Backlight { dir })
    }

    fn read(&self, file: &str) -> Result<u64> {
        let path = self.dir.join(file);
        let text = fs::read_to_string(&path).with_context(|| path.display().to_string())?;
        text.trim()
            .parse()
            .map_err(|_| anyhow!("{}: {:?} is not a number", path.display(), text.trim()))
    }

    /// The level, 0 to 1: `brightness` over `max_brightness`.
    pub fn level(&self) -> Result<f64> {
        let max = self.read("max_brightness")?;
        if max == 0 {
            bail!("{}: max_brightness is 0", self.dir.display());
        }
        Ok(self.read("brightness")? as f64 / max as f64)
    }

    /// Write `brightness`, `level` from 0 to 1 scaled onto `max_brightness`,
    /// and give back the level that the device then reads.
    pub fn set_level(&self, level: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&level) {
            bail!("brightness takes 0 to 1, got {level}");
        }
        let max = self.read("max_brightness")?;
        let path = self.dir.join("brightness");
        let value = (level * max as f64).round() as u64;
        fs::write(&path, format!("{value}\n")).with_context(|| path.display().to_string())?;
        self.level()
    }
}

/// The backlight level of the display on the card node `card`, 0 to 1. None
/// when no card drives a display, or its display has no backlight device.
pub fn brightness(sys: &Path, card: Option<&str>) -> Result<Option<f64>> {
    match card.and_then(|c| Backlight::of_card(sys, c)) {
        Some(b) => b.level().map(Some),
        None => Ok(None),
    }
}

/// Set the backlight of the display on the card node `card`, 0 to 1, and
/// give back the level that it then reads.
pub fn set_brightness(sys: &Path, card: Option<&str>, level: f64) -> Result<f64> {
    let Some(card) = card else {
        bail!("brightness needs a KMS display, and nothing drives one");
    };
    match Backlight::of_card(sys, card) {
        Some(b) => b.set_level(level),
        None => bail!("the display on {card} has no backlight device"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    /// A sysfs tree of one card, card7, whose device holds a backlight
    /// device named as ssd130x names one, after the panel's I2C client.
    fn sys_with_backlight(brightness: u64, max: u64) -> TempDir {
        let sys = TempDir::new("panel-sys");
        let bl = sys.join("class/drm/card7/device/backlight/7-003c");
        fs::create_dir_all(&bl).unwrap();
        fs::write(bl.join("brightness"), format!("{brightness}\n")).unwrap();
        fs::write(bl.join("max_brightness"), format!("{max}\n")).unwrap();
        sys
    }

    #[test]
    fn the_level_is_brightness_over_its_maximum() {
        let sys = sys_with_backlight(51, 255);
        let got = brightness(sys.as_ref(), Some("/dev/dri/card7")).unwrap();
        assert_eq!(got, Some(0.2));
    }

    #[test]
    fn a_level_writes_brightness_scaled_onto_its_maximum() {
        for (level, written) in [(0.0, "0"), (0.5, "128"), (1.0, "255"), (0.2, "51")] {
            let sys = sys_with_backlight(127, 255);
            let back = set_brightness(sys.as_ref(), Some("/dev/dri/card7"), level).unwrap();
            let file = sys.join("class/drm/card7/device/backlight/7-003c/brightness");
            assert_eq!(fs::read_to_string(file).unwrap().trim(), written, "{level}");
            assert_eq!(back, written.parse::<f64>().unwrap() / 255.0, "{level}");
        }
    }

    #[test]
    fn a_display_with_no_backlight_device_reads_none_and_refuses_a_set() {
        let sys = TempDir::new("panel-sys");
        fs::create_dir_all(sys.join("class/drm/card1/device")).unwrap();
        let card = Some("/dev/dri/card1");
        assert_eq!(brightness(sys.as_ref(), card).unwrap(), None);
        assert_eq!(
            set_brightness(sys.as_ref(), card, 0.5)
                .unwrap_err()
                .to_string(),
            "the display on /dev/dri/card1 has no backlight device"
        );
        assert_eq!(brightness(sys.as_ref(), None).unwrap(), None);
        assert_eq!(
            set_brightness(sys.as_ref(), None, 0.5)
                .unwrap_err()
                .to_string(),
            "brightness needs a KMS display, and nothing drives one"
        );
    }

    #[test]
    fn a_level_outside_0_to_1_is_refused_and_writes_nothing() {
        let sys = sys_with_backlight(127, 255);
        let e = set_brightness(sys.as_ref(), Some("/dev/dri/card7"), 1.5).unwrap_err();
        assert_eq!(e.to_string(), "brightness takes 0 to 1, got 1.5");
        let file = sys.join("class/drm/card7/device/backlight/7-003c/brightness");
        assert_eq!(fs::read_to_string(file).unwrap().trim(), "127");
    }
}
