//! Display driver discovery.
//! available_drivers() reports the table below.

mod base;
mod gpio;
mod ssd1306;
mod st7789;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

pub use base::DisplayDriver;

/// The options --display-opts sets. Each k=v pair parses to one value.
pub type Opts = [(String, Value)];

/// A command byte and its data, empty for a command that takes none.
/// A delay is command 0x00 carrying the milliseconds, the encoding
/// panel-mipi-dbi reads.
pub type Command = (u8, Vec<u8>);

/// The kernel's own setup for a panel, what `ghee overlay` writes out.
/// A panel whose driver holds its init sequence has no firmware of its own.
#[derive(Debug)]
pub struct Setup {
    pub firmware: Option<Vec<u8>>,
    /// The config.txt lines that bring the card up.
    pub lines: Vec<String>,
}

/// The firmware file panel-mipi-dbi loads: a 15-byte magic, the format
/// version, then each command as its byte, its parameter count and the
/// parameters.
pub fn firmware(init: &[Command]) -> Vec<u8> {
    let mut out = b"MIPI DBI\0\0\0\0\0\0\0\x01".to_vec();
    for (command, data) in init {
        out.push(*command);
        out.push(data.len() as u8);
        out.extend_from_slice(data);
    }
    out
}

/// A delay in the init sequence, in milliseconds. A panel-mipi-dbi delay
/// carries one byte, so 255 is the longest one.
pub(crate) fn delay(ms: u8) -> Command {
    (0x00, vec![ms])
}

/// A constructor option and its default.
/// --display-opts sets the option, and `ghee displays` lists it.
pub struct DriverOption {
    pub name: &'static str,
    pub default: Option<i64>,
    /// The option describes the panel, so `ghee overlay` takes it too.
    pub kernel: bool,
}

impl DriverOption {
    pub const fn new(name: &'static str, default: Option<i64>) -> DriverOption {
        DriverOption {
            name,
            default,
            kernel: true,
        }
    }

    /// An option the userspace driver alone acts on, like the format it puts
    /// on the wire.
    pub const fn own(name: &'static str, default: Option<i64>) -> DriverOption {
        DriverOption {
            name,
            default,
            kernel: false,
        }
    }
}

pub struct DriverSpec {
    pub name: &'static str,
    pub options: &'static [DriverOption],
    pub open: fn(&Opts) -> Result<Box<dyn DisplayDriver>>,
    /// The kernel setup for this panel, for `ghee overlay`.
    pub kernel: fn(&Opts) -> Result<Setup>,
}

const BUNDLED: [&DriverSpec; 2] = [&st7789::SPEC, &ssd1306::SPEC];

/// Every driver in this build, in table order.
pub fn available_drivers() -> Vec<&'static str> {
    BUNDLED.iter().map(|d| d.name).collect()
}

/// The driver registered under `name`.
pub fn get_driver(name: &str) -> Result<&'static DriverSpec> {
    BUNDLED
        .iter()
        .copied()
        .find(|d| d.name == name)
        .ok_or_else(|| anyhow!("no display driver '{name}'"))
}

/// What a driver's constructor takes, in declaration order.
pub fn driver_options(spec: &DriverSpec) -> &'static [DriverOption] {
    spec.options
}

/// The value set for `name`. The last pair wins.
fn opt<'a>(opts: &'a Opts, name: &str) -> Option<&'a Value> {
    opts.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v)
}

/// True when --display-opts set this option. A driver default describes the
/// panel it was written for, which is the wrong answer for a kernel setup.
pub(crate) fn given(opts: &Opts, name: &str) -> bool {
    opt(opts, name).is_some()
}

pub(crate) fn int_opt(opts: &Opts, name: &str, default: i64) -> Result<i64> {
    match opt(opts, name) {
        None => Ok(default),
        Some(v) => v
            .as_i64()
            .ok_or_else(|| anyhow!("{name} takes an integer, got {v}")),
    }
}

pub(crate) fn i32_opt(opts: &Opts, name: &str, default: i64) -> Result<i32> {
    let v = int_opt(opts, name, default)?;
    i32::try_from(v).map_err(|_| anyhow!("{name} {v} is out of range"))
}

/// A BCM pin number. None means the panel does without the line.
pub(crate) fn pin_opt(opts: &Opts, name: &str, default: Option<i64>) -> Result<Option<u8>> {
    let v = match opt(opts, name) {
        None => default,
        Some(Value::Null) => None,
        Some(v) => Some(
            v.as_i64()
                .ok_or_else(|| anyhow!("{name} takes a GPIO number, got {v}"))?,
        ),
    };
    match v {
        None => Ok(None),
        Some(v) => match u8::try_from(v) {
            Ok(pin) => Ok(Some(pin)),
            Err(_) => bail!("{name} {v} is not a GPIO number"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn the_bundled_drivers_are_available() {
        assert_eq!(available_drivers(), vec!["st7789", "ssd1306"]);
    }

    #[test]
    fn an_unknown_driver_names_itself() {
        let e = get_driver("does-not-exist").err().map(|e| e.to_string());
        assert_eq!(e.as_deref(), Some("no display driver 'does-not-exist'"));
        assert_eq!(get_driver("st7789").unwrap().name, "st7789");
    }

    #[test]
    fn driver_options_are_the_constructor_defaults_in_order() {
        let names: Vec<&str> = driver_options(get_driver("st7789").unwrap())
            .iter()
            .map(|o| o.name)
            .collect();
        assert_eq!(
            names,
            [
                "width",
                "height",
                "x_offset",
                "y_offset",
                "dc",
                "rst",
                "backlight",
                "baudrate",
                "spi_bus",
                "spi_device",
                "madctl",
                "color_bits",
                "partial"
            ]
        );
        let ssd: Vec<&str> = driver_options(get_driver("ssd1306").unwrap())
            .iter()
            .map(|o| o.name)
            .collect();
        assert_eq!(ssd, ["bus", "addr"]);
    }

    #[test]
    fn an_unset_option_takes_the_default() {
        let set = opts(&[("width", json!(40))]);
        assert_eq!(int_opt(&set, "width", 135).unwrap(), 40);
        assert_eq!(int_opt(&set, "height", 240).unwrap(), 240);
        assert_eq!(pin_opt(&set, "backlight", Some(22)).unwrap(), Some(22));
    }

    #[test]
    fn a_none_pin_stays_none() {
        let set = opts(&[("backlight", Value::Null)]);
        assert_eq!(pin_opt(&set, "backlight", Some(22)).unwrap(), None);
        assert_eq!(pin_opt(&[], "rst", None).unwrap(), None);
    }

    #[test]
    fn a_value_that_is_not_a_number_names_the_option() {
        let set = opts(&[("width", json!("wide")), ("dc", json!("high"))]);
        assert_eq!(
            int_opt(&set, "width", 135).unwrap_err().to_string(),
            "width takes an integer, got \"wide\""
        );
        assert_eq!(
            pin_opt(&set, "dc", Some(25)).unwrap_err().to_string(),
            "dc takes a GPIO number, got \"high\""
        );
        assert_eq!(
            pin_opt(&opts(&[("dc", json!(300))]), "dc", Some(25))
                .unwrap_err()
                .to_string(),
            "dc 300 is not a GPIO number"
        );
    }
}
