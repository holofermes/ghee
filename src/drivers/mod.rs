//! Port of ghee/drivers/__init__.py.
//!
//! Display driver discovery. The bundled drivers are compiled in, so there is no
//! import to fail and no entry-point plugin to load: available_drivers() reports
//! the table below.

mod base;
mod gpio;
mod ssd1306;
mod st7789;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

pub use base::DisplayDriver;

/// What --display-opts sets, as the values a k=v pair parses to.
pub type Opts = [(String, Value)];

/// A constructor option and its default: what --display-opts may set and
/// `ghee displays` lists.
pub struct DriverOption {
    pub name: &'static str,
    pub default: Option<i64>,
}

impl DriverOption {
    pub const fn new(name: &'static str, default: Option<i64>) -> DriverOption {
        DriverOption { name, default }
    }
}

pub struct DriverSpec {
    pub name: &'static str,
    pub options: &'static [DriverOption],
    pub open: fn(&Opts) -> Result<Box<dyn DisplayDriver>>,
}

const BUNDLED: [&DriverSpec; 2] = [&st7789::SPEC, &ssd1306::SPEC];

/// name -> "ok" or the reason it is unavailable.
pub fn available_drivers() -> Vec<(&'static str, &'static str)> {
    BUNDLED.iter().map(|d| (d.name, "ok")).collect()
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

/// The value set for `name`, last one wins.
fn opt<'a>(opts: &'a Opts, name: &str) -> Option<&'a Value> {
    opts.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v)
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

/// A BCM pin number, or None for a line this panel does without.
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
        assert_eq!(
            available_drivers(),
            vec![("st7789", "ok"), ("ssd1306", "ok")]
        );
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
                "madctl"
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
