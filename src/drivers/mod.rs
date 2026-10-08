//! Display driver discovery.
//! available_drivers() reports the table below.

mod base;
mod gpio;
mod ssd1306;
mod st7789;

use anyhow::{anyhow, Result};
use serde_json::Value;

pub(crate) use crate::overlay::{i32_opt, int_opt, pin_opt};
pub use base::DisplayDriver;

/// --display-opts sets these options. Each k=v pair parses to one value.
pub type Opts = [(String, Value)];

/// A panel option and its default.
/// --display-opts sets the option, and `ghee displays` lists it.
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
    pub open: fn(&Opts) -> Result<Box<dyn DisplayDriver + Send>>,
    /// The width and height of the frame that the driver opened with these
    /// options takes, known before it opens.
    pub size: fn(&Opts) -> Result<(i32, i32)>,
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

/// What the userspace driver's constructor takes, in declaration order.
pub fn driver_options(spec: &DriverSpec) -> Vec<&'static DriverOption> {
    spec.options.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
