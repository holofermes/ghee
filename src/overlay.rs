//! The kernel's own setup for a panel, what `ghee overlay` writes out: the
//! config.txt lines that hand the panel to a kernel display driver, and the
//! firmware file that the driver loads. The panel then answers `--display
//! kms` like any other display.
//!
//! Pins are BCM numbers.

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

/// --display-opts sets these options. Each k=v pair parses to one value.
pub type Opts = [(String, Value)];

/// A command byte and its data, empty for a command that takes none.
/// A delay is command 0x00 carrying the milliseconds, panel-mipi-dbi's own
/// encoding.
pub type Command = (u8, Vec<u8>);

/// A panel whose driver holds its init sequence has no firmware of its own.
#[derive(Debug)]
pub struct Setup {
    pub firmware: Option<Vec<u8>>,
    /// The config.txt lines that bring the card up.
    pub lines: Vec<String>,
}

/// A panel that `ghee overlay` sets up.
pub struct Panel {
    pub name: &'static str,
    /// What --display-opts sets for this panel, in declaration order.
    pub options: &'static [&'static str],
    pub setup: fn(&Opts) -> Result<Setup>,
}

const PANELS: [&Panel; 2] = [&ST7789, &SSD1306];

/// The panel registered under `name`.
pub fn panel(name: &str) -> Option<&'static Panel> {
    PANELS.iter().copied().find(|p| p.name == name)
}

/// Every panel, in table order.
pub fn panels() -> Vec<&'static str> {
    PANELS.iter().map(|p| p.name).collect()
}

/// panel-mipi-dbi loads this firmware file: a 15-byte magic, the format
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
fn delay(ms: u8) -> Command {
    (0x00, vec![ms])
}

/// The value set for `name`. The last pair wins.
fn opt<'a>(opts: &'a Opts, name: &str) -> Option<&'a Value> {
    opts.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v)
}

/// True when --display-opts set this option.
fn given(opts: &Opts, name: &str) -> bool {
    opt(opts, name).is_some()
}

fn int_opt(opts: &Opts, name: &str, default: i64) -> Result<i64> {
    match opt(opts, name) {
        None => Ok(default),
        Some(v) => v
            .as_i64()
            .ok_or_else(|| anyhow!("{name} takes an integer, got {v}")),
    }
}

fn i32_opt(opts: &Opts, name: &str, default: i64) -> Result<i32> {
    let v = int_opt(opts, name, default)?;
    i32::try_from(v).map_err(|_| anyhow!("{name} {v} is out of range"))
}

/// A BCM pin number. None means the panel does without the line.
fn pin_opt(opts: &Opts, name: &str, default: Option<i64>) -> Result<Option<u8>> {
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

// ST7789 command bytes
const SWRESET: u8 = 0x01;
const SLPOUT: u8 = 0x11;
const NORON: u8 = 0x13;
const INVON: u8 = 0x21;
const DISPON: u8 = 0x29;
const MADCTL: u8 = 0x36;
const COLMOD: u8 = 0x3A;

const ST7789_DC: i64 = 25;
const ST7789_BACKLIGHT: i64 = 22;
const ST7789_SPI_BUS: i64 = 0;
const ST7789_SPI_DEVICE: i64 = 0;
// MADCTL sets row order, column order and RGB/BGR: the panel rotation and mirroring.
// The value is per board. A wrong value flips the image.
const ST7789_MADCTL: i64 = 0xC0;
/// The SPI clock that the overlay carries. panel-mipi-dbi renders noise at
/// 99 MHz on a VC4-class board, and this clock is clean there. A panel that
/// takes more says so with `--display-opts baudrate=`.
const ST7789_BAUDRATE: i64 = 62_500_000;

/// panel-mipi-dbi replays this init sequence from the firmware file. The
/// kernel driver sets the address window itself on every flush, so CASET and
/// RASET stay out, and the offsets ride in the overlay instead. The delays
/// are the controller's own settling times.
fn st7789_init(madctl: u8) -> Vec<Command> {
    vec![
        (SWRESET, vec![]),
        delay(150),
        (SLPOUT, vec![]),
        delay(120),
        (COLMOD, vec![0x55]),
        (MADCTL, vec![madctl]),
        (INVON, vec![]),
        (NORON, vec![]),
        (DISPON, vec![]),
        delay(100),
    ]
}

/// The config.txt lines that hand an ST7789 to panel-mipi-dbi, and the
/// firmware named in those lines.
fn st7789(opts: &Opts) -> Result<Setup> {
    if !given(opts, "width") || !given(opts, "height") {
        bail!("pass width and height");
    }
    let madctl = int_opt(opts, "madctl", ST7789_MADCTL)?;
    let Ok(madctl) = u8::try_from(madctl) else {
        bail!("madctl {madctl} is not a byte");
    };
    let Some(dc) = pin_opt(opts, "dc", Some(ST7789_DC))? else {
        bail!("dc is the panel's command pin and takes a GPIO number");
    };
    let (bus, device) = (
        int_opt(opts, "spi_bus", ST7789_SPI_BUS)?,
        int_opt(opts, "spi_device", ST7789_SPI_DEVICE)?,
    );
    let mut lines = vec![
        format!(
            "dtoverlay=mipi-dbi-spi,spi{bus}-{device},speed={}",
            int_opt(opts, "baudrate", ST7789_BAUDRATE)?
        ),
        format!(
            "dtparam=width={},height={}",
            i32_opt(opts, "width", 0)?,
            i32_opt(opts, "height", 0)?
        ),
    ];
    // The offsets belong to the panel that carries the geometry above, so a
    // panel named here starts at the controller's origin until it says
    // otherwise.
    let (x, y) = (i32_opt(opts, "x_offset", 0)?, i32_opt(opts, "y_offset", 0)?);
    if x != 0 {
        lines.push(format!("dtparam=x-offset={x}"));
    }
    if y != 0 {
        lines.push(format!("dtparam=y-offset={y}"));
    }
    let mut pins = format!("dtparam=dc-gpio={dc}");
    if let Some(bl) = pin_opt(opts, "backlight", Some(ST7789_BACKLIGHT))? {
        pins.push_str(&format!(",backlight-gpio={bl}"));
    }
    if let Some(rst) = pin_opt(opts, "rst", None)? {
        pins.push_str(&format!(",reset-gpio={rst}"));
    }
    lines.push(pins);
    // MISO reaches no hobby breakout, and the driver reads the controller
    // without this.
    lines.push("dtparam=write-only".to_string());
    Ok(Setup {
        firmware: Some(firmware(&st7789_init(madctl))),
        lines,
    })
}

const ST7789: Panel = Panel {
    name: "st7789",
    options: &[
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
    ],
    setup: st7789,
};

const SSD1306_ADDRESS: i64 = 0x3C;
/// The I2C bus rate in Hz. A 128x32 frame takes 4.7 ms at 1 MHz. Above 1 MHz
/// a Pi 5's controller falls back to a far slower rate.
const SSD1306_SPEED: i64 = 1_000_000;

/// The config.txt lines that hand an SSD1306 to ssd130x on I2C bus 1. That
/// driver holds the init sequence for the whole Solomon family, so no
/// firmware file goes with it. The overlay's `address` sets the panel's I2C
/// address. A 128x32 module wires its rows `inverted,sequential`, and a
/// 128x64 module takes the overlay's own wiring. The bus rate line sets
/// every device on the bus.
fn ssd1306(opts: &Opts) -> Result<Setup> {
    let need = "ssd1306 needs height=32 or height=64";
    if !given(opts, "height") {
        bail!("{need}");
    }
    let height = int_opt(opts, "height", 0)?;
    let wiring = match height {
        32 => ",inverted,sequential",
        64 => "",
        h => bail!("{need}, got height={h}"),
    };
    let address = int_opt(opts, "address", SSD1306_ADDRESS)?;
    if !(0..0x80).contains(&address) {
        bail!("address {address} is not a 7-bit I2C address");
    }
    let speed = int_opt(opts, "speed", SSD1306_SPEED)?;
    if speed <= 0 {
        bail!("speed {speed} is not a bus rate in Hz");
    }
    // the overlay's node sits at the default address
    let address = match address {
        SSD1306_ADDRESS => String::new(),
        a => format!(",address=0x{a:02x}"),
    };
    Ok(Setup {
        firmware: None,
        lines: vec![
            "dtparam=i2c_arm=on".to_string(),
            format!("dtparam=i2c_arm_baudrate={speed}"),
            format!("dtoverlay=ssd1306,height={height}{address}{wiring}"),
        ],
    })
}

const SSD1306: Panel = Panel {
    name: "ssd1306",
    options: &["height", "address", "speed"],
    setup: ssd1306,
};

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

    fn object(set: &Value) -> Vec<(String, Value)> {
        set.as_object().unwrap().clone().into_iter().collect()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
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

    /// panel-mipi-dbi loads these bytes for the 240x240 bonnet, read back from
    /// the firmware file that drove the real panel.
    const FIRMWARE_ON_THE_PANEL: &str = "4d495049204442490000000000000001\
                                         010000019611000001783a0155360100\
                                         210013002900000164";

    fn bonnet() -> Vec<(String, Value)> {
        opts(&[
            ("width", json!(240)),
            ("height", json!(240)),
            ("x_offset", json!(0)),
            ("y_offset", json!(0)),
            ("madctl", json!(0)),
            ("backlight", json!(26)),
        ])
    }

    #[test]
    fn the_firmware_matches_the_file_that_drove_the_panel() {
        let blob = st7789(&bonnet()).unwrap().firmware.unwrap();
        assert_eq!(hex(&blob), FIRMWARE_ON_THE_PANEL);
        assert_eq!(blob.len(), 41);
    }

    #[test]
    fn the_overlay_lines_carry_the_panel_described_by_the_options() {
        assert_eq!(
            st7789(&bonnet()).unwrap().lines,
            [
                "dtoverlay=mipi-dbi-spi,spi0-0,speed=62500000",
                "dtparam=width=240,height=240",
                "dtparam=dc-gpio=25,backlight-gpio=26",
                "dtparam=write-only",
            ]
        );
    }

    /// The offsets ride in the overlay, since the kernel driver sets the
    /// address window itself.
    #[test]
    fn an_offset_panel_puts_its_offsets_in_the_overlay() {
        let mut set = bonnet();
        set.retain(|(k, _)| k != "y_offset" && k != "madctl");
        set.push(("y_offset".to_string(), json!(80)));
        let lines = st7789(&set).unwrap().lines;
        assert!(
            lines.contains(&"dtparam=y-offset=80".to_string()),
            "{lines:?}"
        );
        // no x offset was named, so no line carries one
        assert!(!lines.iter().any(|l| l.contains("x-offset")), "{lines:?}");
        // the default madctl reaches the firmware, not the overlay
        let blob = st7789(&set).unwrap().firmware.unwrap();
        assert!(hex(&blob).contains("3601c0"), "{}", hex(&blob));
    }

    /// An ST7789 comes in many sizes, and a geometry that the options do not
    /// name would name the wrong panel.
    #[test]
    fn the_overlay_takes_no_geometry_default() {
        for set in [vec![], vec![("width".to_string(), json!(240))]] {
            assert_eq!(
                st7789(&set).unwrap_err().to_string(),
                "pass width and height"
            );
        }
    }

    #[test]
    fn a_reset_pin_reaches_the_overlay() {
        let mut set = bonnet();
        set.push(("rst".to_string(), json!(27)));
        assert!(st7789(&set)
            .unwrap()
            .lines
            .contains(&"dtparam=dc-gpio=25,backlight-gpio=26,reset-gpio=27".to_string()));
    }

    /// The params come from the overlay's source, ssd1306-overlay.dts in
    /// raspberrypi/linux: `address` sets the node's reg and `height` its
    /// solomon,height. A 128x32 module takes the overlay README's typical
    /// line for it. ssd130x holds the init sequence for the whole family, so
    /// no firmware file goes with the lines.
    #[test]
    fn the_kernel_setup_is_the_bus_rate_and_the_overlay() {
        for (set, rate, overlay) in [
            (
                json!({"height": 32}),
                "dtparam=i2c_arm_baudrate=1000000",
                "dtoverlay=ssd1306,height=32,inverted,sequential",
            ),
            (
                json!({"height": 64}),
                "dtparam=i2c_arm_baudrate=1000000",
                "dtoverlay=ssd1306,height=64",
            ),
            (
                json!({"height": 64, "address": 0x3D}),
                "dtparam=i2c_arm_baudrate=1000000",
                "dtoverlay=ssd1306,height=64,address=0x3d",
            ),
            (
                json!({"height": 32, "address": 0x3D, "speed": 400_000}),
                "dtparam=i2c_arm_baudrate=400000",
                "dtoverlay=ssd1306,height=32,address=0x3d,inverted,sequential",
            ),
        ] {
            let setup = ssd1306(&object(&set)).unwrap();
            assert!(setup.firmware.is_none(), "{set}");
            assert_eq!(setup.lines, ["dtparam=i2c_arm=on", rate, overlay], "{set}");
        }
    }

    #[test]
    fn a_kernel_setup_names_what_the_overlay_cannot_take() {
        for (set, msg) in [
            (json!({}), "ssd1306 needs height=32 or height=64"),
            (
                json!({"height": 48}),
                "ssd1306 needs height=32 or height=64, got height=48",
            ),
            (
                json!({"height": 32, "address": 0x80}),
                "address 128 is not a 7-bit I2C address",
            ),
            (
                json!({"height": 32, "speed": 0}),
                "speed 0 is not a bus rate in Hz",
            ),
        ] {
            assert_eq!(
                ssd1306(&object(&set)).unwrap_err().to_string(),
                msg,
                "{set}"
            );
        }
    }
}
