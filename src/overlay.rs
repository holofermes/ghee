//! The kernel's own setup for a panel, what `ghee overlay` writes out: the
//! config.txt lines that hand the panel to a kernel display driver, and the
//! file that those lines need, the firmware that the driver loads or the
//! source of an overlay that the Pi lacks. The panel then answers `--display
//! kms` like any other display.
//!
//! Pins are BCM numbers.

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

/// --opts sets these options. Each k=v pair parses to one value.
pub type Opts = [(String, Value)];

/// A command byte and its data, empty for a command that takes none.
/// A delay is command 0x00 carrying the milliseconds, panel-mipi-dbi's own
/// encoding.
pub type Command = (u8, Vec<u8>);

/// The file that a panel's config.txt lines need on the Pi.
#[derive(Debug)]
pub enum File {
    /// The init sequence that panel-mipi-dbi loads from
    /// /lib/firmware/panel.bin.
    Firmware(Vec<u8>),
    /// The source of the overlay `name`, which dtc compiles into
    /// /boot/firmware/overlays/<name>.dtbo.
    Overlay { name: &'static str, source: String },
}

#[derive(Debug)]
pub struct Setup {
    /// None when the kernel and a stock overlay hold everything.
    pub file: Option<File>,
    /// The config.txt lines that bring the card up.
    pub lines: Vec<String>,
}

/// A panel that `ghee overlay` sets up.
pub struct Panel {
    pub name: &'static str,
    /// What --opts sets for this panel, in declaration order.
    pub options: &'static [&'static str],
    pub setup: fn(&Opts) -> Result<Setup>,
}

const PANELS: [&Panel; 6] = [&ST7789, &SSD1306, &SH1106, &SH1106_SPI, &SSD1322, &ILI9341];

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

/// True when --opts set this option.
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

fn bool_opt(opts: &Opts, name: &str) -> Result<bool> {
    match opt(opts, name) {
        None => Ok(false),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| anyhow!("{name} takes True or False, got {v}")),
    }
}

/// A bus clock in Hz.
fn speed_opt(opts: &Opts, default: i64) -> Result<i64> {
    let speed = int_opt(opts, "speed", default)?;
    if speed <= 0 {
        bail!("speed {speed} is not a bus rate in Hz");
    }
    Ok(speed)
}

fn address_opt(opts: &Opts, default: i64) -> Result<i64> {
    let address = int_opt(opts, "address", default)?;
    if !(0..0x80).contains(&address) {
        bail!("address {address} is not a 7-bit I2C address");
    }
    Ok(address)
}

/// The chip select of a panel on SPI bus 0. The overlay turns off the
/// spidev node that holds it.
fn cs_opt(opts: &Opts) -> Result<i64> {
    match int_opt(opts, "spi_device", 0)? {
        cs @ (0 | 1) => Ok(cs),
        cs => bail!("spi_device {cs} is not a chip select of SPI bus 0, which has 0 and 1"),
    }
}

// MIPI DCS command bytes, shared by the ST7789 and the ILI9341
const SWRESET: u8 = 0x01;
const SLPOUT: u8 = 0x11;
const NORON: u8 = 0x13;
const INVON: u8 = 0x21;
const GAMSET: u8 = 0x26;
const DISPOFF: u8 = 0x28;
const DISPON: u8 = 0x29;
const MADCTL: u8 = 0x36;
const COLMOD: u8 = 0x3A;

/// What a panel on panel-mipi-dbi takes when --opts leaves it out.
struct MipiDbi {
    dc: i64,
    rst: Option<i64>,
    backlight: Option<i64>,
    /// The SPI clock in Hz.
    speed: i64,
}

/// MADCTL sets row order, column order and RGB/BGR: the panel rotation and
/// mirroring. The value is per board. A wrong value flips the image.
fn madctl_opt(opts: &Opts, default: u8) -> Result<u8> {
    let madctl = int_opt(opts, "madctl", default.into())?;
    u8::try_from(madctl).map_err(|_| anyhow!("madctl {madctl} is not a byte"))
}

/// The config.txt lines that hand a `width` x `height` panel to
/// panel-mipi-dbi, through the Pi's mipi-dbi-spi overlay.
fn mipi_dbi_spi(opts: &Opts, width: i32, height: i32, d: &MipiDbi) -> Result<Vec<String>> {
    let Some(dc) = pin_opt(opts, "dc", Some(d.dc))? else {
        bail!("dc is the panel's command pin and takes a GPIO number");
    };
    let (bus, device) = (
        int_opt(opts, "spi_bus", 0)?,
        int_opt(opts, "spi_device", 0)?,
    );
    let mut lines = vec![
        format!(
            "dtoverlay=mipi-dbi-spi,spi{bus}-{device},speed={}",
            speed_opt(opts, d.speed)?
        ),
        format!("dtparam=width={width},height={height}"),
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
    if let Some(bl) = pin_opt(opts, "backlight", d.backlight)? {
        pins.push_str(&format!(",backlight-gpio={bl}"));
    }
    if let Some(rst) = pin_opt(opts, "rst", d.rst)? {
        pins.push_str(&format!(",reset-gpio={rst}"));
    }
    lines.push(pins);
    // MISO reaches no hobby breakout, and the driver reads the controller
    // without this.
    lines.push("dtparam=write-only".to_string());
    Ok(lines)
}

/// What --opts sets on a panel under panel-mipi-dbi.
const MIPI_DBI_OPTIONS: &[&str] = &[
    "width",
    "height",
    "x_offset",
    "y_offset",
    "dc",
    "rst",
    "backlight",
    "speed",
    "spi_bus",
    "spi_device",
    "madctl",
];

const ST7789_PINS: MipiDbi = MipiDbi {
    dc: 25,
    rst: None,
    backlight: Some(22),
    // panel-mipi-dbi renders noise at 99 MHz on a VC4-class board, and this
    // clock is clean there. A panel that takes more says so with
    // `--opts speed=`.
    speed: 62_500_000,
};
const ST7789_MADCTL: u8 = 0xC0;

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
    let madctl = madctl_opt(opts, ST7789_MADCTL)?;
    let (width, height) = (i32_opt(opts, "width", 0)?, i32_opt(opts, "height", 0)?);
    Ok(Setup {
        lines: mipi_dbi_spi(opts, width, height, &ST7789_PINS)?,
        file: Some(File::Firmware(firmware(&st7789_init(madctl)))),
    })
}

const ST7789: Panel = Panel {
    name: "st7789",
    options: MIPI_DBI_OPTIONS,
    setup: st7789,
};

// ILI9341 command bytes, named as the kernel's ili9341 driver names them
const ILI9341_FRMCTR1: u8 = 0xB1;
const ILI9341_DISCTRL: u8 = 0xB6;
const ILI9341_ETMOD: u8 = 0xB7;
const ILI9341_PWCTRL1: u8 = 0xC0;
const ILI9341_PWCTRL2: u8 = 0xC1;
const ILI9341_VMCTRL1: u8 = 0xC5;
const ILI9341_VMCTRL2: u8 = 0xC7;
const ILI9341_PWCTRLA: u8 = 0xCB;
const ILI9341_PWCTRLB: u8 = 0xCF;
const ILI9341_PGAMCTRL: u8 = 0xE0;
const ILI9341_NGAMCTRL: u8 = 0xE1;
const ILI9341_DTCTRLA: u8 = 0xE8;
const ILI9341_DTCTRLB: u8 = 0xEA;
const ILI9341_PWRSEQ: u8 = 0xED;
const ILI9341_EN3GAM: u8 = 0xF2;
const ILI9341_PUMPCTRL: u8 = 0xF7;

/// The pins of the Adafruit EYESPI Pi Beret, which Waveshare's 2.4inch
/// ILI9341 module shares. The Pi's own overlay for Adafruit's 2.8" ILI9341
/// PiTFT runs it at 32 MHz.
const ILI9341_PINS: MipiDbi = MipiDbi {
    dc: 25,
    rst: Some(27),
    backlight: Some(18),
    speed: 32_000_000,
};
/// The controller's memory is 240x320. The kernel driver scans it upright
/// with MADCTL MX|BGR, and turned a quarter for 320x240 with MV|BGR.
const ILI9341_MADCTL: u8 = 0x48;
const ILI9341_MADCTL_TURNED: u8 = 0x28;

/// panel-mipi-dbi resets the controller and replays this sequence: the one
/// that the kernel's ili9341 driver sends after the same reset, for a TN
/// panel, MADCTL last as the driver sends it. An IPS panel shows its colours
/// inverted under it, and `invert` adds INVON for one.
fn ili9341_init(madctl: u8, invert: bool) -> Vec<Command> {
    let mut init = vec![
        (DISPOFF, vec![]),
        (ILI9341_PWCTRLB, vec![0x00, 0xC1, 0x30]),
        (ILI9341_PWRSEQ, vec![0x64, 0x03, 0x12, 0x81]),
        (ILI9341_DTCTRLA, vec![0x85, 0x00, 0x78]),
        (ILI9341_PWCTRLA, vec![0x39, 0x2C, 0x00, 0x34, 0x02]),
        (ILI9341_PUMPCTRL, vec![0x20]),
        (ILI9341_DTCTRLB, vec![0x00, 0x00]),
        (ILI9341_PWCTRL1, vec![0x23]),
        (ILI9341_PWCTRL2, vec![0x10]),
        (ILI9341_VMCTRL1, vec![0x3E, 0x28]),
        (ILI9341_VMCTRL2, vec![0x86]),
        (COLMOD, vec![0x55]),
        (ILI9341_FRMCTR1, vec![0x00, 0x1B]),
        (ILI9341_EN3GAM, vec![0x00]),
        (GAMSET, vec![0x01]),
        (
            ILI9341_PGAMCTRL,
            vec![
                0x0F, 0x31, 0x2B, 0x0C, 0x0E, 0x08, 0x4E, 0xF1, 0x37, 0x07, 0x10, 0x03, 0x0E, 0x09,
                0x00,
            ],
        ),
        (
            ILI9341_NGAMCTRL,
            vec![
                0x00, 0x0E, 0x14, 0x03, 0x11, 0x07, 0x31, 0xC1, 0x48, 0x08, 0x0F, 0x0C, 0x31, 0x36,
                0x0F,
            ],
        ),
        (ILI9341_ETMOD, vec![0x07]),
        (ILI9341_DISCTRL, vec![0x08, 0x82, 0x27, 0x00]),
    ];
    if invert {
        init.push((INVON, vec![]));
    }
    init.extend([
        (SLPOUT, vec![]),
        delay(100),
        (DISPON, vec![]),
        delay(100),
        (MADCTL, vec![madctl]),
    ]);
    init
}

/// The config.txt lines that hand an ILI9341 to panel-mipi-dbi, and the
/// firmware named in those lines. The panel is 320x240 unless the options
/// say otherwise, and MADCTL follows the geometry until `madctl` sets it.
fn ili9341(opts: &Opts) -> Result<Setup> {
    let (width, height) = (i32_opt(opts, "width", 320)?, i32_opt(opts, "height", 240)?);
    let madctl = match width > height {
        true => madctl_opt(opts, ILI9341_MADCTL_TURNED)?,
        false => madctl_opt(opts, ILI9341_MADCTL)?,
    };
    let invert = bool_opt(opts, "invert")?;
    Ok(Setup {
        lines: mipi_dbi_spi(opts, width, height, &ILI9341_PINS)?,
        file: Some(File::Firmware(firmware(&ili9341_init(madctl, invert)))),
    })
}

const ILI9341: Panel = Panel {
    name: "ili9341",
    options: &[
        "width",
        "height",
        "x_offset",
        "y_offset",
        "dc",
        "rst",
        "backlight",
        "speed",
        "spi_bus",
        "spi_device",
        "madctl",
        "invert",
    ],
    setup: ili9341,
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
    let address = address_opt(opts, SSD1306_ADDRESS)?;
    let speed = speed_opt(opts, SSD1306_SPEED)?;
    // the overlay's node sits at the default address
    let address = match address {
        SSD1306_ADDRESS => String::new(),
        a => format!(",address=0x{a:02x}"),
    };
    Ok(Setup {
        file: None,
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

/// The source of an overlay that puts the node `node`, with the properties
/// `props`, on the bus labelled `bus` in the Pi's device tree. On SPI bus 0,
/// `spidev` names the chip select whose spidev node the overlay turns off.
fn overlay_source(bus: &str, spidev: Option<i64>, node: &str, props: &[String]) -> String {
    let mut out = "/dts-v1/;\n/plugin/;\n\n/ {\n\tcompatible = \"brcm,bcm2835\";\n".to_string();
    let mut fragment = 0;
    if let Some(cs) = spidev {
        out.push_str(&format!(
            "\n\tfragment@0 {{\n\
             \t\ttarget = <&spidev{cs}>;\n\
             \t\t__overlay__ {{\n\
             \t\t\tstatus = \"disabled\";\n\
             \t\t}};\n\
             \t}};\n"
        ));
        fragment = 1;
    }
    out.push_str(&format!(
        "\n\tfragment@{fragment} {{\n\
         \t\ttarget = <&{bus}>;\n\
         \t\t__overlay__ {{\n\
         \t\t\t#address-cells = <1>;\n\
         \t\t\t#size-cells = <0>;\n\
         \t\t\tstatus = \"okay\";\n\
         \n\
         \t\t\t{node} {{\n"
    ));
    for p in props {
        out.push_str(&format!("\t\t\t\t{p};\n"));
    }
    out.push_str("\t\t\t};\n\t\t};\n\t};\n};\n");
    out
}

/// The setup of a panel whose overlay ghee writes: the overlay's config.txt
/// line after `lines`, and the overlay's source.
fn own_overlay(name: &'static str, mut lines: Vec<String>, source: String) -> Setup {
    lines.push(format!("dtoverlay={name}"));
    Setup {
        file: Some(File::Overlay { name, source }),
        lines,
    }
}

/// The node's properties for a panel of the Solomon family on SPI bus 0:
/// its chip select, the clock, and the command and reset pins.
fn ssd130x_spi(opts: &Opts, compatible: &str, speed: i64) -> Result<(i64, Vec<String>)> {
    let cs = cs_opt(opts)?;
    let Some(dc) = pin_opt(opts, "dc", Some(SSD130X_SPI_DC))? else {
        bail!("dc is the panel's command pin and takes a GPIO number");
    };
    let mut props = vec![
        format!("compatible = \"{compatible}\""),
        format!("reg = <{cs}>"),
        format!("spi-max-frequency = <{}>", speed_opt(opts, speed)?),
        format!("dc-gpios = <&gpio {dc} 0>"),
    ];
    // the reset line is active low
    if let Some(rst) = pin_opt(opts, "rst", Some(SSD130X_SPI_RST))? {
        props.push(format!("reset-gpios = <&gpio {rst} 1>"));
    }
    Ok((cs, props))
}

/// The pins of the Pi's own SPI OLED overlays, which Waveshare's 1.3inch
/// OLED HAT shares.
const SSD130X_SPI_DC: i64 = 24;
const SSD130X_SPI_RST: i64 = 25;

const SH1106_COMPATIBLE: &str = "sinowealth,sh1106";
const SH1106_ADDRESS: i64 = 0x3C;
/// The SH1106's I2C fast mode, from its datasheet. A 128x64 frame takes
/// about 25 ms at this rate.
const SH1106_SPEED: i64 = 400_000;
/// The SH1106's SPI clock, a 250 ns cycle at least, from its datasheet.
const SH1106_SPI_SPEED: i64 = 4_000_000;

/// The properties that fit a 128x64 SH1106 module to ssd130x. The glass
/// starts at column 2 of the controller's 132 and at page 0, where ssd130x
/// starts at page 1 unless told, and the module scans its rows reversed,
/// COM scan 0xC8.
fn sh1106_glass() -> Vec<String> {
    [
        "solomon,width = <128>",
        "solomon,height = <64>",
        "solomon,page-offset = <0>",
        "solomon,col-offset = <2>",
        "solomon,com-invdir",
    ]
    .map(String::from)
    .to_vec()
}

/// The config.txt lines that hand an SH1106 to ssd130x on I2C bus 1, and
/// the overlay named in them. The Pi's own SH1106 overlay is for SPI and
/// for the fbtft driver. The bus rate line sets every device on the bus.
fn sh1106(opts: &Opts) -> Result<Setup> {
    let address = address_opt(opts, SH1106_ADDRESS)?;
    let speed = speed_opt(opts, SH1106_SPEED)?;
    let props = [
        vec![
            format!("compatible = \"{SH1106_COMPATIBLE}\""),
            format!("reg = <0x{address:02x}>"),
        ],
        sh1106_glass(),
    ]
    .concat();
    let source = overlay_source("i2c1", None, &format!("oled@{address:x}"), &props);
    let lines = vec![
        "dtparam=i2c_arm=on".to_string(),
        format!("dtparam=i2c_arm_baudrate={speed}"),
    ];
    Ok(own_overlay("ghee-sh1106", lines, source))
}

const SH1106: Panel = Panel {
    name: "sh1106",
    options: &["address", "speed"],
    setup: sh1106,
};

/// The config.txt line that hands an SH1106 to ssd130x on SPI bus 0, and the
/// overlay named in it. The Pi's own sh1106-spi overlay sets properties that
/// only the fbtft driver reads. ssd130x and fbtft both take the SH1106's
/// compatible, so the overlay cannot choose between them.
fn sh1106_spi(opts: &Opts) -> Result<Setup> {
    let (cs, props) = ssd130x_spi(opts, SH1106_COMPATIBLE, SH1106_SPI_SPEED)?;
    let props = [props, sh1106_glass()].concat();
    let source = overlay_source("spi0", Some(cs), &format!("oled@{cs}"), &props);
    Ok(own_overlay("ghee-sh1106-spi", vec![], source))
}

const SH1106_SPI: Panel = Panel {
    name: "sh1106-spi",
    options: &["dc", "rst", "speed", "spi_device"],
    setup: sh1106_spi,
};

/// The SSD1322's 480 segments, four pixels to a column address, and its 128
/// rows.
const SSD1322_SEGMENTS: i64 = 480;
const SSD1322_ROWS: i64 = 128;
/// The SPI clock, a 100 ns cycle at least, from the SSD1322's datasheet.
const SSD1322_SPEED: i64 = 10_000_000;
/// A 256x64 module's glass sits on columns 28 to 91.
const SSD1322_WIDTH: i64 = 256;
const SSD1322_HEIGHT: i64 = 64;
const SSD1322_COL_OFFSET: i64 = 28;

/// The config.txt line that hands an SSD1322 to ssd130x on SPI bus 0, and
/// the overlay named in it. `solomon,col-offset` counts column addresses,
/// four pixels each. A display clock divided by 1 redraws the glass twice
/// as often as the driver's default of 2, and frc16 needs a redraw for
/// each of its frames.
fn ssd1322(opts: &Opts) -> Result<Setup> {
    let width = int_opt(opts, "width", SSD1322_WIDTH)?;
    let height = int_opt(opts, "height", SSD1322_HEIGHT)?;
    if !(1..=SSD1322_SEGMENTS).contains(&width) || !(1..=SSD1322_ROWS).contains(&height) {
        bail!("ssd1322 drives up to {SSD1322_SEGMENTS}x{SSD1322_ROWS}, got {width}x{height}");
    }
    let col = int_opt(opts, "col_offset", SSD1322_COL_OFFSET)?;
    if !(0..=(SSD1322_SEGMENTS - width) / 4).contains(&col) {
        bail!(
            "col_offset {col} with width {width} runs past the SSD1322's {} columns",
            SSD1322_SEGMENTS / 4
        );
    }
    let (cs, mut props) = ssd130x_spi(opts, "solomon,ssd1322", SSD1322_SPEED)?;
    props.extend([
        "spi-cpol".to_string(),
        "spi-cpha".to_string(),
        format!("solomon,width = <{width}>"),
        format!("solomon,height = <{height}>"),
        format!("solomon,col-offset = <{col}>"),
        "solomon,dclk-div = <1>".to_string(),
    ]);
    let source = overlay_source("spi0", Some(cs), &format!("oled@{cs}"), &props);
    Ok(own_overlay("ghee-ssd1322", vec![], source))
}

const SSD1322: Panel = Panel {
    name: "ssd1322",
    options: &[
        "width",
        "height",
        "col_offset",
        "dc",
        "rst",
        "speed",
        "spi_device",
    ],
    setup: ssd1322,
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

    fn firmware_of(setup: Setup) -> Vec<u8> {
        match setup.file {
            Some(File::Firmware(blob)) => blob,
            file => panic!("no firmware: {file:?}"),
        }
    }

    /// The overlay's name and source.
    fn overlay_of(setup: &Setup) -> (&str, &str) {
        match &setup.file {
            Some(File::Overlay { name, source }) => (name, source),
            file => panic!("no overlay: {file:?}"),
        }
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
        let blob = firmware_of(st7789(&bonnet()).unwrap());
        assert_eq!(hex(&blob), FIRMWARE_ON_THE_PANEL);
        assert_eq!(blob.len(), 41);
    }

    /// One option sets the bus clock of every panel, on I2C and on SPI.
    #[test]
    fn every_panel_takes_its_bus_clock_as_speed() {
        for panel in PANELS {
            assert!(panel.options.contains(&"speed"), "{}", panel.name);
        }
        let mut set = bonnet();
        set.push(("speed".to_string(), json!(16_000_000)));
        let fast = "dtoverlay=mipi-dbi-spi,spi0-0,speed=16000000";
        assert_eq!(st7789(&set).unwrap().lines[0], fast);
        assert_eq!(ili9341(&set).unwrap().lines[0], fast);
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
        let blob = firmware_of(st7789(&set).unwrap());
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
            assert!(setup.file.is_none(), "{set}");
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

    /// The node follows ssd1306-overlay.dts in raspberrypi/linux, on the same
    /// bus, with the SH1106's compatible from the ssd1307fb binding. A
    /// 1.3" 128x64 module shows columns 2 to 129 of the controller's 132
    /// (u8g2's sh1106_128x64_noname, x offset 2), starts at page 0, and scans
    /// its COM rows reversed, 0xC8 in u8g2's init for it.
    const SH1106_ON_I2C: &str = "/dts-v1/;
/plugin/;

/ {
	compatible = \"brcm,bcm2835\";

	fragment@0 {
		target = <&i2c1>;
		__overlay__ {
			#address-cells = <1>;
			#size-cells = <0>;
			status = \"okay\";

			oled@3c {
				compatible = \"sinowealth,sh1106\";
				reg = <0x3c>;
				solomon,width = <128>;
				solomon,height = <64>;
				solomon,page-offset = <0>;
				solomon,col-offset = <2>;
				solomon,com-invdir;
			};
		};
	};
};
";

    /// The same module on SPI bus 0 at the SH1106 datasheet's 4 MHz, its
    /// command and reset lines on the pins of the Pi's own sh1106-spi
    /// overlay, whose spidev fragment it keeps.
    const SH1106_ON_SPI: &str = "/dts-v1/;
/plugin/;

/ {
	compatible = \"brcm,bcm2835\";

	fragment@0 {
		target = <&spidev0>;
		__overlay__ {
			status = \"disabled\";
		};
	};

	fragment@1 {
		target = <&spi0>;
		__overlay__ {
			#address-cells = <1>;
			#size-cells = <0>;
			status = \"okay\";

			oled@0 {
				compatible = \"sinowealth,sh1106\";
				reg = <0>;
				spi-max-frequency = <4000000>;
				dc-gpios = <&gpio 24 0>;
				reset-gpios = <&gpio 25 1>;
				solomon,width = <128>;
				solomon,height = <64>;
				solomon,page-offset = <0>;
				solomon,col-offset = <2>;
				solomon,com-invdir;
			};
		};
	};
};
";

    #[test]
    fn an_sh1106_gets_an_overlay_of_its_own_on_either_bus() {
        let setup = sh1106(&[]).unwrap();
        assert_eq!(
            setup.lines,
            [
                "dtparam=i2c_arm=on",
                "dtparam=i2c_arm_baudrate=400000",
                "dtoverlay=ghee-sh1106",
            ]
        );
        assert_eq!(overlay_of(&setup), ("ghee-sh1106", SH1106_ON_I2C));

        let setup = sh1106_spi(&[]).unwrap();
        assert_eq!(setup.lines, ["dtoverlay=ghee-sh1106-spi"]);
        assert_eq!(overlay_of(&setup), ("ghee-sh1106-spi", SH1106_ON_SPI));
    }

    #[test]
    fn the_sh1106_options_reach_its_overlay() {
        let setup = sh1106(&object(&json!({"address": 0x3D, "speed": 1_000_000}))).unwrap();
        assert_eq!(setup.lines[1], "dtparam=i2c_arm_baudrate=1000000");
        let (_, source) = overlay_of(&setup);
        assert!(source.contains("\t\t\toled@3d {\n"), "{source}");
        assert!(source.contains("\t\t\t\treg = <0x3d>;\n"), "{source}");

        let set = json!({"spi_device": 1, "dc": 5, "rst": null, "speed": 8_000_000});
        let setup = sh1106_spi(&object(&set)).unwrap();
        let (_, source) = overlay_of(&setup);
        for want in [
            "target = <&spidev1>;",
            "oled@1 {",
            "reg = <1>;",
            "spi-max-frequency = <8000000>;",
            "dc-gpios = <&gpio 5 0>;",
        ] {
            assert!(source.contains(want), "{want} in {source}");
        }
        assert!(!source.contains("reset-gpios"), "{source}");
    }

    /// The properties of an overlay that drove a 3.12" 256x64 SSD1322
    /// module, SPI mode 3, under ssd130x's SSD1322 support, on the pins of
    /// the Pi's ssd13xx SPI overlays. Its glass is columns 28 to 91, as in
    /// Newhaven's NHD-3.12-25664 init and u8g2's ssd1322_256x64, and 10 MHz
    /// is the datasheet's clock.
    const SSD1322_ON_SPI: &str = "/dts-v1/;
/plugin/;

/ {
	compatible = \"brcm,bcm2835\";

	fragment@0 {
		target = <&spidev0>;
		__overlay__ {
			status = \"disabled\";
		};
	};

	fragment@1 {
		target = <&spi0>;
		__overlay__ {
			#address-cells = <1>;
			#size-cells = <0>;
			status = \"okay\";

			oled@0 {
				compatible = \"solomon,ssd1322\";
				reg = <0>;
				spi-max-frequency = <10000000>;
				dc-gpios = <&gpio 24 0>;
				reset-gpios = <&gpio 25 1>;
				spi-cpol;
				spi-cpha;
				solomon,width = <256>;
				solomon,height = <64>;
				solomon,col-offset = <28>;
				solomon,dclk-div = <1>;
			};
		};
	};
};
";

    #[test]
    fn an_ssd1322_gets_an_overlay_of_its_own() {
        let setup = ssd1322(&[]).unwrap();
        assert_eq!(setup.lines, ["dtoverlay=ghee-ssd1322"]);
        assert_eq!(overlay_of(&setup), ("ghee-ssd1322", SSD1322_ON_SPI));

        // u8g2's ssd1322_240x128 sits at column 24
        let set = json!({"width": 240, "height": 128, "col_offset": 24});
        let setup = ssd1322(&object(&set)).unwrap();
        let (_, source) = overlay_of(&setup);
        for want in [
            "solomon,width = <240>;",
            "solomon,height = <128>;",
            "solomon,col-offset = <24>;",
        ] {
            assert!(source.contains(want), "{want} in {source}");
        }
    }

    /// The kernel's own ili9341 driver (drivers/gpu/drm/tiny/ili9341.c)
    /// sends these commands after the reset that panel-mipi-dbi also runs,
    /// MADCTL last: MV|BGR, its rotation 90 and the one that the Pi's
    /// pitft28 overlay picks, for 320x240.
    const ILI9341_FIRMWARE: &str = "4d495049204442490000000000000001\
                                    2800\
                                    cf0300c130\
                                    ed0464031281\
                                    e803850078\
                                    cb05392c003402\
                                    f70120\
                                    ea020000\
                                    c00123\
                                    c10110\
                                    c5023e28\
                                    c70186\
                                    3a0155\
                                    b102001b\
                                    f20100\
                                    260101\
                                    e00f0f312b0c0e084ef1370710030e0900\
                                    e10f000e1403110731c148080f0c31360f\
                                    b70107\
                                    b60408822700\
                                    1100\
                                    000164\
                                    2900\
                                    000164\
                                    360128";

    #[test]
    fn an_ili9341_takes_the_kernel_drivers_init() {
        let setup = ili9341(&[]).unwrap();
        assert_eq!(
            setup.lines,
            [
                "dtoverlay=mipi-dbi-spi,spi0-0,speed=32000000",
                "dtparam=width=320,height=240",
                "dtparam=dc-gpio=25,backlight-gpio=18,reset-gpio=27",
                "dtparam=write-only",
            ]
        );
        assert_eq!(hex(&firmware_of(setup)), ILI9341_FIRMWARE);
    }

    /// Upright, the kernel driver's rotation 0 is MX|BGR. An IPS panel
    /// takes INVON, which LCDwiki's ILI9341V init sends.
    #[test]
    fn an_ili9341_madctl_follows_its_geometry_and_invert_adds_invon() {
        let set = json!({"width": 240, "height": 320});
        let blob = hex(&firmware_of(ili9341(&object(&set)).unwrap()));
        assert!(blob.ends_with("2900000164360148"), "{blob}");
        let set = json!({"invert": true});
        let blob = hex(&firmware_of(ili9341(&object(&set)).unwrap()));
        assert!(blob.contains("b6040882270021001100"), "{blob}");
        let set = json!({"madctl": 0xE8});
        let blob = hex(&firmware_of(ili9341(&object(&set)).unwrap()));
        assert!(blob.ends_with("3601e8"), "{blob}");
    }

    #[test]
    fn a_panel_names_the_option_that_it_cannot_take() {
        for (name, set, msg) in [
            (
                "sh1106-spi",
                json!({"spi_device": 2}),
                "spi_device 2 is not a chip select of SPI bus 0, which has 0 and 1",
            ),
            (
                "sh1106-spi",
                json!({"dc": null}),
                "dc is the panel's command pin and takes a GPIO number",
            ),
            (
                "ssd1322",
                json!({"width": 500}),
                "ssd1322 drives up to 480x128, got 500x64",
            ),
            (
                "ssd1322",
                json!({"height": 0}),
                "ssd1322 drives up to 480x128, got 256x0",
            ),
            (
                "ssd1322",
                json!({"col_offset": 57}),
                "col_offset 57 with width 256 runs past the SSD1322's 120 columns",
            ),
            (
                "ssd1322",
                json!({"col_offset": -1}),
                "col_offset -1 with width 256 runs past the SSD1322's 120 columns",
            ),
            (
                "ili9341",
                json!({"invert": 1}),
                "invert takes True or False, got 1",
            ),
            (
                "ili9341",
                json!({"madctl": 256}),
                "madctl 256 is not a byte",
            ),
        ] {
            let setup = panel(name).unwrap().setup;
            assert_eq!(
                setup(&object(&set)).unwrap_err().to_string(),
                msg,
                "{name} {set}"
            );
        }
    }
}
