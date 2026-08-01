//! Port of ghee/drivers/st7789.py.
//!
//! Packs each RGB frame to RGB565 and blits it on a private SPI handle, with DC,
//! reset and backlight driven through the GPIO. Chip select is the hardware CE
//! line: the kernel SPI controller owns that GPIO and the pin claim cannot take it.
//!
//! Pins are BCM numbers. The defaults are the 135x240 breakout this was written on.

use std::fs;
use std::thread::sleep;
use std::time::Duration;

use anyhow::{bail, Result};
use rppal::spi::{Bus, Mode, SlaveSelect, Spi};

use crate::drivers::base::{DisplayDriver, RGB565};
use crate::drivers::gpio::Pins;
use crate::drivers::{i32_opt, int_opt, pin_opt, DriverOption, DriverSpec, Opts};
use crate::engine::pack_frame_into;

const SWRESET: u8 = 0x01;
const SLPOUT: u8 = 0x11;
const NORON: u8 = 0x13;
const INVON: u8 = 0x21;
const DISPON: u8 = 0x29;
const RAMWR: u8 = 0x2C;
const CASET: u8 = 0x2A;
const RASET: u8 = 0x2B;
const MADCTL: u8 = 0x36;
const COLMOD: u8 = 0x3A;

const DEFAULT_WIDTH: i64 = 135;
const DEFAULT_HEIGHT: i64 = 240;
const DEFAULT_X_OFFSET: i64 = 53;
const DEFAULT_Y_OFFSET: i64 = 40;
const DEFAULT_DC: i64 = 25;
const DEFAULT_BACKLIGHT: i64 = 22;
const DEFAULT_BAUDRATE: i64 = 99000000;
const DEFAULT_SPI_BUS: i64 = 0;
const DEFAULT_SPI_DEVICE: i64 = 0;
// MADCTL: row/column order and RGB/BGR, so the panel's rotation and mirroring.
// Per board, and a wrong value shows as a flipped image.
const DEFAULT_MADCTL: i64 = 0xC0;

/// A command byte and its data, empty for a command that takes none.
type Command = (u8, Vec<u8>);

/// struct.pack(">HH", a, b).
fn pack_hh(a: i32, b: i32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(4);
    for v in [a, b] {
        let Ok(v) = u16::try_from(v) else {
            bail!("panel coordinate {v} does not fit 16 bits");
        };
        out.extend_from_slice(&v.to_be_bytes());
    }
    Ok(out)
}

fn init_panel(
    width: i32,
    height: i32,
    x_offset: i32,
    y_offset: i32,
    madctl: u8,
) -> Result<Vec<Command>> {
    let cols = pack_hh(x_offset, width + x_offset)?;
    let rows = pack_hh(y_offset, height + y_offset)?;
    Ok(vec![
        (SWRESET, vec![]),
        (SLPOUT, vec![]),
        (COLMOD, vec![0x55]), // 16 bit colour
        (MADCTL, vec![0x08]),
        (CASET, cols),
        (RASET, rows),
        (INVON, vec![]),
        (NORON, vec![]),
        (DISPON, vec![]),
        (MADCTL, vec![madctl]),
    ])
}

fn window(
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    x_offset: i32,
    y_offset: i32,
) -> Result<[Command; 2]> {
    Ok([
        (CASET, pack_hh(x0 + x_offset, x1 + x_offset)?),
        (RASET, pack_hh(y0 + y_offset, y1 + y_offset)?),
    ])
}

/// spidev refuses a transfer larger than its own buffer, so a blit goes out in
/// bufsiz chunks.
fn spi_block_size() -> usize {
    fs::read_to_string("/sys/module/spidev/parameters/bufsiz")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(4096)
}

fn bus(n: i64) -> Result<Bus> {
    Ok(match n {
        0 => Bus::Spi0,
        1 => Bus::Spi1,
        2 => Bus::Spi2,
        3 => Bus::Spi3,
        4 => Bus::Spi4,
        5 => Bus::Spi5,
        6 => Bus::Spi6,
        _ => bail!("no SPI bus {n}"),
    })
}

fn slave_select(n: i64) -> Result<SlaveSelect> {
    Ok(match n {
        0 => SlaveSelect::Ss0,
        1 => SlaveSelect::Ss1,
        2 => SlaveSelect::Ss2,
        3 => SlaveSelect::Ss3,
        4 => SlaveSelect::Ss4,
        5 => SlaveSelect::Ss5,
        6 => SlaveSelect::Ss6,
        7 => SlaveSelect::Ss7,
        _ => bail!("no SPI device {n}"),
    })
}

/// The backlight is claimed on its own, and a refusal is survivable: it is the one
/// line another kernel driver may already hold, and losing it must not cost the
/// panel its DC line.
fn backlight_pins(offset: Option<u8>) -> Option<Pins> {
    let offset = offset?;
    match Pins::new(&[Some(offset)]) {
        Ok(pins) => Some(pins),
        Err(e) => {
            eprintln!("st7789: GPIO{offset} is not available ({e}); no backlight control");
            None
        }
    }
}

pub struct St7789 {
    width: i32,
    height: i32,
    x_offset: i32,
    y_offset: i32,
    dc: u8,
    #[allow(dead_code)]
    madctl: u8,
    pins: Pins,
    bl: Option<u8>,
    bl_pins: Option<Pins>,
    spi: Spi,
    block: usize,
    buf: Vec<u8>,
    window_ready: bool,
}

impl St7789 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: i32,
        height: i32,
        x_offset: i32,
        y_offset: i32,
        dc: u8,
        rst: Option<u8>,
        backlight: Option<u8>,
        baudrate: u32,
        spi_bus: i64,
        spi_device: i64,
        madctl: u8,
    ) -> Result<St7789> {
        let pins = Pins::new(&[Some(dc), rst])?;
        let bl_pins = backlight_pins(backlight);
        let spi = Spi::new(
            bus(spi_bus)?,
            slave_select(spi_device)?,
            baudrate,
            Mode::Mode0,
        )?;
        let mut panel = St7789 {
            width,
            height,
            x_offset,
            y_offset,
            dc,
            madctl,
            pins,
            bl: backlight,
            bl_pins,
            spi,
            block: spi_block_size(),
            buf: vec![0u8; (width as usize) * (height as usize) * 2],
            window_ready: false,
        };
        if let Some(rst) = rst {
            panel.pins.set(rst, false)?;
            sleep(Duration::from_millis(50));
            panel.pins.set(rst, true)?;
            sleep(Duration::from_millis(50));
        }
        for (command, data) in init_panel(width, height, x_offset, y_offset, madctl)? {
            panel.cmd(command, &data)?;
        }
        panel.turn_on()?;
        Ok(panel)
    }

    #[allow(dead_code)]
    pub fn madctl(&self) -> u8 {
        self.madctl
    }

    fn spi_write(&mut self, data: &[u8]) -> Result<()> {
        for chunk in data.chunks(self.block) {
            let sent = self.spi.write(chunk)?;
            if sent != chunk.len() {
                bail!("SPI took {sent} of {} bytes", chunk.len());
            }
        }
        Ok(())
    }

    fn cmd(&mut self, command: u8, data: &[u8]) -> Result<()> {
        self.pins.set(self.dc, false)?;
        self.spi_write(&[command])?;
        if !data.is_empty() {
            self.pins.set(self.dc, true)?;
            self.spi_write(data)?;
        }
        Ok(())
    }

    fn blit(&mut self) -> Result<()> {
        if !self.window_ready {
            for (command, data) in window(
                0,
                0,
                self.width - 1,
                self.height - 1,
                self.x_offset,
                self.y_offset,
            )? {
                self.cmd(command, &data)?;
            }
            self.window_ready = true;
        }
        self.pins.set(self.dc, false)?;
        self.spi_write(&[RAMWR])?;
        self.pins.set(self.dc, true)?;
        let buf = std::mem::take(&mut self.buf);
        let sent = self.spi_write(&buf);
        self.buf = buf;
        sent
    }
}

impl DisplayDriver for St7789 {
    fn width(&self) -> i32 {
        self.width
    }

    fn height(&self) -> i32 {
        self.height
    }

    fn native_format(&self) -> &'static str {
        RGB565
    }

    fn write(&mut self, frame: &[u8]) -> Result<()> {
        pack_frame_into(frame, self.width, self.height, 3, false, &mut self.buf);
        self.blit()
    }

    fn clear(&mut self) -> Result<()> {
        self.buf.fill(0);
        self.blit()
    }

    fn turn_on(&mut self) -> Result<()> {
        match (self.bl, self.bl_pins.as_mut()) {
            (Some(bl), Some(pins)) => pins.set(bl, true),
            _ => Ok(()),
        }
    }

    fn turn_off(&mut self) -> Result<()> {
        match (self.bl, self.bl_pins.as_mut()) {
            (Some(bl), Some(pins)) => pins.set(bl, false),
            _ => Ok(()),
        }
    }
}

const OPTIONS: [DriverOption; 11] = [
    DriverOption::new("width", Some(DEFAULT_WIDTH)),
    DriverOption::new("height", Some(DEFAULT_HEIGHT)),
    DriverOption::new("x_offset", Some(DEFAULT_X_OFFSET)),
    DriverOption::new("y_offset", Some(DEFAULT_Y_OFFSET)),
    DriverOption::new("dc", Some(DEFAULT_DC)),
    DriverOption::new("rst", None),
    DriverOption::new("backlight", Some(DEFAULT_BACKLIGHT)),
    DriverOption::new("baudrate", Some(DEFAULT_BAUDRATE)),
    DriverOption::new("spi_bus", Some(DEFAULT_SPI_BUS)),
    DriverOption::new("spi_device", Some(DEFAULT_SPI_DEVICE)),
    DriverOption::new("madctl", Some(DEFAULT_MADCTL)),
];

fn open(opts: &Opts) -> Result<Box<dyn DisplayDriver>> {
    let baudrate = int_opt(opts, "baudrate", DEFAULT_BAUDRATE)?;
    let Ok(baudrate) = u32::try_from(baudrate) else {
        bail!("baudrate {baudrate} is not a clock speed");
    };
    let madctl = int_opt(opts, "madctl", DEFAULT_MADCTL)?;
    let Ok(madctl) = u8::try_from(madctl) else {
        bail!("madctl {madctl} is not a byte");
    };
    let Some(dc) = pin_opt(opts, "dc", Some(DEFAULT_DC))? else {
        bail!("dc is the panel's command pin and takes a GPIO number");
    };
    Ok(Box::new(St7789::new(
        i32_opt(opts, "width", DEFAULT_WIDTH)?,
        i32_opt(opts, "height", DEFAULT_HEIGHT)?,
        i32_opt(opts, "x_offset", DEFAULT_X_OFFSET)?,
        i32_opt(opts, "y_offset", DEFAULT_Y_OFFSET)?,
        dc,
        pin_opt(opts, "rst", None)?,
        pin_opt(opts, "backlight", Some(DEFAULT_BACKLIGHT))?,
        baudrate,
        int_opt(opts, "spi_bus", DEFAULT_SPI_BUS)?,
        int_opt(opts, "spi_device", DEFAULT_SPI_DEVICE)?,
        madctl,
    )?))
}

pub const SPEC: DriverSpec = DriverSpec {
    name: "st7789",
    options: &OPTIONS,
    open,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::pack_frame;

    /// What _cmd puts on the wire for a command stream: (DC level, bytes).
    fn wire(commands: &[Command]) -> Vec<(bool, String)> {
        let mut out = Vec::new();
        for (command, data) in commands {
            out.push((false, hex(&[*command])));
            if !data.is_empty() {
                out.push((true, hex(data)));
            }
        }
        out
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn recorded(rows: &[(bool, &str)]) -> Vec<(bool, String)> {
        rows.iter()
            .map(|(dc, data)| (*dc, data.to_string()))
            .collect()
    }

    /// What a Blinka-driven adafruit_rgb_display.st7789 puts on the wire for this
    /// panel, recorded off the real one.
    const INIT_ON_THE_WIRE: [(bool, &str); 15] = [
        (false, "01"), // SWRESET
        (false, "11"), // SLPOUT
        (false, "3a"), // COLMOD
        (true, "55"),
        (false, "36"), // MADCTL
        (true, "08"),
        (false, "2a"), // CASET
        (true, "003500bc"),
        (false, "2b"), // RASET
        (true, "00280118"),
        (false, "21"), // INVON
        (false, "13"), // NORON
        (false, "29"), // DISPON
        (false, "36"), // MADCTL
        (true, "c0"),
    ];

    const FIRST_FRAME_ON_THE_WIRE: [(bool, &str); 4] = [
        (false, "2a"),
        (true, "003500bb"),
        (false, "2b"),
        (true, "00280117"),
    ];

    fn defaults() -> Vec<Command> {
        init_panel(135, 240, 53, 40, 0xC0).unwrap()
    }

    #[test]
    fn init_matches_the_recorded_panel_stream() {
        assert_eq!(wire(&defaults()), recorded(&INIT_ON_THE_WIRE));
    }

    #[test]
    fn a_frame_sets_the_window_inside_the_panel_offsets() {
        assert_eq!(
            wire(&window(0, 0, 134, 239, 53, 40).unwrap()),
            recorded(&FIRST_FRAME_ON_THE_WIRE)
        );
    }

    #[test]
    fn geometry_reaches_the_addressing_commands() {
        let cmds = init_panel(240, 240, 0, 80, 0xC0).unwrap();
        assert_eq!(
            wire(&cmds)[6..10],
            recorded(&[
                (false, "2a"),
                (true, "000000f0"),
                (false, "2b"),
                (true, "00500140"),
            ])
        );
    }

    #[test]
    fn madctl_is_the_last_init_write() {
        // A 240x240 board came up rotated 180 under the 0xc0 default, so the
        // rotation/mirroring byte has to reach the panel from the constructor.
        let cmds = init_panel(135, 240, 53, 40, 0x00).unwrap();
        let sent = wire(&cmds);
        assert_eq!(
            sent[sent.len() - 2..],
            recorded(&[(false, "36"), (true, "00")])
        );
        // only that byte changed
        assert_eq!(sent[..sent.len() - 1], recorded(&INIT_ON_THE_WIRE[..14]));
    }

    #[test]
    fn a_coordinate_past_16_bits_is_an_error() {
        assert_eq!(
            init_panel(135, 70000, 53, 40, 0xC0)
                .unwrap_err()
                .to_string(),
            "panel coordinate 70040 does not fit 16 bits"
        );
    }

    #[test]
    fn a_frame_packs_to_big_endian_rgb565() {
        let frame: Vec<u8> = (0..135 * 240).flat_map(|_| [255u8, 0, 0]).collect();
        let packed = pack_frame(&frame, 135, 240, 3, false);
        assert_eq!(packed.len(), 135 * 240 * 2);
        assert_eq!(&packed[..2], b"\xf8\x00");
    }

    #[test]
    fn every_constructor_option_is_listed_with_its_default() {
        let listed: Vec<String> = OPTIONS
            .iter()
            .map(|o| match o.default {
                Some(v) => format!("{}={v}", o.name),
                None => format!("{}=None", o.name),
            })
            .collect();
        assert_eq!(
            listed.join(","),
            "width=135,height=240,x_offset=53,y_offset=40,dc=25,rst=None,\
             backlight=22,baudrate=99000000,spi_bus=0,spi_device=0,madctl=192"
        );
    }
}
