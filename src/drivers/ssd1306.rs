//! The driver packs each RGB frame to 1bpp in the display buffer.
//! The panel wants the 0x40 control byte in front of display RAM.
//! The buffer holds that byte first.
//! One transfer sends the whole buffer.

use anyhow::{bail, Result};
use rppal::i2c::I2c;

use crate::drivers::base::{DisplayDriver, MONO1};
use crate::drivers::{given, int_opt, DriverOption, DriverSpec, Opts, Setup};
use crate::engine::pack_frame_into;

const CMD: u8 = 0x80; // control byte: one command follows
const DATA: u8 = 0x40; // control byte: the rest of the transfer is display RAM

const DISP_OFF: u8 = 0xAE;
const DISP_ON: u8 = 0xAF;
const COL_ADDR: u8 = 0x21;
const PAGE_ADDR: u8 = 0x22;

const WIDTH: i32 = 128;
const HEIGHT: i32 = 32;

const DEFAULT_BUS: i64 = 1;
const DEFAULT_ADDR: i64 = 0x3C;
/// The I2C bus rate in Hz that `ghee overlay` sets. A 128x32 frame takes
/// 4.7 ms at 1 MHz. Above 1 MHz a Pi 5's controller falls back to a far
/// slower rate.
const DEFAULT_SPEED: i64 = 1_000_000;

const INIT: [&[u8]; 18] = [
    &[DISP_ON], // on, then off again on the next command
    &[DISP_OFF],
    &[0x20, 0x00], // horizontal addressing
    &[0x40],       // start line 0
    &[0xA1],       // column 127 maps to SEG0
    &[0xA8, 0x1F], // mux ratio, height - 1
    &[0xC8],       // scan COM[N] to COM[0]
    &[0xD3, 0x00], // no display offset
    &[0xDA, 0x02], // COM pin config, 128x32 layout
    &[0xD5, 0x80], // clock divide
    &[0xD9, 0xF1], // precharge
    &[0xDB, 0x30], // VCOMH deselect
    &[0x81, 0xFF], // contrast
    &[0xA4],       // follow RAM (0xA5 lights every pixel)
    &[0xA6],       // normal display (0xA7 inverts)
    &[0xAD, 0x30], // internal IREF
    &[0x8D, 0x14], // charge pump on
    &[DISP_ON],
];

/// The addressing window for a frame.
fn window_commands(width: i32, height: i32) -> Result<[u8; 6]> {
    let (Ok(last_col), Ok(last_page)) = (u8::try_from(width - 1), u8::try_from(height / 8 - 1))
    else {
        bail!("{width}x{height} does not address on this panel");
    };
    Ok([COL_ADDR, 0, last_col, PAGE_ADDR, 0, last_page])
}

/// The transfer buffer: the data control byte, then the 1bpp pixels.
fn frame_buffer(width: i32, height: i32) -> Vec<u8> {
    let mut buf = vec![0u8; 1 + (width as usize) * (height as usize) / 8];
    buf[0] = DATA;
    buf
}

pub struct Ssd1306 {
    bus: I2c,
    buf: Vec<u8>,
}

impl Ssd1306 {
    pub fn new(bus: u8, addr: u16) -> Result<Ssd1306> {
        let mut bus = I2c::with_bus(bus)?;
        bus.set_slave_address(addr)?;
        let mut panel = Ssd1306 {
            bus,
            buf: frame_buffer(WIDTH, HEIGHT),
        };
        for entry in INIT {
            for cmd in entry {
                panel.cmd(*cmd)?;
            }
        }
        panel.show()?;
        Ok(panel)
    }

    /// A command goes out as a plain two-byte write, not an SMBus transaction:
    /// the bytes on the wire are the same, and the SMBus path reaches this
    /// panel on some boards and not others.
    fn cmd(&mut self, cmd: u8) -> Result<()> {
        let sent = self.bus.write(&[CMD, cmd])?;
        if sent != 2 {
            bail!("I2C took {sent} of 2 bytes");
        }
        Ok(())
    }

    fn show(&mut self) -> Result<()> {
        for cmd in window_commands(WIDTH, HEIGHT)? {
            self.cmd(cmd)?;
        }
        let sent = self.bus.write(&self.buf)?;
        if sent != self.buf.len() {
            bail!("I2C took {sent} of {} bytes", self.buf.len());
        }
        Ok(())
    }
}

impl DisplayDriver for Ssd1306 {
    fn width(&self) -> i32 {
        WIDTH
    }

    fn height(&self) -> i32 {
        HEIGHT
    }

    fn native_format(&self) -> &'static str {
        MONO1
    }

    fn write(&mut self, frame: &[u8]) -> Result<()> {
        pack_frame_into(frame, WIDTH, HEIGHT, 3, true, &mut self.buf[1..]);
        self.show()
    }

    fn clear(&mut self) -> Result<()> {
        self.buf[1..].fill(0);
        self.show()
    }

    fn turn_on(&mut self) -> Result<()> {
        self.cmd(DISP_ON)
    }

    fn turn_off(&mut self) -> Result<()> {
        self.cmd(DISP_OFF)
    }
}

const OPTIONS: [DriverOption; 5] = [
    DriverOption::own("bus", Some(DEFAULT_BUS)),
    DriverOption::own("addr", Some(DEFAULT_ADDR)),
    DriverOption::kernel("height", None),
    DriverOption::kernel("address", Some(DEFAULT_ADDR)),
    DriverOption::kernel("speed", Some(DEFAULT_SPEED)),
];

fn open(opts: &Opts) -> Result<Box<dyn DisplayDriver + Send>> {
    let bus = int_opt(opts, "bus", DEFAULT_BUS)?;
    let Ok(bus) = u8::try_from(bus) else {
        bail!("no I2C bus {bus}");
    };
    let addr = int_opt(opts, "addr", DEFAULT_ADDR)?;
    let Ok(addr) = u16::try_from(addr) else {
        bail!("{addr} is not an I2C address");
    };
    Ok(Box::new(Ssd1306::new(bus, addr)?))
}

/// The config.txt lines that hand this panel to ssd130x on I2C bus 1. That
/// driver holds the init sequence for the whole Solomon family, so no
/// firmware file goes with it. The overlay's `address` sets the panel's I2C
/// address. A 128x32 module wires its rows `inverted,sequential`, and a
/// 128x64 module takes the overlay's own wiring. The bus rate line sets
/// every device on the bus.
fn kernel(opts: &Opts) -> Result<Setup> {
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
    let address = int_opt(opts, "address", DEFAULT_ADDR)?;
    if !(0..0x80).contains(&address) {
        bail!("address {address} is not a 7-bit I2C address");
    }
    let speed = int_opt(opts, "speed", DEFAULT_SPEED)?;
    if speed <= 0 {
        bail!("speed {speed} is not a bus rate in Hz");
    }
    // the overlay's node sits at the default address
    let address = match address {
        DEFAULT_ADDR => String::new(),
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

pub const SPEC: DriverSpec = DriverSpec {
    name: "ssd1306",
    options: &OPTIONS,
    open,
    size: |_| Ok((WIDTH, HEIGHT)),
    kernel,
};

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::engine::pack_frame_into;

    fn opts(set: Value) -> Vec<(String, Value)> {
        set.as_object().unwrap().clone().into_iter().collect()
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
            let setup = kernel(&opts(set.clone())).unwrap();
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
                kernel(&opts(set.clone())).unwrap_err().to_string(),
                msg,
                "{set}"
            );
        }
    }

    /// adafruit_ssd1306's command bytes for a 128x32 panel on internal Vcc.
    /// The order is poweron(), the init table, then the addressing window.
    const INIT_COMMANDS: [u8; 28] = [
        0xAF, 0xAE, 0x20, 0x00, 0x40, 0xA1, 0xA8, 0x1F, 0xC8, 0xD3, 0x00, 0xDA, 0x02, 0xD5, 0x80,
        0xD9, 0xF1, 0xDB, 0x30, 0x81, 0xFF, 0xA4, 0xA6, 0xAD, 0x30, 0x8D, 0x14, 0xAF,
    ];
    const WINDOW_COMMANDS: [u8; 6] = [0x21, 0, 127, 0x22, 0, 3];

    #[test]
    fn init_is_the_recorded_command_stream() {
        let sent: Vec<u8> = INIT.iter().flat_map(|e| e.iter().copied()).collect();
        assert_eq!(sent, INIT_COMMANDS);
        assert_eq!((WIDTH, HEIGHT), (128, 32));
    }

    #[test]
    fn a_frame_addresses_every_column_and_page() {
        assert_eq!(window_commands(WIDTH, HEIGHT).unwrap(), WINDOW_COMMANDS);
    }

    #[test]
    fn the_transfer_carries_the_data_control_byte_in_front() {
        let buf = frame_buffer(WIDTH, HEIGHT);
        assert_eq!(buf.len(), 1 + 128 * 32 / 8);
        assert_eq!(buf[0], DATA);
        assert!(buf[1..].iter().all(|b| *b == 0)); // the panel comes up cleared
    }

    #[test]
    fn a_frame_lands_after_the_control_byte() {
        let mut buf = frame_buffer(WIDTH, HEIGHT);
        let frame = vec![255u8; (WIDTH * HEIGHT * 3) as usize];
        pack_frame_into(&frame, WIDTH, HEIGHT, 3, true, &mut buf[1..]);
        assert_eq!(buf[0], DATA);
        assert!(buf[1..].iter().all(|b| *b == 0xFF));
    }

    #[test]
    fn every_constructor_option_is_listed_with_its_default() {
        let listed: Vec<String> = crate::drivers::driver_options(&SPEC)
            .iter()
            .map(|o| match o.default {
                Some(v) => format!("{}={v}", o.name),
                None => format!("{}=None", o.name),
            })
            .collect();
        assert_eq!(listed.join(","), "bus=1,addr=60");
    }
}
