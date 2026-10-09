# Outputs and displays

## Where does it run?

ghee renders with GLES2 through DRM/GBM on a stock Raspberry Pi OS Lite image,
and it scans its frames straight out to HDMI.

| Hardware | OS | Driver | Hardware H264 |
| --- | --- | --- | --- |
| Pi 1, Pi Zero, Pi Zero W (armv6) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 2, Pi 3, Pi Zero 2 W (arm64) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 4, Pi 400 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | yes |
| Pi 5 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | ffmpeg/libx264 |
| Desktop Linux (x86_64) | any current distro | Mesa | ffmpeg/libx264 |

ghee runs on every Pi back to the Pi 1, and builds ship for all of them. Pi 1
through Pi 4 encode H264 on the Broadcom block. The block needs a frame of
16384 pixels or more, so ghee scales a smaller frame up by the smallest whole
factor that reaches that size, each pixel a square block: a 128x32 panel
streams at 256x64. The Pi 5 has no encoder, so there and on desktop ghee
encodes with ffmpeg/libx264. Install it if needed:

```
sudo apt install ffmpeg
```

## Outputs

Render one of the Python [procedural scenes](scenes.md#procedural-scenes) to
the connected HDMI display, and hand it an argument of its own after `--`:

```
ghee render --demo bluesun --display kms -- --emit 480x300
```

Write raw frames to stdout and pipe them through ffmpeg into an mp4, with
`--timestep` pinning engine time to a fixed 1/60 s step per frame:

```
ghee render --demo bars -o - --frames 300 --timestep 0.0166667 |
  ffmpeg -f rawvideo -pix_fmt rgba -s 320x240 -framerate 60 -i - out.mp4
```

Or stream it as a hardware accelerated H264 Annex-B stream, and pipe it to
mpv:

```
ghee render --demo trail --h264 - --fps 30 | mpv -
```

Or scan it straight out to the connected display, HDMI or an SPI or I2C
panel, as [Hardware displays](#hardware-displays) describes:

```
ghee render --demo trail --display kms
```

You can combine outputs: one render can write PNGs, drive a panel and stream
H264 at the same time, and `--serve` attaches the console to preview and
author the scene:

```
ghee render --demo bars --display kms -o frame.####.png          # panel and a contact sheet
ghee render --demo bars --display kms --serve --host 0.0.0.0     # panel, edit it from a laptop
```

Everything except hardware H264 and KMS scanout reads the frame back with a
blocking glReadPixels (GLES2 has no async readback), so bigger outputs cost
more CPU.

## Hardware displays

```
ghee displays                         # the cards with a connected display
ghee render scene.json --display kms
ghee serve scenes/ --display kms      # the console, driving the display
ghee overlay st7789                   # hand an SPI panel to the kernel
```

`kms` is any connected KMS display: HDMI, or an SPI/I2C panel with a kernel
driver. `ghee displays` lists the cards, and `--dri` picks one.

`--rotate` turns the frame clockwise on the GPU, by 90, 180 or 270 degrees,
on its way to every output: the display, the files, the raw frames and the
console's video. At 90 and 270 the scene renders at the output's size turned,
so a 128x32 panel gives the scene 32x128. `--brightness` sets a KMS display's
backlight from 0 to 1, through the backlight device that the kernel
registered for it. When the display has no backlight device, or its device
refuses the write, ghee prints the error once and renders on. ghee puts the
earlier level back when it exits, and a crash leaves the set level on the
display. On a panel under the kernel's ssd130x driver, the backlight
sets the panel's contrast.

Show the `stats` demo on a panel mounted upside down, its backlight at 0.5:

```
ghee render --demo stats --display kms --rotate 180 --brightness 0.5
```

The kernel drives a range of SPI and I2C panels, each set up with a
device-tree overlay. `ghee overlay` writes what an ST7789, an ILI9341, an
SSD1306, an SH1106 or an SSD1322 needs and says how to install it, and the
panel then answers `--display kms`. Careful: the ILI9341, SH1106 and SSD1322
setups follow the kernel's drivers and bindings, and are untested on a panel.

`ghee overlay st7789` needs the panel's `width` and `height`. `madctl` sets
its rotation and mirroring, 0xC0 by default, `x_offset` and `y_offset` place
it in the controller's memory, 0 by default, `dc`, `rst` and `backlight`
take its GPIOs, 25, none and 22 by default, and `speed` its SPI clock in Hz,
62500000 by default.

Write the setup of the Adafruit 1.3" TFT Bonnet, a 240x240 ST7789:

```
$ ghee overlay st7789 --opts width=240,height=240,madctl=0x00,backlight=26
dtoverlay=mipi-dbi-spi,spi0-0,speed=62500000
dtparam=width=240,height=240
dtparam=dc-gpio=25,backlight-gpio=26
dtparam=write-only
wrote panel.bin. Copy it to /lib/firmware/panel.bin, add the lines above to
/boot/firmware/config.txt, and reboot
```

`ghee overlay ili9341` takes the same options, and its firmware replays the
init of the kernel's own ili9341 driver. The panel is 320x240 by default, its
`madctl` 0x28 when it is wider than tall and 0x48 otherwise, its `speed`
32000000, and `dc`, `rst` and `backlight` are 25, 27 and 18 by default, the
pins of Adafruit's EYESPI Pi Beret. An IPS panel shows its colours inverted
under that init, and `invert=True` turns them back.

`ghee overlay ssd1306` needs the panel's `height`, 32 or 64. `address` takes
the panel's I2C address, 0x3C by default, and `speed` the bus rate in Hz,
1000000 by default. The rate line sets the whole of I2C bus 1, every device
on it included. On a Pi 5, a 128x32 SSD1306 reaches about 216 fps at 1 MHz
and about 86 at 400 kHz, and above 1 MHz the Pi 5's I2C controller slows it
to about 43.

Write the setup of a 128x32 SSD1306 at the default address:

```
$ ghee overlay ssd1306 --opts height=32
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=1000000
dtoverlay=ssd1306,height=32,inverted,sequential
add the lines above to /boot/firmware/config.txt, and reboot
```

`ghee overlay sh1106` sets up a 1.3" 128x64 SH1106 on I2C bus 1, and `ghee
overlay sh1106-spi` one on SPI bus 0. The Pi ships no overlay that fits this
panel to the kernel's ssd130x driver, so each writes one of its own,
`ghee-sh1106.dts` or `ghee-sh1106-spi.dts`, and dtc on the Pi compiles it into
/boot/firmware/overlays. `sh1106` takes `address`, 0x3C by default, and
`speed`, 400000 by default, the SH1106's I2C fast mode. `sh1106-spi` takes
`spi_device`, the chip select, 0 by default, `speed`, 4000000 by default, and
`dc` and `rst`, 24 and 25 by default, the pins of the Pi's own SPI OLED
overlays and of Waveshare's 1.3inch OLED HAT.

Write the setup of an SH1106 on I2C at the default address:

```
$ ghee overlay sh1106
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=400000
dtoverlay=ghee-sh1106
wrote ghee-sh1106.dts. Compile it with `sudo dtc -@ -I dts -O dtb -o
/boot/firmware/overlays/ghee-sh1106.dtbo ghee-sh1106.dts`, add the lines
above to /boot/firmware/config.txt, and reboot
```

Careful: up to Linux 7.3, and in the Pi's 6.18 kernel, the ssd130x driver
leaves an SSD1322 dark. `ghee overlay ssd1322` writes its overlay,
`ghee-ssd1322.dts`, for a kernel whose ssd130x drives it. The overlay fits a
256x64 module by default, its glass from `col_offset` 28, counted in column
addresses of 4 pixels, and `width`, `height` and `col_offset` fit another.
`spi_device`, `dc` and `rst` take the defaults of `sh1106-spi`, and `speed` is
10000000, the SSD1322's datasheet clock. On a Pi 5 a 256x64 SSD1322 draws
about 145 fps at 10 MHz, and 206 at 16 MHz on short wires. The overlay divides
the panel's display clock by 1, so the glass redraws as fast as `frc256`
needs.
