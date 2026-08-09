//! Decode an asset file's bytes into pixels for the engine.
//!
//! scene.rs resolves and reads the file. This module turns the bytes into
//! (width, height, components, pixels), and encoding stays hand-rolled in
//! output.rs.

use anyhow::{bail, Result};

/// One decoded asset: tightly packed rows of `components` values per pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub components: u32,
    pub pixels: Pixels,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

/// Bare pixels at the declared layout. f32 is little-endian.
pub fn decode_raw(
    bytes: &[u8],
    width: u32,
    height: u32,
    components: u32,
    is_f32: bool,
) -> Result<Image> {
    let values = width as usize * height as usize * components as usize;
    let want = values * if is_f32 { 4 } else { 1 };
    if bytes.len() != want {
        bail!(
            "{} bytes, but {width}x{height}x{components} {} needs {want}",
            bytes.len(),
            if is_f32 { "f32" } else { "u8" }
        );
    }
    let pixels = if is_f32 {
        Pixels::F32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    } else {
        Pixels::U8(bytes.to_vec())
    };
    Ok(Image {
        width,
        height,
        components,
        pixels,
    })
}

/// An 8-bit PNG. A palette expands to its colours. Gray plus alpha widens to
/// RGBA, because the engine has no two-channel upload. A 16-bit file is
/// refused.
pub fn decode_png(bytes: &[u8]) -> Result<Image> {
    let mut decoder = png::Decoder::new(bytes);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    // EXPAND promotes any depth under 8, so only 16 remains.
    if info.bit_depth != png::BitDepth::Eight {
        bail!("16-bit png, ghee reads 8-bit only");
    }
    buf.truncate(info.buffer_size());
    let (width, height) = (info.width, info.height);
    let (components, pixels) = match info.color_type {
        png::ColorType::Grayscale => (1, Pixels::U8(buf)),
        png::ColorType::Rgb => (3, Pixels::U8(buf)),
        png::ColorType::Rgba => (4, Pixels::U8(buf)),
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(buf.len() * 2);
            for ga in buf.chunks_exact(2) {
                out.extend_from_slice(&[ga[0], ga[0], ga[0], ga[1]]);
            }
            (4, Pixels::U8(out))
        }
        png::ColorType::Indexed => bail!("indexed png did not expand"),
    };
    Ok(Image {
        width,
        height,
        components,
        pixels,
    })
}

/// A baseline JPEG: grayscale or RGB.
pub fn decode_jpeg(bytes: &[u8]) -> Result<Image> {
    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    let data = decoder.decode()?;
    let info = match decoder.info() {
        Some(i) => i,
        None => bail!("no jpeg frame header"),
    };
    let components = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => 1,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::L16 => bail!("16-bit jpeg, ghee reads 8-bit only"),
        jpeg_decoder::PixelFormat::CMYK32 => bail!("cmyk jpeg, ghee reads grayscale or rgb"),
    };
    Ok(Image {
        width: u32::from(info.width),
        height: u32::from(info.height),
        components,
        pixels: Pixels::U8(data),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::looper::Frame;
    use crate::output::{encode_jpeg, encode_png};

    fn frame(data: &[u8], width: i32, height: i32, components: i32) -> Frame<'_> {
        Frame {
            data,
            width,
            height,
            components,
        }
    }

    #[test]
    fn an_rgb_png_round_trips_through_the_project_encoder() {
        let data: Vec<u8> = (0..2 * 2 * 3).map(|i| i * 20).collect();
        let png = encode_png(frame(&data, 2, 2, 3)).unwrap();
        let img = decode_png(&png).unwrap();
        assert_eq!((img.width, img.height, img.components), (2, 2, 3));
        assert_eq!(img.pixels, Pixels::U8(data));
    }

    #[test]
    fn an_rgba_png_round_trips_through_the_project_encoder() {
        let data: Vec<u8> = (0..2 * 2 * 4).map(|i| i * 15).collect();
        let png = encode_png(frame(&data, 2, 2, 4)).unwrap();
        let img = decode_png(&png).unwrap();
        assert_eq!((img.width, img.height, img.components), (2, 2, 4));
        assert_eq!(img.pixels, Pixels::U8(data));
    }

    /// output.rs writes RGB and RGBA only, so the other shapes come from the
    /// png crate's own encoder.
    fn write_png(color: png::ColorType, depth: png::BitDepth, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, 2, 2);
        enc.set_color(color);
        enc.set_depth(depth);
        if color == png::ColorType::Indexed {
            enc.set_palette(vec![255, 0, 0, 0, 255, 0]);
        }
        enc.write_header().unwrap().write_image_data(data).unwrap();
        out
    }

    #[test]
    fn a_gray_png_decodes_to_one_component() {
        let png = write_png(
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            &[0, 85, 170, 255],
        );
        let img = decode_png(&png).unwrap();
        assert_eq!((img.width, img.height, img.components), (2, 2, 1));
        assert_eq!(img.pixels, Pixels::U8(vec![0, 85, 170, 255]));
    }

    #[test]
    fn a_gray_alpha_png_widens_to_rgba() {
        let png = write_png(
            png::ColorType::GrayscaleAlpha,
            png::BitDepth::Eight,
            &[10, 255, 20, 128, 30, 0, 40, 64],
        );
        let img = decode_png(&png).unwrap();
        assert_eq!(img.components, 4);
        assert_eq!(
            img.pixels,
            Pixels::U8(vec![
                10, 10, 10, 255, 20, 20, 20, 128, 30, 30, 30, 0, 40, 40, 40, 64,
            ])
        );
    }

    #[test]
    fn an_indexed_png_expands_to_its_colours() {
        let png = write_png(png::ColorType::Indexed, png::BitDepth::Eight, &[0, 1, 1, 0]);
        let img = decode_png(&png).unwrap();
        assert_eq!(img.components, 3);
        assert_eq!(
            img.pixels,
            Pixels::U8(vec![255, 0, 0, 0, 255, 0, 0, 255, 0, 255, 0, 0])
        );
    }

    #[test]
    fn a_sixteen_bit_png_is_refused() {
        let png = write_png(
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            &[0, 0, 0, 85, 0, 170, 0, 255],
        );
        let e = decode_png(&png).unwrap_err().to_string();
        assert!(e.contains("16-bit"), "{e}");
    }

    #[test]
    fn a_jpeg_round_trips_through_the_project_encoder() {
        let data = vec![100u8; 8 * 8 * 3];
        let jpeg = encode_jpeg(frame(&data, 8, 8, 3), 100).unwrap();
        let img = decode_jpeg(&jpeg).unwrap();
        assert_eq!((img.width, img.height, img.components), (8, 8, 3));
        let Pixels::U8(px) = img.pixels else {
            panic!("jpeg decoded to floats");
        };
        assert_eq!(px.len(), data.len());
        // JPEG is lossy, so a flat frame comes back near its value.
        assert!(px.iter().all(|&v| (i32::from(v) - 100).abs() <= 3));
    }
}
