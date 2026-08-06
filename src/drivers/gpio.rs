//! Offsets are BCM numbers.
//! rppal finds the SoC gpiochip by label.
//! rppal holds one process-wide claim for each pin.
//! A Pins group stands on its own.
//! Put a line that can fail alone, such as an optional backlight, in its own group.

use anyhow::{anyhow, Result};
use rppal::gpio::{Gpio, OutputPin};

/// A group drives these lines. A None pin is not a line.
pub fn line_offsets(offsets: &[Option<u8>]) -> Vec<u8> {
    offsets.iter().flatten().copied().collect()
}

pub struct Pins {
    #[allow(dead_code)]
    pub offsets: Vec<u8>,
    lines: Vec<(u8, OutputPin)>,
}

impl Pins {
    pub fn new(offsets: &[Option<u8>]) -> Result<Pins> {
        let offsets = line_offsets(offsets);
        let gpio = Gpio::new()?;
        let mut lines = Vec::with_capacity(offsets.len());
        for &offset in &offsets {
            // a claim that fails part way drops the lines already taken
            lines.push((offset, gpio.get(offset)?.into_output_low()));
        }
        Ok(Pins { offsets, lines })
    }

    pub fn set(&mut self, offset: u8, high: bool) -> Result<()> {
        let line = self
            .lines
            .iter_mut()
            .find(|(o, _)| *o == offset)
            .ok_or_else(|| anyhow!("GPIO{offset} is not in this group"))?;
        if high {
            line.1.set_high();
        } else {
            line.1.set_low();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_none_pin_is_not_a_line() {
        assert_eq!(line_offsets(&[Some(25), None, Some(22)]), vec![25, 22]);
        assert_eq!(line_offsets(&[None]), Vec::<u8>::new());
    }
}
