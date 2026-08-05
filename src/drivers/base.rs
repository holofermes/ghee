//! A driver is an Output bound to a panel.
//! The driver owns the panel geometry and the native pixel format.
//! The driver packs each RGB frame before it writes the frame.

use anyhow::Result;

use crate::stats::DEST_PANEL;

#[allow(dead_code)]
pub const RGB565: &str = "rgb565";
pub const RGB444: &str = "rgb444";
#[allow(dead_code)]
pub const MONO1: &str = "mono1";

/// The first and last rows that differ. None means the frames are identical.
/// A partial-update driver sends only that row span.
pub(crate) fn rows_changed(
    prev: &[u8],
    next: &[u8],
    bytes_per_row: usize,
) -> Option<(usize, usize)> {
    let rows = next.len() / bytes_per_row;
    let differs = |r: usize| {
        prev[r * bytes_per_row..(r + 1) * bytes_per_row]
            != next[r * bytes_per_row..(r + 1) * bytes_per_row]
    };
    let first = (0..rows).find(|r| differs(*r))?;
    let last = (first..rows).rev().find(|r| differs(*r)).unwrap_or(first);
    Some((first, last))
}

pub trait DisplayDriver {
    fn dest(&self) -> &'static str {
        DEST_PANEL
    }

    /// The panel holds its last frame with no bus traffic.
    fn persists(&self) -> bool {
        true
    }

    fn width(&self) -> i32 {
        0
    }

    fn height(&self) -> i32 {
        0
    }

    fn components(&self) -> i32 {
        3
    }

    /// RGB565, RGB444 or MONO1.
    #[allow(dead_code)]
    fn native_format(&self) -> &'static str {
        RGB565
    }

    #[allow(dead_code)]
    fn bw(&self) -> bool {
        self.native_format() == MONO1
    }

    fn write(&mut self, frame: &[u8]) -> Result<()>;

    fn clear(&mut self) -> Result<()> {
        Ok(())
    }

    fn turn_on(&mut self) -> Result<()> {
        Ok(())
    }

    fn turn_off(&mut self) -> Result<()> {
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Panel(&'static str);

    impl DisplayDriver for Panel {
        fn native_format(&self) -> &'static str {
            self.0
        }

        fn write(&mut self, _frame: &[u8]) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn bw_follows_the_native_format() {
        assert!(Panel(MONO1).bw());
        assert!(!Panel(RGB565).bw());
    }

    #[test]
    fn a_panel_is_a_persistent_panel_destination() {
        let p = Panel(RGB565);
        assert_eq!(p.dest(), DEST_PANEL);
        assert!(p.persists());
        assert_eq!((p.width(), p.height(), p.components()), (0, 0, 3));
    }

    #[test]
    fn changed_rows_bound_the_partial_window() {
        let a = vec![0u8; 8 * 4]; // 8 rows, 4 bytes each
        let mut b = a.clone();
        assert_eq!(rows_changed(&a, &b, 4), None);
        b[2 * 4] = 1;
        assert_eq!(rows_changed(&a, &b, 4), Some((2, 2)));
        b[6 * 4 + 3] = 9;
        assert_eq!(rows_changed(&a, &b, 4), Some((2, 6)));
    }
}
