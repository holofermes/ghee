use core::convert::TryFrom;

use crate::parser::Stream;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeltaSetIndexMap<'a> {
    data: &'a [u8],
}

impl<'a> DeltaSetIndexMap<'a> {
    #[inline]
    pub(crate) fn new(data: &'a [u8]) -> Self {
        DeltaSetIndexMap { data }
    }

    #[inline]
    pub(crate) fn map(&self, mut index: u32) -> Option<(u16, u16)> {
        let mut s = Stream::new(self.data);
        let format = s.read::<u8>()?;
        let entry_format = s.read::<u8>()?;
        let map_count = if format == 0 {
            s.read::<u16>()? as u32
        } else {
            s.read::<u32>()?
        };

        if map_count == 0 {
            return None;
        }

        // 'If a given glyph ID is greater than mapCount-1, then the last entry is used.'
        if index >= map_count {
            index = map_count - 1;
        }

        let entry_size = ((entry_format >> 4) & 3) + 1;
        let inner_index_bit_count = u32::from((entry_format & 0xF) + 1);

        // `index` is bounded by `map_count`, which is a raw `u32` from the font, so an
        // unchecked advance can push the offset far past the buffer. Reject it here rather
        // than leaving a nonsensical offset for the read below to trip over. ~keep
        s.advance_checked(usize::from(entry_size).checked_mul(usize::try_from(index).ok()?)?)?;

        let mut n = 0u32;
        for b in s.read_bytes(usize::from(entry_size))? {
            n = (n << 8) + u32::from(*b);
        }

        let outer_index = n >> inner_index_bit_count;
        let inner_index = n & ((1 << inner_index_bit_count) - 1);
        Some((
            u16::try_from(outer_index).ok()?,
            u16::try_from(inner_index).ok()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // format 1, entryFormat with entry_size 4 and 4 inner-index bits, then mapCount,
    // then a single 4-byte entry.
    fn format1_map(map_count: u32) -> [u8; 10] {
        let c = map_count.to_be_bytes();
        [1, 0x33, c[0], c[1], c[2], c[3], 0, 0, 0, 7]
    }

    #[test]
    fn maps_an_in_range_index() {
        let data = format1_map(1);
        assert_eq!(DeltaSetIndexMap::new(&data).map(0), Some((0, 7)));
    }

    #[test]
    fn clamps_an_index_past_the_end_to_the_last_entry() {
        let data = format1_map(1);
        assert_eq!(DeltaSetIndexMap::new(&data).map(500), Some((0, 7)));
    }

    // A `mapCount` this large is malformed, but it is a raw `u32` and the index is scaled by
    // the entry size, so the resulting offset is far past the buffer. It used to be applied
    // with an unchecked `advance`, leaving a nonsensical offset for the following read.
    #[test]
    fn rejects_an_index_whose_offset_overflows_the_buffer() {
        let data = format1_map(u32::MAX);
        assert_eq!(DeltaSetIndexMap::new(&data).map(u32::MAX - 1), None);
    }

    #[test]
    fn rejects_an_empty_map() {
        let data = format1_map(0);
        assert_eq!(DeltaSetIndexMap::new(&data).map(0), None);
    }
}
