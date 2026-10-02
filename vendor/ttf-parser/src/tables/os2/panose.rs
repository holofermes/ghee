//! A [PANOSE classification](https://monotype.github.io/panose/pan1.htm) implementation.

use crate::parser::FromData;

/// The number of PANOSE digits. Fixed by the spec.
const DIGITS: usize = 10;

// Digit positions. Only the first is family-independent; the meaning of every other
// position is determined by it, which is why the accessors below dispatch on the
// family type before indexing. ~keep
const FAMILY_TYPE: usize = 0;
const WEIGHT: usize = 2;
/// Proportion for Latin Text, spacing for Latin Handwritten and Latin Symbol,
/// aspect ratio for Latin Decorative. All four encode "monospaced" at this position.
const PROPORTION: usize = 3;
/// Letterform for Latin Text, hand form for Latin Handwritten.
const LETTERFORM: usize = 7;

const FAMILY_ANY: u8 = 0;
const FAMILY_NO_FIT: u8 = 1;
const FAMILY_LATIN_TEXT: u8 = 2;
const FAMILY_LATIN_HANDWRITTEN: u8 = 3;
const FAMILY_LATIN_DECORATIVE: u8 = 4;
const FAMILY_LATIN_SYMBOL: u8 = 5;

const WEIGHT_BOLD: u8 = 8;

/// Oblique letterform values for Latin Text, which run to the end of the range.
const TEXT_OBLIQUE_FIRST: u8 = 9;
/// Oblique hand form values for Latin Handwritten. Unlike Latin Text this is not an
/// open-ended range: 10 and above are the "exaggerated" upright forms. ~keep
const HANDWRITTEN_OBLIQUE_FIRST: u8 = 6;
const HANDWRITTEN_OBLIQUE_LAST: u8 = 9;

const TEXT_MONOSPACED: u8 = 9;
const DECORATIVE_MONOSPACED: u8 = 9;
const SPACING_MONOSPACED: u8 = 3;

/// A PANOSE family type, the first of the ten PANOSE digits.
///
/// The family type determines how the remaining nine digits are interpreted; the same
/// digit position means different things for different families.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum FamilyType {
    Any,
    NoFit,
    LatinText,
    LatinHandwritten,
    LatinDecorative,
    LatinSymbol,
    /// A value not defined by PANOSE 2.0.
    Other(u8),
}

/// A [PANOSE classification](https://monotype.github.io/panose/pan1.htm).
///
/// Ten bytes describing a face's visual characteristics, stored in the `OS/2` table.
///
/// The raw digits are public, because PANOSE has more digit values than this crate models
/// and their meaning depends on [`FamilyType`]. Only the classifications that map onto
/// something this crate already exposes have accessors.
///
/// PANOSE is largely of historical interest and real-world values are frequently
/// zero-filled or simply wrong, so treat it as a hint rather than a source of truth.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Panose(pub [u8; DIGITS]);

impl Panose {
    /// Returns the family type, which determines how the remaining digits are read.
    #[inline]
    pub fn family_type(&self) -> FamilyType {
        match self.0[FAMILY_TYPE] {
            FAMILY_ANY => FamilyType::Any,
            FAMILY_NO_FIT => FamilyType::NoFit,
            FAMILY_LATIN_TEXT => FamilyType::LatinText,
            FAMILY_LATIN_HANDWRITTEN => FamilyType::LatinHandwritten,
            FAMILY_LATIN_DECORATIVE => FamilyType::LatinDecorative,
            FAMILY_LATIN_SYMBOL => FamilyType::LatinSymbol,
            n => FamilyType::Other(n),
        }
    }

    /// Checks if the weight digit is set to *Bold*.
    ///
    /// Returns `false` for the `Any` and `NoFit` families, whose remaining digits
    /// carry no meaning.
    #[inline]
    pub fn is_bold(&self) -> bool {
        match self.family_type() {
            FamilyType::LatinText
            | FamilyType::LatinHandwritten
            | FamilyType::LatinDecorative
            | FamilyType::LatinSymbol => self.0[WEIGHT] == WEIGHT_BOLD,
            _ => false,
        }
    }

    /// Checks if the letterform digit is set to one of the oblique values.
    ///
    /// Only Latin Text and Latin Handwritten classify letterform; the other families
    /// return `false`.
    #[inline]
    pub fn is_italic(&self) -> bool {
        let letterform = self.0[LETTERFORM];
        match self.family_type() {
            FamilyType::LatinText => letterform >= TEXT_OBLIQUE_FIRST,
            FamilyType::LatinHandwritten => {
                (HANDWRITTEN_OBLIQUE_FIRST..=HANDWRITTEN_OBLIQUE_LAST).contains(&letterform)
            }
            _ => false,
        }
    }

    /// Checks if the proportion digit is set to *Monospaced*.
    #[inline]
    pub fn is_monospaced(&self) -> bool {
        let proportion = self.0[PROPORTION];
        match self.family_type() {
            FamilyType::LatinText => proportion == TEXT_MONOSPACED,
            FamilyType::LatinDecorative => proportion == DECORATIVE_MONOSPACED,
            FamilyType::LatinHandwritten | FamilyType::LatinSymbol => {
                proportion == SPACING_MONOSPACED
            }
            _ => false,
        }
    }
}

impl FromData for Panose {
    const SIZE: usize = DIGITS;

    // Every 10-byte sequence is accepted. Out-of-range digits are preserved rather than
    // rejected: PANOSE fields in real fonts are routinely junk, and discarding the whole
    // classification because one digit is unknown would also discard the digits that are
    // fine. Callers see the raw value and can decide. ~keep
    #[inline]
    fn parse(data: &[u8]) -> Option<Self> {
        let digits: [u8; DIGITS] = data.get(0..DIGITS)?.try_into().ok()?;
        Some(Panose(digits))
    }
}
