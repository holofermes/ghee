//! CSS's colour names: the 148 named colors of CSS Color 4, in sRGB, and
//! `transparent`, transparent black. A scene writes a name in any case, and
//! the name keeps that case in the scene. Also CSS's hex colours.

/// The longest name, `lightgoldenrodyellow`.
const LONGEST: usize = 20;

/// The named colors of CSS Color 4's table, in its order, each with its red,
/// green and blue.
const NAMED: [(&str, [u8; 3]); 148] = [
    ("aliceblue", [0xf0, 0xf8, 0xff]),
    ("antiquewhite", [0xfa, 0xeb, 0xd7]),
    ("aqua", [0x00, 0xff, 0xff]),
    ("aquamarine", [0x7f, 0xff, 0xd4]),
    ("azure", [0xf0, 0xff, 0xff]),
    ("beige", [0xf5, 0xf5, 0xdc]),
    ("bisque", [0xff, 0xe4, 0xc4]),
    ("black", [0x00, 0x00, 0x00]),
    ("blanchedalmond", [0xff, 0xeb, 0xcd]),
    ("blue", [0x00, 0x00, 0xff]),
    ("blueviolet", [0x8a, 0x2b, 0xe2]),
    ("brown", [0xa5, 0x2a, 0x2a]),
    ("burlywood", [0xde, 0xb8, 0x87]),
    ("cadetblue", [0x5f, 0x9e, 0xa0]),
    ("chartreuse", [0x7f, 0xff, 0x00]),
    ("chocolate", [0xd2, 0x69, 0x1e]),
    ("coral", [0xff, 0x7f, 0x50]),
    ("cornflowerblue", [0x64, 0x95, 0xed]),
    ("cornsilk", [0xff, 0xf8, 0xdc]),
    ("crimson", [0xdc, 0x14, 0x3c]),
    ("cyan", [0x00, 0xff, 0xff]),
    ("darkblue", [0x00, 0x00, 0x8b]),
    ("darkcyan", [0x00, 0x8b, 0x8b]),
    ("darkgoldenrod", [0xb8, 0x86, 0x0b]),
    ("darkgray", [0xa9, 0xa9, 0xa9]),
    ("darkgreen", [0x00, 0x64, 0x00]),
    ("darkgrey", [0xa9, 0xa9, 0xa9]),
    ("darkkhaki", [0xbd, 0xb7, 0x6b]),
    ("darkmagenta", [0x8b, 0x00, 0x8b]),
    ("darkolivegreen", [0x55, 0x6b, 0x2f]),
    ("darkorange", [0xff, 0x8c, 0x00]),
    ("darkorchid", [0x99, 0x32, 0xcc]),
    ("darkred", [0x8b, 0x00, 0x00]),
    ("darksalmon", [0xe9, 0x96, 0x7a]),
    ("darkseagreen", [0x8f, 0xbc, 0x8f]),
    ("darkslateblue", [0x48, 0x3d, 0x8b]),
    ("darkslategray", [0x2f, 0x4f, 0x4f]),
    ("darkslategrey", [0x2f, 0x4f, 0x4f]),
    ("darkturquoise", [0x00, 0xce, 0xd1]),
    ("darkviolet", [0x94, 0x00, 0xd3]),
    ("deeppink", [0xff, 0x14, 0x93]),
    ("deepskyblue", [0x00, 0xbf, 0xff]),
    ("dimgray", [0x69, 0x69, 0x69]),
    ("dimgrey", [0x69, 0x69, 0x69]),
    ("dodgerblue", [0x1e, 0x90, 0xff]),
    ("firebrick", [0xb2, 0x22, 0x22]),
    ("floralwhite", [0xff, 0xfa, 0xf0]),
    ("forestgreen", [0x22, 0x8b, 0x22]),
    ("fuchsia", [0xff, 0x00, 0xff]),
    ("gainsboro", [0xdc, 0xdc, 0xdc]),
    ("ghostwhite", [0xf8, 0xf8, 0xff]),
    ("gold", [0xff, 0xd7, 0x00]),
    ("goldenrod", [0xda, 0xa5, 0x20]),
    ("gray", [0x80, 0x80, 0x80]),
    ("green", [0x00, 0x80, 0x00]),
    ("greenyellow", [0xad, 0xff, 0x2f]),
    ("grey", [0x80, 0x80, 0x80]),
    ("honeydew", [0xf0, 0xff, 0xf0]),
    ("hotpink", [0xff, 0x69, 0xb4]),
    ("indianred", [0xcd, 0x5c, 0x5c]),
    ("indigo", [0x4b, 0x00, 0x82]),
    ("ivory", [0xff, 0xff, 0xf0]),
    ("khaki", [0xf0, 0xe6, 0x8c]),
    ("lavender", [0xe6, 0xe6, 0xfa]),
    ("lavenderblush", [0xff, 0xf0, 0xf5]),
    ("lawngreen", [0x7c, 0xfc, 0x00]),
    ("lemonchiffon", [0xff, 0xfa, 0xcd]),
    ("lightblue", [0xad, 0xd8, 0xe6]),
    ("lightcoral", [0xf0, 0x80, 0x80]),
    ("lightcyan", [0xe0, 0xff, 0xff]),
    ("lightgoldenrodyellow", [0xfa, 0xfa, 0xd2]),
    ("lightgray", [0xd3, 0xd3, 0xd3]),
    ("lightgreen", [0x90, 0xee, 0x90]),
    ("lightgrey", [0xd3, 0xd3, 0xd3]),
    ("lightpink", [0xff, 0xb6, 0xc1]),
    ("lightsalmon", [0xff, 0xa0, 0x7a]),
    ("lightseagreen", [0x20, 0xb2, 0xaa]),
    ("lightskyblue", [0x87, 0xce, 0xfa]),
    ("lightslategray", [0x77, 0x88, 0x99]),
    ("lightslategrey", [0x77, 0x88, 0x99]),
    ("lightsteelblue", [0xb0, 0xc4, 0xde]),
    ("lightyellow", [0xff, 0xff, 0xe0]),
    ("lime", [0x00, 0xff, 0x00]),
    ("limegreen", [0x32, 0xcd, 0x32]),
    ("linen", [0xfa, 0xf0, 0xe6]),
    ("magenta", [0xff, 0x00, 0xff]),
    ("maroon", [0x80, 0x00, 0x00]),
    ("mediumaquamarine", [0x66, 0xcd, 0xaa]),
    ("mediumblue", [0x00, 0x00, 0xcd]),
    ("mediumorchid", [0xba, 0x55, 0xd3]),
    ("mediumpurple", [0x93, 0x70, 0xdb]),
    ("mediumseagreen", [0x3c, 0xb3, 0x71]),
    ("mediumslateblue", [0x7b, 0x68, 0xee]),
    ("mediumspringgreen", [0x00, 0xfa, 0x9a]),
    ("mediumturquoise", [0x48, 0xd1, 0xcc]),
    ("mediumvioletred", [0xc7, 0x15, 0x85]),
    ("midnightblue", [0x19, 0x19, 0x70]),
    ("mintcream", [0xf5, 0xff, 0xfa]),
    ("mistyrose", [0xff, 0xe4, 0xe1]),
    ("moccasin", [0xff, 0xe4, 0xb5]),
    ("navajowhite", [0xff, 0xde, 0xad]),
    ("navy", [0x00, 0x00, 0x80]),
    ("oldlace", [0xfd, 0xf5, 0xe6]),
    ("olive", [0x80, 0x80, 0x00]),
    ("olivedrab", [0x6b, 0x8e, 0x23]),
    ("orange", [0xff, 0xa5, 0x00]),
    ("orangered", [0xff, 0x45, 0x00]),
    ("orchid", [0xda, 0x70, 0xd6]),
    ("palegoldenrod", [0xee, 0xe8, 0xaa]),
    ("palegreen", [0x98, 0xfb, 0x98]),
    ("paleturquoise", [0xaf, 0xee, 0xee]),
    ("palevioletred", [0xdb, 0x70, 0x93]),
    ("papayawhip", [0xff, 0xef, 0xd5]),
    ("peachpuff", [0xff, 0xda, 0xb9]),
    ("peru", [0xcd, 0x85, 0x3f]),
    ("pink", [0xff, 0xc0, 0xcb]),
    ("plum", [0xdd, 0xa0, 0xdd]),
    ("powderblue", [0xb0, 0xe0, 0xe6]),
    ("purple", [0x80, 0x00, 0x80]),
    ("rebeccapurple", [0x66, 0x33, 0x99]),
    ("red", [0xff, 0x00, 0x00]),
    ("rosybrown", [0xbc, 0x8f, 0x8f]),
    ("royalblue", [0x41, 0x69, 0xe1]),
    ("saddlebrown", [0x8b, 0x45, 0x13]),
    ("salmon", [0xfa, 0x80, 0x72]),
    ("sandybrown", [0xf4, 0xa4, 0x60]),
    ("seagreen", [0x2e, 0x8b, 0x57]),
    ("seashell", [0xff, 0xf5, 0xee]),
    ("sienna", [0xa0, 0x52, 0x2d]),
    ("silver", [0xc0, 0xc0, 0xc0]),
    ("skyblue", [0x87, 0xce, 0xeb]),
    ("slateblue", [0x6a, 0x5a, 0xcd]),
    ("slategray", [0x70, 0x80, 0x90]),
    ("slategrey", [0x70, 0x80, 0x90]),
    ("snow", [0xff, 0xfa, 0xfa]),
    ("springgreen", [0x00, 0xff, 0x7f]),
    ("steelblue", [0x46, 0x82, 0xb4]),
    ("tan", [0xd2, 0xb4, 0x8c]),
    ("teal", [0x00, 0x80, 0x80]),
    ("thistle", [0xd8, 0xbf, 0xd8]),
    ("tomato", [0xff, 0x63, 0x47]),
    ("turquoise", [0x40, 0xe0, 0xd0]),
    ("violet", [0xee, 0x82, 0xee]),
    ("wheat", [0xf5, 0xde, 0xb3]),
    ("white", [0xff, 0xff, 0xff]),
    ("whitesmoke", [0xf5, 0xf5, 0xf5]),
    ("yellow", [0xff, 0xff, 0x00]),
    ("yellowgreen", [0x9a, 0xcd, 0x32]),
];

/// A colour name as a scene wrote it. It holds its bytes inline, so a colour
/// stays a Copy value.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Name {
    len: u8,
    bytes: [u8; LONGEST],
}

impl Name {
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)]).unwrap_or_default()
    }
}

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

/// The colour that `text` names, in any case: its red, green and blue, its
/// alpha, 1 or 0 for `transparent`, and the name as written. None for text
/// that names no colour.
pub fn named(text: &str) -> Option<([u8; 3], f64, Name)> {
    let (rgb, alpha) = match text.eq_ignore_ascii_case("transparent") {
        true => ([0; 3], 0.0),
        false => (
            NAMED.iter().find(|(n, _)| n.eq_ignore_ascii_case(text))?.1,
            1.0,
        ),
    };
    let mut bytes = [0u8; LONGEST];
    bytes[..text.len()].copy_from_slice(text.as_bytes());
    let name = Name {
        len: text.len() as u8,
        bytes,
    };
    Some((rgb, alpha, name))
}

/// The colour of a CSS hex colour, `#rgb`, `#rgba`, `#rrggbb` or
/// `#rrggbbaa` in any case: its red, green and blue, its alpha, and its count
/// of digits. None for any other text.
pub fn hex(text: &str) -> Option<([u8; 3], f64, u8)> {
    let digits = text.strip_prefix('#')?;
    let n = digits.len();
    if !matches!(n, 3 | 4 | 6 | 8) || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let wide = if n <= 4 { 1 } else { 2 };
    let byte = |i: usize| {
        let v = u8::from_str_radix(&digits[i * wide..(i + 1) * wide], 16).unwrap_or(0);
        if wide == 1 {
            v * 17
        } else {
            v
        }
    };
    let alpha = match n {
        4 | 8 => f64::from(byte(3)) / 255.0,
        _ => 1.0,
    };
    Some(([byte(0), byte(1), byte(2)], alpha, n as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The named colors of CSS Color 4, from the table of section 6.1 of
    /// the Candidate Recommendation Draft of 30 September 2026,
    /// https://www.w3.org/TR/2026/CRD-css-color-4-20260930/#named-colors,
    /// each row cut to its Color name and Hex rgb columns.
    const SPEC: &str = include_str!("../tools/fixtures/css/named-colors.txt");

    #[test]
    fn every_named_color_of_css_color_4_takes_the_value_of_the_spec() {
        let spec: Vec<(&str, [u8; 3])> = SPEC
            .lines()
            .map(|l| {
                let (name, hex) = l.split_once(' ').unwrap();
                let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
                (name, [byte(1), byte(3), byte(5)])
            })
            .collect();
        assert_eq!(&NAMED[..], &spec[..]);
        let longest = spec.iter().map(|(n, _)| n.len()).max().unwrap();
        assert_eq!(longest, LONGEST);
    }

    #[test]
    fn a_hex_colour_takes_each_of_css_four_forms() {
        // CSS Color 4 section 5.2: a digit of the short forms repeats
        for (text, want) in [
            ("#f80", Some(([0xff, 0x88, 0x00], 1.0, 3))),
            ("#F808", Some(([0xff, 0x88, 0x00], 0x88 as f64 / 255.0, 4))),
            ("#0a0B0c", Some(([0x0a, 0x0b, 0x0c], 1.0, 6))),
            ("#0a0b0c00", Some(([0x0a, 0x0b, 0x0c], 0.0, 8))),
            ("#ff80", Some(([0xff, 0xff, 0x88], 0.0, 4))),
            ("#ff800", None),
            ("#ff8000ff0", None),
            ("#ggg", None),
            ("f80", None),
            ("#", None),
            ("#+f8", None),
        ] {
            assert_eq!(hex(text), want, "{text:?}");
        }
    }

    #[test]
    fn a_name_matches_in_any_case_and_keeps_the_case_it_was_written_in() {
        // Each row: the text, and its colour and alpha, or None.
        let navy = Some(([0, 0, 128], 1.0));
        for (text, want) in [
            ("navy", navy),
            ("NAVY", navy),
            ("Navy", navy),
            ("grey", Some(([128, 128, 128], 1.0))),
            ("LightGoldenrodYellow", Some(([250, 250, 210], 1.0))),
            ("transparent", Some(([0, 0, 0], 0.0))),
            ("Transparent", Some(([0, 0, 0], 0.0))),
            ("navyy", None),
            (" navy", None),
            ("", None),
            ("currentcolor", None),
            ("#000080", None),
            // A Kelvin sign in place of the k of darkkhaki.
            ("dar\u{212a}khaki", None),
        ] {
            let got = named(text);
            assert_eq!(got.map(|(rgb, a, _)| (rgb, a)), want, "{text:?}");
            if let Some((_, _, name)) = got {
                assert_eq!(name.as_str(), text);
            }
        }
    }
}
