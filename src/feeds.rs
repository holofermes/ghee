//! The placeholder array data ghee feeds a scene when nothing else does.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::scene::ArraySize;

/// Normalised coordinates over an array. A 1D size takes x alone. A 2D size
/// takes x and y over the whole grid.
enum Coords {
    Linear(Vec<f32>),
    Grid(Vec<f32>, Vec<f32>),
}

/// Keyed (n, 0) for a 1D size and (w, h) for a 2D one. scene::ArrayInput
/// validates a grid dimension as 1 or more, so the two keys never collide.
static GRIDS: Mutex<BTreeMap<(i64, i64), Coords>> = Mutex::new(BTreeMap::new());

fn key(size: ArraySize) -> (i64, i64) {
    match size {
        ArraySize::Linear(n) => (n, 0),
        ArraySize::Grid(w, h) => (w, h),
    }
}

/// Normalised coordinates for a size, built once and kept. Only the size
/// changes them.
fn build(size: ArraySize) -> Coords {
    match size {
        ArraySize::Linear(n) => {
            let span = (n - 1).max(1) as f32;
            Coords::Linear((0..n).map(|i| i as f32 / span).collect())
        }
        ArraySize::Grid(w, h) => {
            let (sx, sy) = ((w - 1).max(1) as f32, (h - 1).max(1) as f32);
            let mut xs = Vec::with_capacity((w * h) as usize);
            let mut ys = Vec::with_capacity((w * h) as usize);
            for j in 0..h {
                for i in 0..w {
                    xs.push(i as f32 / sx);
                    ys.push(j as f32 / sy);
                }
            }
            Coords::Grid(xs, ys)
        }
    }
}

/// Placeholder values for any named array. A scene with no data attached
/// still animates. The name belongs to the provider and stays unread here.
pub fn demo_feed(_name: &str, size: ArraySize, frame: u64) -> Vec<f32> {
    // t stays f64. Each term narrows to f32 on its own, which the pinned
    // values depend on.
    let t = frame as f64 * 0.05;
    let (t1, t06, t035) = (t as f32, (t * 0.6) as f32, (t * 0.35) as f32);
    let mut grids = GRIDS.lock().unwrap_or_else(|e| e.into_inner());
    match grids.entry(key(size)).or_insert_with(|| build(size)) {
        Coords::Linear(xs) => xs
            .iter()
            .map(|&x| {
                let v = 0.45 + 0.28 * (x * 4.4 + t1).sin() + 0.10 * (x * 9.7 - t06).sin();
                v.clamp(0.0, 1.0)
            })
            .collect(),
        Coords::Grid(xs, ys) => xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let v = 0.42
                    + 0.24 * (x * 4.4 + t1).sin()
                    + 0.10 * (y * 9.7 - t06).sin()
                    + 0.08 * ((x + y) * 6.1 + t035).sin();
                v.clamp(0.0, 1.0)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cached_ptr(size: ArraySize) -> *const f32 {
        let grids = GRIDS.lock().unwrap_or_else(|e| e.into_inner());
        match &grids[&key(size)] {
            Coords::Linear(xs) | Coords::Grid(xs, _) => xs.as_ptr(),
        }
    }

    #[test]
    fn demo_feed_2d_matches_its_formula_and_caches_the_grid() {
        let (w, h) = (7i64, 3i64);
        let got = demo_feed("mask", ArraySize::Grid(w, h), 4);
        let t = 4.0 * 0.05;
        let want: Vec<f64> = (0..w * h)
            .map(|k| {
                let (x, y) = ((k % w) as f64 / 6.0, (k / w) as f64 / 2.0);
                (0.42
                    + 0.24 * (x * 4.4 + t).sin()
                    + 0.10 * (y * 9.7 - t * 0.6).sin()
                    + 0.08 * ((x + y) * 6.1 + t * 0.35).sin())
                .clamp(0.0, 1.0)
            })
            .collect();
        assert_eq!(got.len(), 21);
        for (g, w) in got.iter().zip(&want) {
            assert!((f64::from(*g) - w).abs() < 1e-6, "{g} vs {w}");
        }
        let grid = cached_ptr(ArraySize::Grid(w, h));
        demo_feed("mask", ArraySize::Grid(w, h), 5);
        assert_eq!(cached_ptr(ArraySize::Grid(w, h)), grid);
    }

    #[test]
    fn demo_feed_1d_is_two_travelling_sines_and_stays_in_range() {
        let a = demo_feed("amps", ArraySize::Linear(24), 0);
        let b = demo_feed("amps", ArraySize::Linear(24), 40);
        assert_eq!((a.len(), b.len()), (24, 24));
        assert!(a.iter().all(|v| (0.0..=1.0).contains(v)));
        // the second sine keeps the crest off the beat, so no frame is a
        // shift of another
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3));
        let rolled = b.iter().cycle().skip(23).take(24);
        assert!(a.iter().zip(rolled).any(|(x, y)| (x - y).abs() > 1e-2));
    }

    #[test]
    fn the_placeholder_reproduces_the_recorded_values() {
        let got = demo_feed("amps", ArraySize::Linear(5), 3);
        let want = [
            0.482_854_8,
            0.787_909_03,
            0.549_325_8,
            0.443_453_64,
            0.155_267_25,
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-7, "{g} vs {w}");
        }
        let got = demo_feed("mask", ArraySize::Grid(4, 3), 7);
        let want = [
            0.491_224_97,
            0.698_631_5,
            0.295_952_98,
            0.154_471_86,
            0.400_085_12,
            0.482_587_55,
            0.351_708_38,
            0.092_566_82,
            0.490_926_03,
            0.719_891_7,
            0.318_722_55,
            0.154_335_33,
        ];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-7, "{g} vs {w}");
        }
    }

    #[test]
    fn a_one_element_array_holds_the_origin() {
        // max(size - 1, 1) keeps the division off zero
        assert_eq!(demo_feed("amps", ArraySize::Linear(1), 0), vec![0.45]);
        let got = demo_feed("mask", ArraySize::Grid(1, 1), 11);
        assert!((got[0] - 0.528_345_64).abs() < 1e-7, "{got:?}");
    }
}
