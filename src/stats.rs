//! The clock, the frame rate and the output names that /stats reports.

use std::sync::OnceLock;
use std::time::Instant;

/// /stats "out" entries.
pub const DEST_KMS: &str = "kms";
pub const DEST_PANEL: &str = "panel";
pub const DEST_FILE: &str = "file";
pub const DEST_PIPE: &str = "pipe";
pub const DEST_H264: &str = "h264";
pub const DEST_MJPEG: &str = "mjpeg";

/// Seconds off the monotonic clock. The whole process shares one origin. One
/// caller's timestamp means the same to another.
pub fn monotonic() -> f64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Completed work per elapsed second.
pub fn rate(count: u64, elapsed: f64) -> f64 {
    if elapsed > 0.0 {
        count as f64 / elapsed
    } else {
        0.0
    }
}

/// Frames completed per elapsed second, over a trailing window.
///
/// tick() belongs to the thread that does the work. value() only reads the
/// clock, so any thread calls it.
pub struct FrameRate {
    pub window: f64,
    start: f64,
    last: f64,
    count: u64,
    /// the last closed window's rate. None until a window closes
    rate: Option<f64>,
}

impl FrameRate {
    /// A counter over the default one-second window.
    pub fn new() -> FrameRate {
        FrameRate::with_window(1.0)
    }

    pub fn with_window(window: f64) -> FrameRate {
        let mut fr = FrameRate {
            window,
            start: 0.0,
            last: 0.0,
            count: 0,
            rate: None,
        };
        fr.reset(None);
        fr
    }

    pub fn reset(&mut self, now: Option<f64>) {
        self.start = now.unwrap_or_else(monotonic);
        self.last = self.start;
        self.count = 0;
        self.rate = None;
    }

    pub fn tick(&mut self, now: Option<f64>) {
        let now = now.unwrap_or_else(monotonic);
        self.count += 1;
        self.last = now;
        let elapsed = now - self.start;
        if elapsed >= self.window {
            self.rate = Some(rate(self.count, elapsed));
            self.start = now;
            self.count = 0;
        }
    }

    /// The last closed window's rate. A whole window with no completed work
    /// reads zero.
    pub fn value(&self, now: Option<f64>) -> f64 {
        let now = now.unwrap_or_else(monotonic);
        if now - self.last >= self.window {
            return 0.0;
        }
        let elapsed = now - self.start;
        match self.rate {
            Some(r) if elapsed < self.window => r,
            _ => rate(self.count, elapsed),
        }
    }
}

impl Default for FrameRate {
    fn default() -> FrameRate {
        FrameRate::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tolerance relative to the expected value.
    fn approx(got: f64, want: f64, rel: f64) -> bool {
        (got - want).abs() <= rel * want.abs() + 1e-12
    }

    /// Tick `fr` at the given periods, cycling, for `seconds` of clock.
    fn feed(fr: &mut FrameRate, periods: &[f64], seconds: f64) -> (f64, u64) {
        let mut t = 1000.0;
        fr.reset(Some(t));
        let end = t + seconds;
        let mut i = 0u64;
        while t < end {
            t += periods[(i as usize) % periods.len()];
            fr.tick(Some(t));
            i += 1;
        }
        (t, i)
    }

    #[test]
    fn a_bimodal_sequence_publishes_its_true_rate() {
        // alternating 4 ms and 32 ms frames give 2 frames per 36 ms
        let mut fr = FrameRate::new();
        let (t, _) = feed(&mut fr, &[0.004, 0.032], 4.0);
        assert!(approx(fr.value(Some(t)), 2.0 / 0.036, 0.02));
        // a mean of reciprocals reads higher
        assert!(fr.value(Some(t)) < 0.5 * ((1.0 / 0.004) + (1.0 / 0.032)) / 2.0);
    }

    #[test]
    fn a_constant_sequence_publishes_that_rate() {
        let mut fr = FrameRate::new();
        let (t, _) = feed(&mut fr, &[0.02], 4.0);
        assert!(approx(fr.value(Some(t)), 50.0, 0.02));
    }

    #[test]
    fn the_rate_is_completed_work_over_elapsed_time() {
        assert_eq!(rate(60, 6.0), 10.0);
        assert_eq!(rate(1, 0.0), 0.0);
        assert_eq!(rate(0, 5.0), 0.0);
    }

    #[test]
    fn a_skipped_frame_invents_nothing() {
        let mut fr = FrameRate::new();
        fr.reset(Some(0.0));
        for i in 1..=50 {
            fr.tick(Some(f64::from(i) * 0.02));
        }
        let published = fr.value(Some(1.0));
        assert!(approx(published, 50.0, 1e-6));
        // a skipped frame is not a tick, so the published rate stands
        assert_eq!(fr.value(Some(1.02)), published);
        assert_eq!(fr.value(Some(1.5)), published);
    }

    #[test]
    fn an_idle_stretch_falls_with_the_clock() {
        let mut fr = FrameRate::new();
        fr.reset(Some(0.0));
        for i in 1..=50 {
            fr.tick(Some(f64::from(i) * 0.02)); // the window closes at 1.0 s and carries nothing
        }
        assert_eq!(fr.value(Some(2.0)), 0.0);
        assert_eq!(fr.value(Some(11.0)), 0.0);
        fr.tick(Some(11.1)); // one frame closes a ten-second window
        assert!(approx(fr.value(Some(11.1)), 1.0 / 10.1, 1e-6));
        assert_eq!(fr.value(Some(21.1)), 0.0);
    }

    #[test]
    fn a_counter_that_stops_mid_window_reads_zero() {
        // the viewer goes away: the ticks stop with a part-filled window, and
        // count/elapsed only decays towards zero
        let mut fr = FrameRate::new();
        fr.reset(Some(0.0));
        for i in 1..=30 {
            fr.tick(Some(f64::from(i) * 0.01)); // the window is still open at 0.3 s
        }
        assert!(fr.value(Some(0.4)) > 0.0);
        assert_eq!(fr.value(Some(1.3)), 0.0);
        assert_eq!(fr.value(Some(60.0)), 0.0);
    }

    #[test]
    fn a_fresh_counter_reports_the_window_it_has() {
        let mut fr = FrameRate::new();
        fr.reset(Some(0.0));
        assert_eq!(fr.value(Some(0.5)), 0.0);
        for i in 1..=10 {
            fr.tick(Some(f64::from(i) * 0.02));
        }
        assert!(approx(fr.value(Some(0.2)), 50.0, 1e-6));
    }

    #[test]
    fn reset_drops_the_open_window() {
        let mut fr = FrameRate::new();
        let (t, _) = feed(&mut fr, &[0.02], 4.0);
        assert!(fr.value(Some(t)) > 0.0);
        fr.reset(Some(t));
        assert_eq!(fr.value(Some(t + 2.0)), 0.0);
    }

    #[test]
    fn the_clock_is_one_forward_running_origin() {
        let a = monotonic();
        let b = monotonic();
        assert!(b >= a);
        assert!(b - a < 1.0);
    }
}
