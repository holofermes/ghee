//! A handle on a render that runs on another thread: stop it, or pause it.
//!
//! The host makes a [`Control`], hands a clone to the console through
//! `console::server::Options::control`, and keeps one. Any thread then stops
//! or pauses the render through it. [`stop_on_signals`] stops it on SIGINT and
//! SIGTERM.

use std::ffi::{c_int, c_void};
use std::io::{self, Read};
use std::os::fd::IntoRawFd;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// A handle on one render. Clones share it.
///
/// A stopped render stays stopped: make a new Control for the next run.
#[derive(Clone, Default)]
pub struct Control(Arc<Inner>);

/// What a host wakes a render with when its state changes.
type Wake = Box<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Inner {
    stopped: AtomicBool,
    paused: AtomicBool,
    /// The signal that stopped the render, 0 for none.
    signal: AtomicI32,
    /// A signal stops the render: catch() took it.
    caught: AtomicBool,
    wake: Mutex<Option<Wake>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Control {
    pub fn new() -> Control {
        Control::default()
    }

    /// End the render. Call it from any thread, before the render starts
    /// too, but never from a signal handler, since it takes a lock: use
    /// [`stop_on_signals`] for signals.
    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::SeqCst);
        self.wake();
    }

    /// True once the render stops. A caught signal stops it at once, before
    /// ghee's thread wakes the console.
    pub fn stopped(&self) -> bool {
        self.0.stopped.load(Ordering::SeqCst) || self.signal().is_some()
    }

    /// Pause the render, or let it go on. A paused render blanks the
    /// display, turns its sinks off and draws nothing. The console keeps
    /// taking requests meanwhile, and they land on the first frame after the
    /// pause. Call it from any thread, before the render starts too.
    pub fn pause(&self, on: bool) {
        self.0.paused.store(on, Ordering::SeqCst);
        self.wake();
    }

    pub fn paused(&self) -> bool {
        self.0.paused.load(Ordering::SeqCst)
    }

    /// The signal that stopped the render through [`stop_on_signals`], such
    /// as 15 for SIGTERM.
    pub fn signal(&self) -> Option<i32> {
        let sig = match self.0.caught.load(Ordering::SeqCst) {
            true => SIGNALLED.load(Ordering::SeqCst),
            false => self.0.signal.load(Ordering::SeqCst),
        };
        (sig != 0).then_some(sig)
    }

    /// Have `wake` called on every stop and pause, until detach().
    pub(crate) fn attach(&self, wake: Wake) {
        *lock(&self.0.wake) = Some(wake);
    }

    pub(crate) fn detach(&self) {
        *lock(&self.0.wake) = None;
    }

    fn wake(&self) {
        if let Some(wake) = lock(&self.0.wake).as_ref() {
            wake();
        }
    }

    fn same(&self, other: &Control) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

// Linux signal numbers. The handler is a plain function pointer, so the
// install needs neither SIG_DFL nor SIG_IGN.
pub(crate) const SIGINT: c_int = 2;
pub(crate) const SIGTERM: c_int = 15;

extern "C" {
    fn signal(sig: c_int, handler: usize) -> usize;
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn __errno_location() -> *mut c_int;
}

/// The socket that on_signal() writes the signal to, -1 until the first
/// catch().
static WAKE: AtomicI32 = AtomicI32::new(-1);

/// The last signal caught since the last catch(), 0 for none.
static SIGNALLED: AtomicI32 = AtomicI32::new(0);

/// The renders that a signal stops.
static CAUGHT: Mutex<Vec<Control>> = Mutex::new(Vec::new());

/// A signal handler may only make async-signal-safe calls, and taking a lock
/// is not one: the thread that the signal lands on may hold that lock.
extern "C" fn on_signal(sig: c_int) {
    SIGNALLED.store(sig, Ordering::SeqCst);
    unsafe {
        let errno = *__errno_location();
        write(WAKE.load(Ordering::SeqCst), [sig as u8].as_ptr().cast(), 1);
        *__errno_location() = errno;
    }
}

/// Stop `control` on SIGINT and SIGTERM. A thread of ghee's own stops it, so
/// the signal handler takes no lock. Call it once per render, on the thread
/// that starts the render, before the render runs. The handlers stay
/// installed after the render ends.
pub fn stop_on_signals(control: &Control) -> io::Result<()> {
    catch(control).map(|_| ())
}

/// stop_on_signals(), returning the handlers of SIGINT and SIGTERM that it
/// replaced, for release().
pub(crate) fn catch(control: &Control) -> io::Result<[usize; 2]> {
    if WAKE.load(Ordering::SeqCst) < 0 {
        let (wake, woken) = std::os::unix::net::UnixStream::pair()?;
        wake.set_nonblocking(true)?;
        WAKE.store(wake.into_raw_fd(), Ordering::SeqCst);
        std::thread::Builder::new()
            .name("ghee-signals".into())
            .spawn(move || {
                let mut bytes = [0u8; 64];
                loop {
                    match (&woken).read(&mut bytes) {
                        Ok(0) => return,
                        Ok(n) => {
                            let sig = i32::from(bytes[n - 1]);
                            for c in lock(&CAUGHT).iter() {
                                c.0.signal.store(sig, Ordering::SeqCst);
                                c.stop();
                            }
                        }
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                        Err(_) => return,
                    }
                }
            })?;
    }
    SIGNALLED.store(0, Ordering::SeqCst);
    control.0.caught.store(true, Ordering::SeqCst);
    lock(&CAUGHT).push(control.clone());
    let handler = on_signal as extern "C" fn(c_int) as usize;
    unsafe { Ok([signal(SIGINT, handler), signal(SIGTERM, handler)]) }
}

/// Put back the handlers of SIGINT and SIGTERM that catch() replaced, and
/// let `control` go.
pub(crate) fn release(control: &Control, held: [usize; 2]) {
    control
        .0
        .signal
        .store(SIGNALLED.load(Ordering::SeqCst), Ordering::SeqCst);
    control.0.caught.store(false, Ordering::SeqCst);
    lock(&CAUGHT).retain(|c| !c.same(control));
    unsafe {
        signal(SIGINT, held[0]);
        signal(SIGTERM, held[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn a_stop_and_a_pause_wake_the_render_and_a_stop_stays() {
        let c = Control::new();
        let woken = Arc::new(AtomicUsize::new(0));
        let w = Arc::clone(&woken);
        c.attach(Box::new(move || {
            w.fetch_add(1, Ordering::SeqCst);
        }));
        c.pause(true);
        assert!(c.paused() && !c.stopped());
        c.clone().stop();
        assert!(c.stopped());
        assert_eq!(woken.load(Ordering::SeqCst), 2);
        c.detach();
        c.pause(false);
        assert_eq!(woken.load(Ordering::SeqCst), 2);
        assert_eq!(c.signal(), None);
    }

    #[test]
    fn a_signal_stops_the_render_that_caught_it() {
        extern "C" {
            fn raise(sig: c_int) -> c_int;
        }
        let c = Control::new();
        let held = catch(&c).unwrap();
        unsafe { raise(SIGTERM) };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !c.stopped() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        release(&c, held);
        assert!(c.stopped());
        assert_eq!(c.signal(), Some(SIGTERM));
    }
}
