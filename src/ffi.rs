//! Bindings for the C engine (ghee/_engine/engine.h).
//!
//! Each item matches one header declaration, in header order.
//! ghee_mp_set_float_array is not here. ghee_mp_set_array replaces it.
#![allow(dead_code)]

use core::ffi::{c_char, c_double, c_int, c_void};

#[repr(C)]
pub struct GheeScene {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct GheeEncoder {
    _opaque: [u8; 0],
}

pub const PARAM_TYPE_FLOAT: c_int = 0;
pub const PARAM_TYPE_INT: c_int = 1;

#[repr(C)]
pub struct FloatParam {
    pub value: f32,
}

#[repr(C)]
pub struct IntParam {
    pub value: c_int,
}

#[repr(C)]
pub union ParamValue {
    pub float_param: *mut FloatParam,
    pub int_param: *mut IntParam,
}

#[repr(C)]
pub struct Parameter {
    pub name: [c_char; 256],
    pub ty: c_int,
    pub value: ParamValue,
}

extern "C" {
    pub fn ghee_create(
        width: c_int,
        height: c_int,
        components: c_int,
        dri_device: *const c_char,
        err: *mut *mut c_char,
    ) -> *mut GheeScene;
    pub fn ghee_display_create(
        width: c_int,
        height: c_int,
        dri_device: *const c_char,
        rgb565: c_int,
        err: *mut *mut c_char,
    ) -> *mut GheeScene;
    pub fn ghee_set_verbosity(level: c_int);
    pub fn ghee_renderer(g: *mut GheeScene) -> *const c_char;
    pub fn ghee_last_error(g: *mut GheeScene) -> *const c_char;
    pub fn ghee_destroy(g: *mut GheeScene);
    pub fn ghee_mp_add_pass(g: *mut GheeScene, glsl_source: *const c_char) -> c_int;
    pub fn ghee_mp_set_pass_size(g: *mut GheeScene, pass: c_int, w: c_int, h: c_int) -> bool;
    pub fn ghee_mp_commit(g: *mut GheeScene) -> c_int;
    pub fn ghee_mp_abort(g: *mut GheeScene);
    pub fn ghee_mp_set_channel(
        g: *mut GheeScene,
        pass: c_int,
        channel: c_int,
        src_pass: c_int,
    ) -> bool;
    pub fn ghee_mp_set_uniform(g: *mut GheeScene, pass: c_int, param: *const Parameter) -> bool;
    pub fn ghee_mp_declare_array(g: *mut GheeScene, pass: c_int, name: *const c_char) -> c_int;
    pub fn ghee_mp_set_array(
        g: *mut GheeScene,
        name: *const c_char,
        data: *const f32,
        count: c_int,
        rows: c_int,
    );
    pub fn ghee_mp_set_array_u8(
        g: *mut GheeScene,
        name: *const c_char,
        data: *const u8,
        count: c_int,
        rows: c_int,
    );
    pub fn ghee_mp_set_asset(
        g: *mut GheeScene,
        name: *const c_char,
        data: *const c_void,
        width: c_int,
        height: c_int,
        components: c_int,
        is_f32: c_int,
        linear: c_int,
    );
    pub fn ghee_mp_render(g: *mut GheeScene, buffer_rgb: *mut u8, pw: c_int, ph: c_int);
    pub fn ghee_mp_present(g: *mut GheeScene, buffer_rgb: *mut u8, pw: c_int, ph: c_int);
    pub fn ghee_mp_reset(g: *mut GheeScene);
    pub fn ghee_max_texture_units(g: *mut GheeScene) -> c_int;
    pub fn ghee_data_texture_linear(g: *mut GheeScene) -> bool;
    pub fn ghee_set_profiling(g: *mut GheeScene, on: bool);
    pub fn ghee_set_time(g: *mut GheeScene, t: f32);
    pub fn ghee_clear_time(g: *mut GheeScene);
    pub fn ghee_set_dt(g: *mut GheeScene, dt: f32);
    pub fn ghee_get_timings(g: *mut GheeScene, draw_ms: *mut c_double, read_ms: *mut c_double);
    pub fn ghee_display_vrefresh(g: *mut GheeScene) -> c_int;
    pub fn ghee_display_probe(
        dri_device: *const c_char,
        width: *mut c_int,
        height: *mut c_int,
    ) -> c_int;
    pub fn ghee_display_renders(dri_device: *const c_char) -> c_int;
    pub fn ghee_display_blank(g: *mut GheeScene, blank: c_int) -> c_int;
    pub fn ghee_display_blanked(g: *mut GheeScene) -> c_int;
    pub fn ghee_encoder_create(
        g: *mut GheeScene,
        w: c_int,
        h: c_int,
        fps: c_int,
        bitrate: c_int,
        err: *mut *mut c_char,
    ) -> *mut GheeEncoder;
    pub fn ghee_encoder_encode_last(
        g: *mut GheeScene,
        e: *mut GheeEncoder,
        nal: *mut *const u8,
        len: *mut c_int,
        is_key: *mut c_int,
    ) -> c_int;
    pub fn ghee_encoder_force_keyframe(e: *mut GheeEncoder);
    pub fn ghee_encoder_flush(
        g: *mut GheeScene,
        e: *mut GheeEncoder,
        nal: *mut *const u8,
        len: *mut c_int,
        is_key: *mut c_int,
    ) -> c_int;
    pub fn ghee_encoder_destroy(g: *mut GheeScene, e: *mut GheeEncoder);
    pub fn ghee_pack_frame(
        width: c_int,
        height: c_int,
        components: c_int,
        bw: c_int,
        rgb: *const u8,
        packed: *mut u8,
    );
}
