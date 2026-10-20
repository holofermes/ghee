//! Bindings for the C engine (ghee/_engine/engine.h).
//!
//! Each item matches one header declaration, in header order.
#![allow(dead_code)]

use core::ffi::{c_char, c_int, c_void};

#[repr(C)]
pub struct GheeScene {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct GheeEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct GheeCapture {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct GheeHwdec {
    _opaque: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GheeCaptureMode {
    pub fourcc: u32,
    pub width: c_int,
    pub height: c_int,
    pub fps: f64,
}

pub const PARAM_TYPE_FLOAT: c_int = 0;
pub const PARAM_TYPE_INT: c_int = 1;
pub const PARAM_TYPE_VEC2: c_int = 2;
pub const PARAM_TYPE_VEC3: c_int = 3;
pub const PARAM_TYPE_VEC4: c_int = 4;

pub const GHEE_BLEND_NORMAL: c_int = 0;
pub const GHEE_BLEND_ADD: c_int = 1;
pub const GHEE_BLEND_MULTIPLY: c_int = 2;
pub const GHEE_BLEND_SCREEN: c_int = 3;

pub const GHEE_FILTER_NEAREST: c_int = 0;
pub const GHEE_FILTER_LINEAR: c_int = 1;
pub const GHEE_FILTER_MIPMAP: c_int = 2;

#[repr(C)]
pub struct FloatParam {
    pub value: f32,
}

#[repr(C)]
pub struct IntParam {
    pub value: c_int,
}

#[repr(C)]
pub struct VecParam {
    pub value: [f32; 4],
}

#[repr(C)]
pub union ParamValue {
    pub float_param: *mut FloatParam,
    pub int_param: *mut IntParam,
    pub vec_param: *mut VecParam,
}

#[repr(C)]
pub struct Parameter {
    pub name: *const c_char,
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
    pub fn ghee_display_create_on(
        card: c_int,
        width: c_int,
        height: c_int,
        dri_device: *const c_char,
        rgb565: c_int,
        err: *mut *mut c_char,
    ) -> *mut GheeScene;
    pub fn ghee_set_verbosity(level: c_int);
    pub fn ghee_renderer(g: *mut GheeScene) -> *const c_char;
    pub fn ghee_last_error(g: *mut GheeScene) -> *const c_char;
    pub fn ghee_last_error_is_compile(g: *mut GheeScene) -> bool;
    pub fn ghee_destroy(g: *mut GheeScene);
    pub fn ghee_mp_add_pass(
        g: *mut GheeScene,
        name: *const c_char,
        glsl_source: *const c_char,
        alpha: c_int,
    ) -> c_int;
    pub fn ghee_mp_set_pass_size(g: *mut GheeScene, pass: c_int, w: c_int, h: c_int) -> bool;
    pub fn ghee_mp_set_pass_sampling(g: *mut GheeScene, pass: c_int, filter: c_int) -> bool;
    pub fn ghee_mp_carry(g: *mut GheeScene, pass: c_int, from: c_int) -> bool;
    pub fn ghee_mp_commit(g: *mut GheeScene) -> c_int;
    pub fn ghee_mp_abort(g: *mut GheeScene);
    pub fn ghee_mp_set_channel(
        g: *mut GheeScene,
        pass: c_int,
        channel: c_int,
        src_pass: c_int,
    ) -> bool;
    pub fn ghee_mp_set_uniform(g: *mut GheeScene, pass: c_int, param: *const Parameter) -> bool;
    pub fn ghee_mp_set_pass_uniform(
        g: *mut GheeScene,
        pass: c_int,
        param: *const Parameter,
    ) -> bool;
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
        filter: c_int,
    ) -> bool;
    pub fn ghee_mp_set_asset_rows(
        g: *mut GheeScene,
        name: *const c_char,
        data: *const c_void,
        y: c_int,
        rows: c_int,
    ) -> bool;
    pub fn ghee_mp_set_asset_rect(
        g: *mut GheeScene,
        name: *const c_char,
        level: c_int,
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
        data: *const c_void,
    ) -> bool;
    pub fn ghee_mp_drop_asset(g: *mut GheeScene, name: *const c_char);
    pub fn ghee_mp_set_asset_dmabuf(
        g: *mut GheeScene,
        name: *const c_char,
        fd: c_int,
        width: c_int,
        height: c_int,
        pitch: c_int,
    ) -> bool;
    pub fn ghee_mp_forget_dmabufs(g: *mut GheeScene, name: *const c_char);
    pub fn ghee_mp_reads(g: *mut GheeScene, name: *const c_char) -> bool;
    pub fn ghee_mp_add_batch(
        g: *mut GheeScene,
        pass: c_int,
        name: *const c_char,
        vertex_source: *const c_char,
        fragment_source: *const c_char,
        blend: c_int,
    ) -> c_int;
    pub fn ghee_mp_set_batch_program(
        g: *mut GheeScene,
        batch: c_int,
        vertex_source: *const c_char,
        fragment_source: *const c_char,
    ) -> bool;
    pub fn ghee_mp_keep_program(
        g: *mut GheeScene,
        vertex_source: *const c_char,
        fragment_source: *const c_char,
    ) -> bool;
    pub fn ghee_mp_let_go_program(
        g: *mut GheeScene,
        vertex_source: *const c_char,
        fragment_source: *const c_char,
    );
    pub fn ghee_mp_set_batch_uniform(
        g: *mut GheeScene,
        batch: c_int,
        param: *const Parameter,
    ) -> bool;
    pub fn ghee_mp_set_batch_opacity(g: *mut GheeScene, batch: c_int, opacity: f32) -> bool;
    pub fn ghee_mp_insert_batch(
        g: *mut GheeScene,
        pass: c_int,
        after: c_int,
        name: *const c_char,
        vertex_source: *const c_char,
        fragment_source: *const c_char,
        blend: c_int,
    ) -> c_int;
    pub fn ghee_mp_remove_batch(g: *mut GheeScene, batch: c_int) -> bool;
    pub fn ghee_mp_set_batch_quads(
        g: *mut GheeScene,
        batch: c_int,
        data: *const f32,
        quads: c_int,
        nattr: c_int,
    ) -> bool;
    pub fn ghee_mp_set_batch_quads_at(
        g: *mut GheeScene,
        batch: c_int,
        first: c_int,
        data: *const f32,
        quads: c_int,
    ) -> bool;
    pub fn ghee_mp_batch_quads(g: *mut GheeScene, batch: c_int, nattr: c_int) -> c_int;
    pub fn ghee_mp_bind_asset(
        g: *mut GheeScene,
        batch: c_int,
        name: *const c_char,
        asset: *const c_char,
    ) -> bool;
    pub fn ghee_mp_render(g: *mut GheeScene, buffer_rgb: *mut u8, pw: c_int, ph: c_int);
    pub fn ghee_mp_present(g: *mut GheeScene, buffer_rgb: *mut u8, pw: c_int, ph: c_int);
    pub fn ghee_read_linear(g: *mut GheeScene, mode: c_int) -> bool;
    pub fn ghee_mp_repeat(g: *mut GheeScene, buffer_rgb: *mut u8, pw: c_int, ph: c_int) -> bool;
    pub fn ghee_mp_drew(g: *mut GheeScene, buffer_rgb: *const u8, pw: c_int, ph: c_int);
    pub fn ghee_flip_rotate(
        prev: *mut *mut c_void,
        shown: *mut *mut c_void,
        bo: *mut c_void,
        flipped: bool,
    ) -> *mut c_void;
    pub fn ghee_mp_reset(g: *mut GheeScene);
    pub fn ghee_keep_alpha(g: *mut GheeScene, keep: c_int);
    pub fn ghee_set_rotate(g: *mut GheeScene, quarter_turns: c_int) -> bool;
    pub fn ghee_max_texture_units(g: *mut GheeScene) -> c_int;
    pub fn ghee_gl_texture_units(g: *mut GheeScene) -> c_int;
    pub fn ghee_max_texture_size(g: *mut GheeScene) -> c_int;
    pub fn ghee_gl_fd(g: *mut GheeScene) -> c_int;
    pub fn ghee_data_texture_linear(g: *mut GheeScene) -> bool;
    pub fn ghee_mipmaps_npot(g: *mut GheeScene) -> bool;
    pub fn ghee_gles3(g: *mut GheeScene) -> bool;
    pub fn ghee_compiles(g: *mut GheeScene, fragment: *const c_char) -> bool;
    pub fn ghee_compiled(g: *mut GheeScene) -> c_int;
    pub fn ghee_draws(g: *mut GheeScene) -> u64;
    pub fn ghee_set_time(g: *mut GheeScene, t: f32);
    pub fn ghee_clear_time(g: *mut GheeScene);
    pub fn ghee_restart_clock(g: *mut GheeScene);
    pub fn ghee_set_dt(g: *mut GheeScene, dt: f32);
    pub fn ghee_display_vrefresh(g: *mut GheeScene) -> c_int;
    pub fn ghee_display_probe(
        dri_device: *const c_char,
        width: *mut c_int,
        height: *mut c_int,
    ) -> c_int;
    pub fn ghee_display_open(
        dri_device: *const c_char,
        width: *mut c_int,
        height: *mut c_int,
    ) -> c_int;
    pub fn ghee_display_renders(dri_device: *const c_char) -> c_int;
    pub fn ghee_display_renders_on(card: c_int) -> c_int;
    pub fn ghee_display_blank(g: *mut GheeScene, blank: c_int) -> c_int;
    pub fn ghee_display_blanked(g: *mut GheeScene) -> c_int;
    pub fn ghee_display_set_drop(g: *mut GheeScene, drop: c_int) -> c_int;
    pub fn ghee_display_taken(g: *mut GheeScene) -> u64;
    pub fn ghee_jit_start(draw: *const f64, ndraw: c_int, busy: *const f64, nbusy: c_int) -> f64;
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
    pub fn ghee_jpeg_create(
        g: *mut GheeScene,
        w: c_int,
        h: c_int,
        quality: c_int,
        err: *mut *mut c_char,
    ) -> *mut GheeEncoder;
    pub fn ghee_capture_probe(path: *const c_char, name: *mut c_char, name_len: c_int) -> c_int;
    pub fn ghee_capture_open(
        path: *const c_char,
        err: *mut c_char,
        err_len: c_int,
    ) -> *mut GheeCapture;
    pub fn ghee_capture_modes(c: *mut GheeCapture, out: *mut GheeCaptureMode, max: c_int) -> c_int;
    pub fn ghee_capture_start(
        c: *mut GheeCapture,
        fourcc: u32,
        w: c_int,
        h: c_int,
        fps: f64,
        err: *mut c_char,
        err_len: c_int,
    ) -> c_int;
    pub fn ghee_capture_format(
        c: *mut GheeCapture,
        fourcc: *mut u32,
        w: *mut c_int,
        h: *mut c_int,
        bytesperline: *mut c_int,
    );
    pub fn ghee_capture_next(
        c: *mut GheeCapture,
        timeout_ms: c_int,
        data: *mut *const u8,
        len: *mut usize,
        err: *mut c_char,
        err_len: c_int,
    ) -> c_int;
    pub fn ghee_capture_release(c: *mut GheeCapture);
    pub fn ghee_capture_close(c: *mut GheeCapture);
    pub fn ghee_thread_idle() -> c_int;
    pub fn ghee_hwdec_open(
        w: c_int,
        h: c_int,
        ow: c_int,
        oh: c_int,
        err: *mut c_char,
        err_len: c_int,
    ) -> *mut GheeHwdec;
    pub fn ghee_hwdec_push(
        d: *mut GheeHwdec,
        jpg: *const u8,
        len: usize,
        scale: c_int,
        timeout_ms: c_int,
        index: *mut c_int,
        err: *mut c_char,
        err_len: c_int,
    ) -> c_int;
    pub fn ghee_hwdec_out(
        d: *mut GheeHwdec,
        index: c_int,
        fd: *mut c_int,
        w: *mut c_int,
        h: *mut c_int,
        pitch: *mut c_int,
    );
    pub fn ghee_hwdec_give(d: *mut GheeHwdec, index: c_int);
    pub fn ghee_hwdec_close(d: *mut GheeHwdec);
}
