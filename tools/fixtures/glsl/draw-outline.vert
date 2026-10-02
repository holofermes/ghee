#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
precision mediump int;
uniform vec2 resolution;
uniform float time;
uniform float DT;
uniform float frame;
uniform float vflip;
uniform float sung;
uniform int n;
uniform float ink_r;
uniform float ink_g;
uniform float ink_b;
#define ink vec3(ink_r, ink_g, ink_b)
uniform float gain;
attribute vec4 glyph_a0;
attribute vec4 glyph_a1;
attribute vec4 glyph_a2;
attribute vec4 glyph_a3;
uniform float t_w;
uniform float t_h;
#define t_size vec2(t_w, t_h)
varying vec2 v_texCoord;
varying vec4 glyph_v0;
varying vec4 glyph_v1;
varying vec4 glyph_v2;
varying vec4 glyph_v3;
varying vec4 glyph_v4;
float glyph_index;
float glyph_word;
float glyph_line;
vec2 glyph_span;
vec4 glyph_box;

vec2 glyph_place(vec2 em) {
    vec2 off = (resolution - t_size * 46.0) * 0.5;
    return off + em * 46.0;
}

void main() {
    glyph_index = glyph_a0.z;
    glyph_word = glyph_a0.w;
    glyph_line = glyph_a3.x;
    glyph_span = glyph_a3.yz;
    glyph_box = glyph_a1;
    vec2 glyph_corner = mix(glyph_a1.xy, glyph_a1.zw, glyph_a0.xy);
    vec2 glyph_p = glyph_place(glyph_corner);
    // The pixels moved by glyph_place per em along each em axis, exact for an
    // affine glyph_place.
    vec2 glyph_jx = (glyph_place(glyph_corner + vec2(0.0625, 0.0)) - glyph_p) * 16.0;
    vec2 glyph_jy = (glyph_place(glyph_corner + vec2(0.0, 0.0625)) - glyph_p) * 16.0;
    // glyph_det stays above 0, so a glyph scaled to nothing keeps every
    // position finite.
    float glyph_sdet = glyph_jx.x * glyph_jy.y - glyph_jx.y * glyph_jy.x;
    float glyph_det = max(abs(glyph_sdet), 1.0 / 16384.0);
    // Half a pixel of antialiasing, and 1/8 px more, since VC4 lands a vertex
    // up to 1/16 px off the position set by this shader.
    vec2 glyph_grow = max(0.625 * vec2(length(glyph_jy), length(glyph_jx)) / glyph_det, vec2(0.05));
    vec2 glyph_e = glyph_corner + (glyph_a0.xy * 2.0 - 1.0) * glyph_grow;
    vec2 glyph_px = glyph_place(glyph_e);
    // The em spanned by a screen pixel along each em axis, as fwidth gives it.
    vec2 glyph_fw = vec2(abs(glyph_jy.x) + abs(glyph_jy.y),
                         abs(glyph_jx.x) + abs(glyph_jx.y)) / glyph_det;
    vec2 glyph_sz = glyph_a1.zw - glyph_a1.xy;
    // The sample on the glyph's lattice, y up, and the pixels spanned by a
    // lattice step.
    glyph_v0 = vec4(vec2(glyph_e.x - glyph_a1.x, glyph_a1.w - glyph_e.y) / glyph_sz * 2047.0,
                    glyph_sz / (glyph_fw * 2047.0));
    // glyph_ppem: the pixels per em of a placement that scales both axes
    // alike.
    glyph_v1 = vec4(glyph_a2.xyz, sqrt(glyph_det));
    glyph_v2 = vec4(glyph_e, glyph_a0.zw);
    glyph_v3 = vec4(glyph_a3.xyz, 0.0);
    // The lattice steps spanned by a pixel along x and along y of the pass.
    vec2 glyph_k = 2047.0 / glyph_sz;
    glyph_v4 = vec4(glyph_k.x * glyph_jy.y, -glyph_k.x * glyph_jy.x,
                    glyph_k.y * glyph_jx.y, -glyph_k.y * glyph_jx.x)
             / (glyph_sdet < 0.0 ? -glyph_det : glyph_det);
    v_texCoord = glyph_px / resolution;
    gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, vflip), 0.0, 1.0);
}