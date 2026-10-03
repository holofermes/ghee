#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
precision mediump int;
varying vec2 v_texCoord;
uniform vec2 resolution;
uniform float time;
uniform float DT;
uniform float frame;
#define gl_FragCoord vec4(v_texCoord * resolution, 0.0, 1.0)
uniform sampler2D times_tex;
const int times_size = 64;
float times_at(int i0){ return texture2D(times_tex, vec2((float(i0)+0.5)/float(times_size), 0.5)).r; }
float times_sample(float t){
  float x = t * float(times_size) - 0.5;
  float i = floor(x);
  float last = float(times_size - 1);
  float a = times_at(int(clamp(i, 0.0, last)));
  float b = times_at(int(clamp(i + 1.0, 0.0, last)));
  return mix(a, b, x - i);
}
uniform vec2 t_size;
uniform float sung;
uniform int n;
uniform vec3 ink;
uniform float gain;
uniform float ghee_vflip;

float twice(float x) { return 2.0 * x; }

attribute vec4 ghee_a0;
attribute vec4 ghee_a1;
attribute vec4 ghee_a2;
attribute vec4 ghee_a3;
varying vec4 ghee_v0;
varying vec4 ghee_v1;
varying vec4 ghee_v2;
varying vec4 ghee_v3;
varying vec4 ghee_v4;
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
    glyph_index = ghee_a0.z;
    glyph_word = ghee_a0.w;
    glyph_line = ghee_a3.x;
    glyph_span = ghee_a3.yz;
    glyph_box = ghee_a1;
    vec2 ghee_corner = mix(ghee_a1.xy, ghee_a1.zw, ghee_a0.xy);
    vec2 ghee_p = glyph_place(ghee_corner);
    // The pixels moved by glyph_place per em along each em axis, exact for an
    // affine glyph_place.
    vec2 ghee_jx = (glyph_place(ghee_corner + vec2(0.0625, 0.0)) - ghee_p) * 16.0;
    vec2 ghee_jy = (glyph_place(ghee_corner + vec2(0.0, 0.0625)) - ghee_p) * 16.0;
    // ghee_det stays above 0, so a glyph scaled to nothing keeps every
    // position finite.
    float ghee_sdet = ghee_jx.x * ghee_jy.y - ghee_jx.y * ghee_jy.x;
    float ghee_det = max(abs(ghee_sdet), 1.0 / 16384.0);
    // Half a pixel of antialiasing, and 1/8 px more, since VC4 lands a vertex
    // up to 1/16 px off the position set by this shader.
    vec2 ghee_grow = max(0.625 * vec2(length(ghee_jy), length(ghee_jx)) / ghee_det, vec2(0.05));
    vec2 ghee_e = ghee_corner + (ghee_a0.xy * 2.0 - 1.0) * ghee_grow;
    vec2 ghee_px = glyph_place(ghee_e);
    // The em spanned by a screen pixel along each em axis, as fwidth gives it.
    vec2 ghee_fw = vec2(abs(ghee_jy.x) + abs(ghee_jy.y),
                        abs(ghee_jx.x) + abs(ghee_jx.y)) / ghee_det;
    vec2 ghee_sz = ghee_a1.zw - ghee_a1.xy;
    // The sample on the glyph's lattice, y up, and the pixels spanned by a
    // lattice step.
    ghee_v0 = vec4(vec2(ghee_e.x - ghee_a1.x, ghee_a1.w - ghee_e.y) / ghee_sz * 2047.0,
                   ghee_sz / (ghee_fw * 2047.0));
    // glyph_ppem: the pixels per em of a placement that scales both axes
    // alike.
    ghee_v1 = vec4(ghee_a2.xyz, sqrt(ghee_det));
    ghee_v2 = vec4(ghee_e, ghee_a0.zw);
    ghee_v3 = vec4(ghee_a3.xyz, 0.0);
    // The lattice steps spanned by a pixel along x and along y of the pass.
    vec2 ghee_k = 2047.0 / ghee_sz;
    ghee_v4 = vec4(ghee_k.x * ghee_jy.y, -ghee_k.x * ghee_jy.x,
                   ghee_k.y * ghee_jx.y, -ghee_k.y * ghee_jx.x)
            / (ghee_sdet < 0.0 ? -ghee_det : ghee_det);
    v_texCoord = ghee_px / resolution;
    gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);
}