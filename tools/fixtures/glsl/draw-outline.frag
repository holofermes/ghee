#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
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
uniform sampler2D ghee_t_tex;
uniform float ghee_t_tw;
uniform float ghee_t_itw;
uniform float ghee_t_ith;
uniform vec2 t_size;
uniform float ghee_t_gx;
uniform float ghee_t_gy;
uniform float ghee_t_cols;
uniform float ghee_t_rows;
uniform float ghee_t_cell;
uniform float ghee_t_cells;
uniform float ghee_t_nb;
// A texture read on the Pi 5 returns a byte too coarse to take bits from as it
// is, so every byte goes through int() first: a read of byte b lies near
// b / 255, and int(t * 255.5) floors to b. On VC4, float(int(x)) takes 2
// instructions and floor(x) 4, and every floor here takes x >= 0.
vec4 ghee_t_texel(float i) {
    float row = float(int(i * ghee_t_itw));
    vec2 at = vec2(i - row * ghee_t_tw + 0.5, row + 0.5);
    return vec4(ivec4(texture2D(ghee_t_tex, at * vec2(ghee_t_itw, ghee_t_ith)) * 255.5));
}
// The texture coordinate of a band's list: the first list for the ray toward
// +, the second for the ray toward -.
vec2 ghee_t_list(vec4 band, float dir) {
    float i = (band.r * 256.0 + band.g) * 256.0 + band.b + (dir < 0.0 ? band.a * 2.0 : 0.0);
    float row = float(int(i * ghee_t_itw));
    return vec2(i - row * ghee_t_tw + 0.5, row + 0.5) * vec2(ghee_t_itw, ghee_t_ith);
}
// One texel of a curve: a lattice point, and the offset of p2 from p3 on one
// axis.
vec3 ghee_t_lattice(vec4 t) {
    vec4 b = vec4(ivec4(t * 255.5));
    vec2 hi = vec2(ivec2(b.ba * (1.0 / 32.0)));
    return vec3(b.rg * 8.0 + hi, (b.b - hi.x * 32.0) * 32.0 + b.a - hi.y * 32.0 - 512.0);
}
// Where the curve crosses y = 0, as x. The roots of a t^2 - 2b t + c, each
// from the form of the quadratic formula that subtracts no near-equal values.
vec2 ghee_t_cross(vec2 p1, vec2 p2, vec2 p3) {
    vec2 a = p1 - p2 * 2.0 + p3;
    vec2 b = p1 - p2;
    float disc = b.y * b.y - a.y * p1.y;
    float d = sqrt(max(disc, 0.0));
    float q = b.y >= 0.0 ? b.y + d : b.y - d;
    float ra = 1.0 / a.y;
    float cq = q != 0.0 ? p1.y / q : 0.0;
    float qa = q * ra;
    float t1 = b.y >= 0.0 ? cq : qa;
    float t2 = b.y >= 0.0 ? qa : cq;
    // A curve that misses the ray has both roots at its extreme, where they
    // cancel.
    float tm = b.y * ra;
    t1 = disc > 0.0 ? t1 : tm;
    t2 = disc > 0.0 ? t2 : tm;
    return vec2((a.x * t1 - b.x * 2.0) * t1 + p1.x, (a.x * t2 - b.x * 2.0) * t2 + p1.x);
}
// The coverage of a glyph at u on its lattice, y up, drawn at ppu pixels a
// lattice step. Its band entries start at texel h: nb.x horizontal bands, then
// nb.y vertical ones.
float ghee_t_outline(float h, vec2 nb, vec2 u, vec2 ppu) {
    vec2 m = 0.5 / ppu;
    float cov = 0.0;
    if (u.x > -m.x && u.y > -m.y && u.x < 2047.0 + m.x && u.y < 2047.0 + m.y) {
        vec2 band = min(vec2(ivec2(max(u.yx, 0.0) * (nb / 2047.0))), nb - 1.0);
        vec4 hb = ghee_t_texel(h + band.x);
        // Each ray runs toward the nearer edge of the box. A ray toward -
        // reads its band mirrored, so its coverage flips sign.
        float dx = u.x < 1023.5 ? -1.0 : 1.0;
        vec2 uh = vec2(dx > 0.0 ? u.x : 2047.0 - u.x, u.y);
        float xcov = 0.0;
        float xwgt = 0.0;
        vec2 at = ghee_t_list(hb, dx);
        for (int i = 0; i < 64; i++) {
            if (float(i) >= hb.a) break;
            vec3 c1 = ghee_t_lattice(texture2D(ghee_t_tex, at));
            vec3 c3 = ghee_t_lattice(texture2D(ghee_t_tex, at + vec2(ghee_t_itw, 0.0)));
            at.x += ghee_t_itw * 2.0;
            vec2 p1 = c1.xy - uh;
            vec2 p3 = c3.xy - uh;
            vec2 p2 = p3 + vec2(c1.z, c3.z);
            // Sorted by their far end, largest first: a curve wholly behind
            // the pixel adds nothing, and neither does any after it.
            if (max(max(p1.x, p2.x), p3.x) * ppu.x < -0.5) break;
            // Which roots count, by the signs of p1.y, p2.y and p3.y: Slug's
            // lookup table 0x2E74, written out as logic.
            bool n1 = p1.y < 0.0;
            bool n2 = p2.y < 0.0;
            bool n3 = p3.y < 0.0;
            bool e1 = n3 ? !(n1 && n2) : n2 && !n1;
            bool e2 = n3 ? n1 && !n2 : n1 || n2;
            if (e1 || e2) {
                vec2 r = ghee_t_cross(p1, p2, p3) * ppu.x;
                xcov += e1 ? clamp(r.x + 0.5, 0.0, 1.0) : 0.0;
                xwgt = max(xwgt, e1 ? clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0) : 0.0);
                xcov -= e2 ? clamp(r.y + 0.5, 0.0, 1.0) : 0.0;
                xwgt = max(xwgt, e2 ? clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0) : 0.0);
            }
        }
        xcov *= dx;
        vec4 vb = ghee_t_texel(h + nb.x + band.y);
        float dy = u.y < 1023.5 ? -1.0 : 1.0;
        vec2 uv = vec2(u.x, dy > 0.0 ? u.y : 2047.0 - u.y);
        float ycov = 0.0;
        float ywgt = 0.0;
        at = ghee_t_list(vb, dy);
        for (int i = 0; i < 64; i++) {
            if (float(i) >= vb.a) break;
            vec3 c1 = ghee_t_lattice(texture2D(ghee_t_tex, at));
            vec3 c3 = ghee_t_lattice(texture2D(ghee_t_tex, at + vec2(ghee_t_itw, 0.0)));
            at.x += ghee_t_itw * 2.0;
            vec2 p1 = (c1.xy - uv).yx;
            vec2 p3 = (c3.xy - uv).yx;
            vec2 p2 = p3 + vec2(c3.z, c1.z);
            if (max(max(p1.x, p2.x), p3.x) * ppu.y < -0.5) break;
            // Which roots count, by the signs of p1.y, p2.y and p3.y: Slug's
            // lookup table 0x2E74, written out as logic.
            bool n1 = p1.y < 0.0;
            bool n2 = p2.y < 0.0;
            bool n3 = p3.y < 0.0;
            bool e1 = n3 ? !(n1 && n2) : n2 && !n1;
            bool e2 = n3 ? n1 && !n2 : n1 || n2;
            if (e1 || e2) {
                vec2 r = ghee_t_cross(p1, p2, p3) * ppu.y;
                ycov -= e1 ? clamp(r.x + 0.5, 0.0, 1.0) : 0.0;
                ywgt = max(ywgt, e1 ? clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0) : 0.0);
                ycov += e2 ? clamp(r.y + 0.5, 0.0, 1.0) : 0.0;
                ywgt = max(ywgt, e2 ? clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0) : 0.0);
            }
        }
        ycov *= dy;
        cov = clamp(max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
                        min(abs(xcov), abs(ycov))), 0.0, 1.0);
    }
    return cov;
}
// The coverage of the glyph whose data starts at texel h, at q in em from its
// box's corner. Its data holds its box size, then its band entries.
float ghee_t_glyph(float h, vec2 q, vec2 sz, float ppem) {
    return ghee_t_outline(h + 1.0, vec2(ghee_t_nb), q / sz * 2047.0, sz * (ppem / 2047.0));
}
// The coverage of the words at p, in em from the top left of the block, y
// down like v_texCoord, drawn at ppem pixels per em.
float t_cover(vec2 p, float ppem) {
    float cov = 0.0;
    vec2 at = vec2(p.x, t_size.y - p.y);
    vec2 g = (at - vec2(ghee_t_gx, ghee_t_gy)) / ghee_t_cell;
    if (g.x >= 0.0 && g.y >= 0.0 && g.x < ghee_t_cols && g.y < ghee_t_rows) {
        vec2 c = vec2(ivec2(g));
        float cell = ghee_t_cells + (c.y * ghee_t_cols + c.x) * 8.0;
        float row = float(int(cell * ghee_t_itw));
        vec2 slot = vec2(cell - row * ghee_t_tw + 0.5, row + 0.5) * vec2(ghee_t_itw, ghee_t_ith);
        vec2 pc = at - vec2(ghee_t_gx, ghee_t_gy) - c * ghee_t_cell;
        // An empty slot, or the one marked last, ends the cell.
        for (int k = 0; k < 4; k++) {
            vec4 a = vec4(ivec4(texture2D(ghee_t_tex, slot) * 255.5));
            if (a.r + a.g + a.b < 0.5) break;
            vec4 b = vec4(ivec4(texture2D(ghee_t_tex, slot + vec2(ghee_t_itw, 0.0)) * 255.5));
            slot.x += ghee_t_itw * 2.0;
            float last = a.r >= 128.0 ? 1.0 : 0.0;
            float h = ((a.r - last * 128.0) * 256.0 + a.g) * 256.0 + a.b;
            vec2 lo = (b.rb * 256.0 + b.ga) * (1.0 / 8192.0) - 7.0;
            vec4 s = ghee_t_texel(h);
            vec2 sz = (s.rb * 256.0 + s.ga) * (1.0 / 8192.0);
            cov = max(cov, ghee_t_glyph(h, pc - lo, sz, ppem));
            if (last > 0.5) break;
        }
    }
    return cov;
}
uniform float sung;
uniform int n;
uniform vec3 ink;
uniform float gain;

float twice(float x) { return 2.0 * x; }

varying vec4 ghee_v0;
varying vec4 ghee_v1;
varying vec4 ghee_v2;
varying vec4 ghee_v3;
varying vec4 ghee_v4;
// A varying that holds one value at every vertex may still interpolate a hair
// off it, so an index rounds.
#define glyph_em ghee_v2.xy
#define glyph_index float(int(ghee_v2.z + 0.5))
#define glyph_word float(int(ghee_v2.w + 0.5))
#define glyph_line float(int(ghee_v3.x + 0.5))
#define glyph_span ghee_v3.yz
#define glyph_ppem ghee_v1.w
float ghee_c = 0.0;
float glyph_cover() { return ghee_c; }
// The author's main runs as ghee_main, after the fill: the fill computes the
// coverage once a pixel, and glyph_cover() returns it at every call.
#define main ghee_main

void main() {
    float c = glyph_cover() * gain;
    float k = clamp(sung - glyph_index, 0.0, 1.0);
    gl_FragColor = vec4(ink * c * twice(k), c);
}

#undef main
void main() {
    // VC4 lands a vertex up to 1/16 px off the position set by the vertex
    // shader, and the varyings follow the vertex, so the sample moves to the
    // pixel centre given by v_texCoord.
    vec2 ghee_px = v_texCoord * resolution;
    vec2 ghee_d = floor(ghee_px) + 0.5 - ghee_px;
    vec2 ghee_u = ghee_v0.xy + vec2(dot(ghee_v4.xy, ghee_d), dot(ghee_v4.zw, ghee_d));
    ghee_c = ghee_t_outline(float(int(ghee_v1.x + 0.5)), vec2(ivec2(ghee_v1.yz + 0.5)),
                               ghee_u, ghee_v0.zw);
    ghee_main();
}