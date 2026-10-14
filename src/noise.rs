//! `ghee_hash` and `ghee_noise`, the GLSL functions that ghee writes into a
//! pass whose code calls them.
//!
//! `ghee_hash(p)` takes a float, a vec2 or a vec3 to a value in [0, 1), the
//! same bits on every GPU. A hash that rounds anywhere differs between
//! GPUs: one fuses a multiply into the add after it, another rewrites an
//! expression, and VC4's multiply does not round to nearest. So every value
//! here is an integer or a binary fraction that a float holds exactly, and
//! no operation rounds.
//!
//! Each coordinate x becomes round(2048 x) mod 2^23, a fraction on a 2^-23
//! grid: x read to the nearest 1/2048, repeating every 4096. A whole number,
//! a half or a pixel centre's uv that a GPU computes a unit in the last place
//! off reads the same. A float input also takes its multiple of 4096 as a
//! second part, so it does not repeat. A part may be 1, which the sums take
//! as 0.
//!
//! The state is such a fraction. A round splits it into its top 11 bits `a`
//! and its low 12 bits `b`, and returns `b` shifted up by 11 bits plus
//! a(2a + 3) + 1031, modulo 2^23. A round is a bijection, since a(2a + 3)
//! modulo 2^11 is a permutation. The state starts as the first part plus a
//! constant of its own for a float, a vec2 and a vec3. Each later part joins
//! it by a sum modulo 1 after a round, and four rounds end the hash. `step`
//! takes 1 off a sum in [1, 2).
//!
//! A compiler may sum the terms of a sum in any order, so every sum holds
//! terms of magnitude under 2 on the 2^-23 grid, exact in any order, and a
//! value of x's own size meets a sum only behind a select or a multiply.
//! `q - floor(q)` rounds for q in (-1, 0), and only a comparison with 0.5
//! reads it. VC4's floor fails from 2^31 on, where every float is whole, so
//! `ghee_hash_floor` and `ghee_hash_fract` take such a value as whole.
//!
//! `ghee_noise(p)` blends the hashes of the four or eight lattice points
//! around `p` with smoothstep's weights, a value in [0, 1]. Its lattice
//! values are those of `ghee_hash`, and its blend rounds as each GPU rounds.

/// The functions of `ghee_hash`, with the helpers that `ghee_noise` shares.
const HASH: &str = r#"float ghee_hash_wrap(float v) { return v - step(1.0, v); }
vec2 ghee_hash_wrap(vec2 v) { return v - step(1.0, v); }
vec4 ghee_hash_wrap(vec4 v) { return v - step(1.0, v); }
float ghee_hash_round(float s) {
  float a = floor(s * 2048.0);
  float b = (s - a * 4.8828125e-4) * 2048.0;
  return ghee_hash_wrap(b + (a * (a * 2.0 + 3.0) + 1031.0) * 1.1920928955078125e-7);
}
vec2 ghee_hash_round(vec2 s) {
  vec2 a = floor(s * 2048.0);
  vec2 b = (s - a * 4.8828125e-4) * 2048.0;
  return ghee_hash_wrap(b + (a * (a * 2.0 + 3.0) + 1031.0) * 1.1920928955078125e-7);
}
vec4 ghee_hash_round(vec4 s) {
  vec4 a = floor(s * 2048.0);
  vec4 b = (s - a * 4.8828125e-4) * 2048.0;
  return ghee_hash_wrap(b + (a * (a * 2.0 + 3.0) + 1031.0) * 1.1920928955078125e-7);
}
float ghee_hash_mix(float s) { return ghee_hash_round(ghee_hash_round(ghee_hash_round(ghee_hash_round(s)))); }
vec4 ghee_hash_mix(vec4 s) { return ghee_hash_round(ghee_hash_round(ghee_hash_round(ghee_hash_round(s)))); }
float ghee_hash_floor(float v) { return abs(v) < 8388608.0 ? floor(v) : v; }
vec3 ghee_hash_floor(vec3 v) { return vec3(ghee_hash_floor(v.x), ghee_hash_floor(v.y), ghee_hash_floor(v.z)); }
float ghee_hash_fract(float v) { return abs(v) < 8388608.0 ? v - floor(v) : 0.0; }
vec3 ghee_hash_cell(vec3 p) {
  vec3 q = p * 2048.0;
  vec3 x = ghee_hash_floor(q);
  vec3 u = x * 1.1920928955078125e-7;
  vec3 c = vec3(ghee_hash_fract(u.x), ghee_hash_fract(u.y), ghee_hash_fract(u.z));
  return c + step(0.5, q - x) * 1.1920928955078125e-7;
}
float ghee_hash(float x) {
  float q = x * 2048.0;
  float f = ghee_hash_floor(q);
  float u = f * 1.1920928955078125e-7;
  float c = ghee_hash_fract(u) + step(0.5, q - f) * 1.1920928955078125e-7;
  float hi = ghee_hash_fract(ghee_hash_floor(u) * 1.1920928955078125e-7) + step(1.0, c) * 1.1920928955078125e-7;
  float s = ghee_hash_round(ghee_hash_wrap(c + 0.6180267333984375));
  return ghee_hash_mix(ghee_hash_wrap(s + hi));
}
float ghee_hash(vec2 p) {
  vec3 c = ghee_hash_cell(vec3(p, 0.0));
  return ghee_hash_mix(ghee_hash_wrap(ghee_hash_round(ghee_hash_wrap(c.x + 0.4142303466796875)) + c.y));
}
float ghee_hash(vec3 p) {
  vec3 c = ghee_hash_cell(p);
  float s = ghee_hash_wrap(ghee_hash_round(ghee_hash_wrap(c.x + 0.1415863037109375)) + c.y);
  return ghee_hash_mix(ghee_hash_wrap(ghee_hash_round(s) + c.z));
}
"#;

/// The functions of `ghee_noise`. A lattice point one further along an axis
/// is the cell 2^-12 further on, wrapped.
const NOISE: &str = r#"float ghee_noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = p - i;
  vec2 c = ghee_hash_cell(vec3(i, 0.0)).xy;
  vec2 e = ghee_hash_wrap(c + 0.000244140625);
  vec2 x = ghee_hash_round(ghee_hash_wrap(vec2(c.x, e.x) + 0.4142303466796875));
  vec4 h = ghee_hash_mix(ghee_hash_wrap(x.xyxy + vec4(c.yy, e.yy)));
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h.x, h.y, f.x), mix(h.z, h.w, f.x), f.y);
}
float ghee_noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = p - i;
  vec3 c = ghee_hash_cell(i);
  vec3 e = ghee_hash_wrap(vec4(c, 0.0) + 0.000244140625).xyz;
  vec2 x = ghee_hash_round(ghee_hash_wrap(vec2(c.x, e.x) + 0.1415863037109375));
  vec4 y = ghee_hash_round(ghee_hash_wrap(x.xyxy + vec4(c.yy, e.yy)));
  vec4 h0 = ghee_hash_mix(ghee_hash_wrap(y + c.z));
  vec4 h1 = ghee_hash_mix(ghee_hash_wrap(y + e.z));
  f = f * f * (3.0 - 2.0 * f);
  vec4 h = mix(h0, h1, f.z);
  return mix(mix(h.x, h.y, f.x), mix(h.z, h.w, f.x), f.y);
}
"#;

/// The lines that a pass's code `tokens` needs: none, the hash, or the hash
/// and the noise.
pub(crate) fn lines(tokens: &[&str]) -> Vec<String> {
    let noise = tokens.contains(&"ghee_noise");
    if !noise && !tokens.contains(&"ghee_hash") {
        return Vec::new();
    }
    let mut l: Vec<String> = HASH.lines().map(str::to_string).collect();
    if noise {
        l.extend(NOISE.lines().map(str::to_string));
    }
    l
}
