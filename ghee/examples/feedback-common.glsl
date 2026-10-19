// Positions are in the picture, 0 to 1, x across and y down. The flow and the
// strokes work in heights, positions times (ASPECT, 1), so a swirl keeps its
// shape at any frame size.
#define ASPECT (iOutputResolution.x / iOutputResolution.y)

float hash(vec2 p) {
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}

float waterline(float x) {
  float t = iTime * wave_speed * 6.2831853;
  return water_level - wave_height * (0.28 * sin(x * 4.4 + t) + 0.10 * sin(x * 9.7 - t * 0.6));
}

float depth(vec2 p) {
  return p.y - waterline(p.x);
}

// A buffer is RGB8, so a position takes three bytes: twelve bits per axis
// over [-0.25, 1.25], the high byte of each in R and G and both low nibbles
// in B.
vec3 pack_pos(vec2 p) {
  vec2 v = floor(clamp((p + 0.25) / 1.5, 0.0, 1.0) * 4095.0 + 0.5);
  vec2 hi = floor(v / 16.0);
  vec2 lo = v - hi * 16.0;
  return vec3(hi / 255.0, (lo.x * 16.0 + lo.y) / 255.0);
}

vec2 unpack_pos(vec3 e) {
  vec3 b = floor(e * 255.0 + 0.5);
  float lx = floor(b.z / 16.0);
  return (b.xy * 16.0 + vec2(lx, b.z - lx * 16.0)) * (1.5 / 4095.0) - 0.25;
}

// The high bytes alone: the position, short by up to one low nibble.
vec2 coarse_pos(vec3 e) {
  return e.xy * (4080.0 * 1.5 / 4095.0) - 0.25;
}

// Each particle lives this share of `lifetime`, from 0.4 to 1.
float lifespan(vec2 cell) {
  return mix(0.4, 1.0, hash(cell * 5.1));
}

// An age is two bytes, high in R and low in G.
float unpack_age(vec2 e) {
  return (floor(e.x * 255.0 + 0.5) + e.y) / 255.0;
}
