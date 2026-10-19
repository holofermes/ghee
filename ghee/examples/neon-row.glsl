// The street lays out from the frame's top: a frame point `at` runs from 0
// to 1, y down, as `base` does.
//
// The row of shopfronts, in row heights: x across the frame from its middle,
// y up from the pavement. The row stands as tall as the wall in a wide
// frame, and in a narrower one a third of the frame's width, under the
// upper floors.
float row_height() {
  return min(base - 0.05, 0.3 * iResolution.x / iResolution.y);
}

vec2 row_at(vec2 at) {
  return vec2((at.x - 0.5) * iResolution.x / iResolution.y, base - at.y) / row_height();
}

// The shop under a point of the row. x is its number: 0 for the ramen shop
// in the middle, 1, 2... to its right and -1, -2... to its left. y is the
// point's distance from the shop's left edge, and z the shop's width.
vec3 shop_at(vec2 s) {
  float n = ceil(max(abs(s.x) - 1.1, 0.0) / 0.9);
  float ramen = 1.0 - clamp(n, 0.0, 1.0);
  float side = s.x / max(abs(s.x), 0.0001);
  float left = mix(side * (1.1 + (n - 0.5) * 0.9) - 0.45, -1.1, ramen);
  return vec3(side * n, s.x - left, mix(0.9, 2.2, ramen));
}

float box(vec2 p, vec2 lo, vec2 hi) {
  vec2 d = max(lo - p, p - hi);
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// The ticker's board under the ramen sign, 56 by 8 LEDs, each a whole
// number of pixels wide.
float led_pitch() {
  return max(1.0, floor(0.0268 * row_height() * iResolution.y + 0.5));
}

vec2 led_at(vec2 at) {
  float top = floor((base - 0.62 * row_height()) * iResolution.y);
  return vec2((at.x - 0.5) * iResolution.x / led_pitch() + 28.0, (at.y * iResolution.y - top) / led_pitch());
}
