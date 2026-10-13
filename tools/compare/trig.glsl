// sin and cos to float precision on every GPU. VC4's own evaluate one
// degree-9 series over a whole turn, off by up to 7e-3 at its ends, which
// moves floors and stripes that Python's double puts elsewhere.

vec2 sincos(float x) {
  float q = floor(x * 0.6366197723675814 + 0.5);
  // pi / 2 in two parts, the first exact in 8 bits
  float r = (x - q * 1.5703125) - q * 4.838267948966e-4;
  float r2 = r * r;
  float s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0
    + r2 * (1.0 / 362880.0)))));
  float c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0
    + r2 * (1.0 / 40320.0 - r2 * (1.0 / 3628800.0)))));
  float m = q - 4.0 * floor((q + 0.5) / 4.0);
  if (m == 0.0) return vec2(s, c);
  if (m == 1.0) return vec2(c, -s);
  if (m == 2.0) return vec2(-s, -c);
  return vec2(-c, s);
}

float sine(float x) {
  return sincos(x).x;
}

float cosine(float x) {
  return sincos(x).y;
}
