// The lens at this frame, in widths of the frame: its centre, then its
// radius.
vec3 lens_now(float aspect) {
  float t = iTime * speed;
  return vec3(0.5 + 0.34 * sin(1.2 * t), aspect * (0.5 + 0.3 * sin(2.1 * t + 0.6)), 0.27 * aspect);
}

vec2 turned(vec2 q, float aspect) {
  float a = -0.02;
  vec2 middle = vec2(0.5, 0.5 * aspect);
  return middle + mat2(cos(a), sin(a), -sin(a), cos(a)) * (q - middle);
}

// How hard the glass bends at its rim.
const float bow = 1.3;

// The page point that the lens shows at q, and its magnification, the
// smaller of its two axes. Magnification is flat to mid radius, and toward
// the rim the glass bends harder and crowds the page into a ring.
vec2 seen_at(vec2 q, vec3 lens, float aspect, out float m) {
  vec2 d = q - lens.xy;
  float rho = length(d) / lens.z;
  m = 1.0;
  if (rho >= 1.0) return turned(q, aspect);
  float r4 = rho * rho * rho * rho;
  float s = rho / zoom + (1.0 - 1.0 / zoom) * bow * (1.0 - sqrt(1.0 - r4));
  float ds = 1.0 / zoom + (1.0 - 1.0 / zoom) * bow * 2.0 * rho * rho * rho / sqrt(max(1.0 - r4, 0.0004));
  m = min(zoom, min(rho / max(s, 0.0001), 1.0 / ds));
  return turned(lens.xy + d * (s / max(rho, 0.0001)), aspect);
}
