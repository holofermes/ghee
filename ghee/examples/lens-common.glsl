// The page's height, in its widths. A frame wider than that shows a strip of
// the page and scrolls down it.
const float page_shape = 0.75;

// The rows of the page that the view has scrolled past at time t, a whole
// number that grows on, and the page's height in rows.
vec2 scrolled(float width, float aspect, float t) {
  float rows = floor(page_shape * width + 0.5);
  if (aspect >= page_shape) return vec2(0.0, floor(aspect * width + 0.5));
  return vec2(floor(t * scroll / 60.0 * rows), rows);
}

// The lens at this frame, in widths of the frame: its centre, then its
// radius. Over a scrolling strip it glides across and back at a steady pace.
vec3 lens_now(float aspect) {
  float t = iTime * speed;
  if (aspect < page_shape) return vec3(0.5 + 0.38 * asin(0.97 * sin(t)) / asin(0.97), 0.5 * aspect, 0.42 * aspect);
  return vec3(0.5 + 0.34 * sin(1.2 * t), aspect * (0.5 + 0.3 * sin(2.1 * t + 0.6)), 0.27 * aspect);
}

vec2 turned(vec2 q, float aspect) {
  float a = -0.02;
  vec2 middle = vec2(0.5, 0.5 * aspect);
  return middle + mat2(cos(a), sin(a), -sin(a), cos(a)) * (q - middle);
}

// How hard the glass bends at its rim.
const float bow = 1.3;

// The page point at frame point q, in widths of the frame, scrolled by s
// rows of a page that is `rows` tall.
vec2 page_of(vec2 q, float width, vec2 s) {
  q.y = mod(q.y * width + s.x, s.y) / width;
  return turned(q, s.y / width);
}

// The page point that the lens shows at q, and its magnification, the
// smaller of its two axes. Magnification is flat to mid radius, and toward
// the rim the glass bends harder and crowds the page into a ring.
vec2 seen_at(vec2 q, vec3 lens, float width, vec2 s, out float m) {
  vec2 d = q - lens.xy;
  float rho = length(d) / lens.z;
  m = 1.0;
  if (rho >= 1.0) return page_of(q, width, s);
  float r4 = rho * rho * rho * rho;
  float r = rho / zoom + (1.0 - 1.0 / zoom) * bow * (1.0 - sqrt(1.0 - r4));
  float ds = 1.0 / zoom + (1.0 - 1.0 / zoom) * bow * 2.0 * rho * rho * rho / sqrt(max(1.0 - r4, 0.0004));
  m = min(zoom, min(rho / max(r, 0.0001), 1.0 / ds));
  return page_of(lens.xy + d * (r / max(rho, 0.0001)), width, s);
}
