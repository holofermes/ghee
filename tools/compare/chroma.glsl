// luma.examples' chroma.py: a tunnel, a swirl, a roto-zooming checker board
// and interfering sine blobs, 500 steps each, the last 100 a crossfade into
// the next. One sample covers a 2x2 block, as chroma.py samples at half the
// panel's size. The step is chroma.py's frame counter.
//
// The tunnel's and the checker's colours land on whole numbers at many
// steps, where float and Python's double floor to different sides. Outside
// a crossfade and the tunnel's dark centre, they come from the palette: one
// row a step, chroma.py's own colours as it floors them.

const float PI = 3.141592653589793;

bool whole;

// x mod n for a whole x
float wrap(float x, float n) {
  return x - n * floor((x + 0.5) / n);
}

vec3 palette(float column, float step) {
  vec2 at = (vec2(column, wrap(step, 255.0)) + 0.5) / palette_size;
  return floor(texture2D(palette_tex, at).rgb * 255.0 + 0.5);
}

// colorsys.hsv_to_rgb at full saturation
vec3 hsv(float h, float v) {
  float i = floor(h * 6.0);
  float f = h * 6.0 - i;
  float q = v * (1.0 - f);
  float t = v * (1.0 - (1.0 - f));
  i = mod(i, 6.0);
  if (i == 0.0) return vec3(v, t, 0.0);
  if (i == 1.0) return vec3(q, v, 0.0);
  if (i == 2.0) return vec3(0.0, v, t);
  if (i == 3.0) return vec3(0.0, q, v);
  if (i == 4.0) return vec3(t, 0.0, v);
  return vec3(v, 0.0, q);
}

vec3 tunnel(vec2 p, float step) {
  float speed = step / 100.0;
  float x = p.x - iResolution.x / 4.0 + sine(step / 27.0) * 2.0;
  float y = p.y - iResolution.y / 4.0 + cosine(step / 18.0) * 2.0;
  float angle;
  if (y == 0.0) {
    angle = x < 0.0 ? -(PI / 2.0) : PI / 2.0;
  } else {
    angle = atan(x / y);
  }
  if (y > 0.0) angle += PI;
  angle /= 2.0 * PI;
  float r = sqrt(x * x + y * y);
  float shade = min(r / 2.1, 1.0);
  angle += speed;
  float depth = speed + r / 10.0;
  bool bright = mod(floor(abs(angle * 6.0)), 2.0) == 0.0;
  bool ring = mod(floor(abs(depth * 3.0)), 2.0) == 0.0;
  if (whole && shade == 1.0) {
    return palette((bright ? 0.0 : 2.0) + (ring ? 1.0 : 0.0), step);
  }
  float h = wrap(step, 255.0) / 255.0;
  vec3 col = bright ? hsv(h, 0.8) : hsv(h, 0.3);
  float td = ring ? 0.3 : 0.0;
  return (col + td) * shade * 255.0;
}

vec3 swirl(vec2 p, float step) {
  float x = p.x - iResolution.x / 4.0;
  float y = p.y - iResolution.y / 4.0;
  float dist = sqrt(x * x + y * y) / 2.0;
  float angle = step / 10.0 + dist * 1.5;
  float s = sine(angle);
  float c = cosine(angle);
  float xs = x * c - y * s;
  float ys = x * s + y * c;
  float r = abs(xs + ys) * 64.0 - 20.0;
  return vec3(r, r + s * 130.0, r + c * 130.0);
}

vec3 checker(vec2 p, float step) {
  float x = p.x - iResolution.x / 4.0;
  float y = p.y - iResolution.y / 4.0;
  float s = sine(step / 10.0);
  float c = cosine(step / 10.0);
  float xs = x * c - y * s - sine(step / 200.0) * 40.0;
  float ys = x * s + y * c - cosine(step / 200.0) * 40.0;
  float scale = sine(step / 50.0) / 8.0 + 0.25;
  xs *= scale;
  ys *= scale;
  if (mod(floor(xs) + floor(ys), 2.0) != 0.0) return vec3(0.0);
  bool lit = fract(abs(xs)) > 0.1 && fract(abs(ys)) > 0.1;
  if (whole) return palette(lit ? 5.0 : 4.0, step);
  return hsv(wrap(step, 255.0) / 255.0, lit ? 1.0 : 0.5) * 255.0;
}

vec3 rainbow(vec2 p, float step) {
  float xs = sine(step / 100.0) * 20.0;
  float ys = cosine(step / 100.0) * 20.0;
  float scale = (sine(step / 60.0) + 1.0) / 5.0 + 0.2;
  float r = sine((p.x + xs) * scale) + cosine((p.y + xs) * scale);
  float g = sine((p.x + xs) * scale) + cosine((p.y + ys) * scale);
  float b = sine((p.x + ys) * scale) + cosine((p.y + ys) * scale);
  return vec3(r, g, b) * 255.0;
}

vec3 effect(float k, vec2 p, float step) {
  if (k == 0.0) return tunnel(p, step);
  if (k == 1.0) return swirl(p, step);
  if (k == 2.0) return checker(p, step);
  return rainbow(p, step);
}

void main() {
  vec2 p = floor(uv * iResolution.xy * 0.5);
  float step = float(iFrame) + first;
  float i = wrap(step, 500.0);
  float k = wrap((step - i) / 500.0, 4.0);
  whole = i <= 400.0;
  vec3 c = effect(k, p, step);
  if (i > 400.0) {
    float ratio = (500.0 - i) / 100.0;
    c = c * ratio + effect(wrap(k + 1.0, 4.0), p, step) * (1.0 - ratio);
  }
  gl_FragColor = vec4(floor(clamp(c, 0.0, 255.0)) / 255.0, 1.0);
}
