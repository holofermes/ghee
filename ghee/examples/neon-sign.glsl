float seg(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

float half_ring(vec2 p, vec2 c, float r, vec2 side) {
  vec2 q = p - c;
  vec2 e = vec2(side.y, -side.x) * r;
  return dot(q, side) > 0.0 ? abs(length(q) - r) : min(length(q - e), length(q + e));
}

float letter_r(vec2 p) {
  float d = seg(p, vec2(0.0, 0.0), vec2(0.0, 1.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.27, 0.0)));
  d = min(d, half_ring(p, vec2(0.27, 0.25), 0.25, vec2(1.0, 0.0)));
  d = min(d, seg(p, vec2(0.0, 0.5), vec2(0.27, 0.5)));
  return min(d, seg(p, vec2(0.22, 0.5), vec2(0.55, 1.0)));
}

float letter_a(vec2 p) {
  float d = seg(p, vec2(0.0, 1.0), vec2(0.3, 0.0));
  d = min(d, seg(p, vec2(0.3, 0.0), vec2(0.6, 1.0)));
  return min(d, seg(p, vec2(0.11, 0.64), vec2(0.49, 0.64)));
}

float letter_m(vec2 p) {
  float d = seg(p, vec2(0.0, 1.0), vec2(0.0, 0.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.36, 0.62)));
  d = min(d, seg(p, vec2(0.36, 0.62), vec2(0.72, 0.0)));
  return min(d, seg(p, vec2(0.72, 0.0), vec2(0.72, 1.0)));
}

float letter_e(vec2 p) {
  float d = seg(p, vec2(0.0, 0.0), vec2(0.0, 1.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.46, 0.0)));
  d = min(d, seg(p, vec2(0.0, 0.5), vec2(0.38, 0.5)));
  return min(d, seg(p, vec2(0.0, 1.0), vec2(0.46, 1.0)));
}

float letter_n(vec2 p) {
  float d = seg(p, vec2(0.0, 1.0), vec2(0.0, 0.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.58, 1.0)));
  return min(d, seg(p, vec2(0.58, 1.0), vec2(0.58, 0.0)));
}

float cup(vec2 p) {
  float d = half_ring(p, vec2(0.0, 0.42), 0.62, vec2(0.0, 1.0));
  d = min(d, seg(p, vec2(-0.62, 0.42), vec2(0.62, 0.42)));
  return min(d, seg(p, vec2(-0.22, 1.16), vec2(0.22, 1.16)));
}

float sticks(vec2 p) {
  float d = seg(p, vec2(0.28, 0.3), vec2(0.98, -0.48));
  return min(d, seg(p, vec2(0.42, 0.34), vec2(1.1, -0.4)));
}

float wisp(vec2 p, float x) {
  float y = clamp(p.y, -0.7, 0.16);
  return length(vec2(p.x - x - 0.07 * sin(y * 7.0 + 1.0), p.y - y));
}

// A character of a neighbour's sign: a bar or a stem, and the strokes h picks
// from five more, in a cell f from 0 to 1, y down.
float glyph(vec2 f, float h) {
  float n = floor(h * 64.0);
  float d = mix(seg(f, vec2(0.5, 0.1), vec2(0.5, 0.9)), seg(f, vec2(0.15, 0.5), vec2(0.85, 0.5)), mod(n, 2.0));
  d = min(d, seg(f, vec2(0.15, 0.15), vec2(0.85, 0.15)) + 9.0 * (1.0 - mod(floor(n / 2.0), 2.0)));
  d = min(d, seg(f, vec2(0.85, 0.15), vec2(0.65, 0.9)) + 9.0 * (1.0 - mod(floor(n / 4.0), 2.0)));
  d = min(d, seg(f, vec2(0.35, 0.4), vec2(0.12, 0.88)) + 9.0 * (1.0 - mod(floor(n / 8.0), 2.0)));
  d = min(d, seg(f, vec2(0.2, 0.85), vec2(0.8, 0.85)) + 9.0 * (1.0 - mod(floor(n / 16.0), 2.0)));
  return min(d, seg(f, vec2(0.15, 0.15), vec2(0.15, 0.6)) + 9.0 * (1.0 - mod(floor(n / 32.0), 2.0)));
}

// A line of characters along a box, across it or down it, with as many
// square cells as fit.
float sign_text(vec2 p, vec2 lo, vec2 hi, float seed) {
  vec2 size = hi - lo;
  float down = step(size.x, size.y);
  float cell = min(size.x, size.y);
  vec2 g = (p - lo) / cell;
  vec2 f = vec2(fract(mix(g.x, g.y, down)), 0.0);
  f = mix(vec2(f.x, 1.0 - g.y), vec2(g.x, f.x), down);
  float i = floor(mix(g.x, g.y, down));
  return glyph(0.2 + 0.6 * f, ghee_hash(vec2(i, seed))) * cell * 0.6 + 9.0 * step(0.0, box(p, lo, hi));
}

vec4 tube(float d, float w, float px, vec3 color, float on) {
  float body = 1.0 - smoothstep(w - px, w + px, d);
  float core = exp(-d * d / (0.27 * w * w));
  return body * vec4(color * mix(0.1, 1.0, on), core * on);
}

vec4 lit(float d, float px, vec3 color) {
  return vec4(color, 0.0) * (1.0 - smoothstep(-px, px, d));
}

vec3 shop_color(float n) {
  float i = mod(n, 4.0);
  return mix(mix(shop_1, shop_2, step(0.5, i)), mix(shop_3, shop_4, step(2.5, i)), step(1.5, i));
}

vec4 neighbour(vec2 s, vec3 b, float px) {
  vec2 p = vec2(b.y, s.y);
  vec3 c = shop_color(b.x);
  float h = ghee_hash(vec2(b.x, 3.0));
  float kind = floor(h * 4.0);
  float flick = step(0.8, fract(h * 13.0)) * step(0.85, fract(iTime / 5.0 + h)) * step(0.5, ghee_hash(floor(iTime * 20.0) + b.x));
  vec4 col = vec4(0.0);
  float w = 0.014;

  vec2 lo = vec2(0.1, 0.7);
  vec2 hi = vec2(0.8, 0.92);
  if (p.y > 0.62) {
    float frame = box(p, lo, hi);
    float text = sign_text(p, lo + vec2(0.04, 0.0), hi - vec2(0.04, 0.0), b.x);
    if (fract(h * 7.0) < 0.5) {
      col += tube(abs(frame), w, px, c, 1.0 - flick);
      col += tube(text, w, px, mix(c, vec3(1.0), 0.35), 1.0 - flick);
    } else {
      col += lit(frame, px, mix(c, vec3(1.0), 0.3) * 0.7) * smoothstep(0.015, 0.03, text);
      col += tube(abs(frame), w * 0.6, px, c, 1.0);
    }
  } else if (kind == 0.0) {
    float glass = box(p, vec2(0.08, 0.05), vec2(0.82, 0.6));
    float mull = abs(fract((p.x - 0.08) / 0.74 * 3.0) - 0.5) * 0.25 - 0.115;
    vec3 inside = mix(c, rooms, 0.6) * (0.1 + 0.25 * smoothstep(0.3, 0.6, p.y));
    col += lit(max(glass, -mull), px, inside);
    col += tube(abs(p.y - 0.57), 0.008, px, vec3(1.0), 1.0) * step(glass, 0.0);
  } else if (kind == 1.0) {
    col += lit(box(p, vec2(0.08, 0.03), vec2(0.82, 0.055)), px, c * 0.9);
  } else if (kind == 2.0) {
    col += lit(box(p, vec2(0.08, 0.05), vec2(0.82, 0.58)), px, c * 0.07);
    vec2 a = vec2(0.24, 0.3);
    vec2 z = vec2(0.66, 0.42);
    col += tube(abs(box(p, a, z)), w * 0.8, px, c, 1.0);
    col += tube(sign_text(p, a + vec2(0.02, 0.0), z - vec2(0.02, 0.0), b.x + 40.0), w * 0.8, px, vec3(1.0), 1.0);
  } else {
    float door = box(p, vec2(0.3, 0.03), vec2(0.6, 0.55));
    col += lit(door, px, rooms * 0.3);
    col += tube(abs(door - 0.03), w, px, c, 1.0);
  }

  if (kind != 1.0) {
    float i = floor((p.x - 0.1) / 0.07);
    float on = step(0.4, fract(i / 3.0 - iTime * 1.5));
    float bulb = length(vec2(fract((p.x - 0.1) / 0.07) - 0.5, (p.y - 0.645) / 0.07)) * 0.07;
    col += tube(bulb + 9.0 * step(0.36, abs(p.x - 0.45)), 0.012, px, rooms, on);
  }

  if (fract(h * 29.0) < 0.6 && p.x > 0.7) {
    vec2 vlo = vec2(0.78, 0.25);
    vec2 vhi = vec2(0.88, 1.35);
    float v = box(p, vlo, vhi);
    vec3 vc = shop_color(b.x + 2.0);
    vec4 vs = lit(v, px, mix(vc, vec3(1.0), 0.3) * 0.6) * smoothstep(0.008, 0.02, sign_text(p, vlo + vec2(0.0, 0.02), vhi, b.x + 80.0));
    vs += tube(abs(v), 0.008, px, vc, 1.0);
    col = mix(col, vs, 1.0 - smoothstep(-px, px, v - 0.01));
  }
  return col;
}

vec4 upstairs(vec2 s, vec3 b, float px) {
  float floors = (s.y - 1.08) / 0.55;
  float storey = floor(floors);
  float across = max(floor(b.z / 0.45), 1.0);
  vec2 f = vec2(fract(b.y / b.z * across), fract(floors));
  float i = floor(b.y / b.z * across);
  float h = ghee_hash(vec2(b.x * 7.0 + i, storey));
  float pane = box(f, vec2(0.3, 0.3), vec2(0.7, 0.75)) * 0.45;
  float blind = 0.6 + 0.4 * smoothstep(0.3, 0.6, fract(f.y * 9.0)) * step(0.8, fract(h * 5.0));
  vec3 room = mix(rooms, shop_color(h * 9.0), step(0.85, fract(h * 3.0)) * 0.6) * (0.04 + 0.12 * fract(h * 11.0)) * blind * step(0.55, h) * step(0.0, floors);
  return lit(pane, px, room);
}

void main() {
  vec2 s = row_at(uv);
  vec3 b = shop_at(s);
  float px = 1.0 / (row_height() * iResolution.y);
  vec4 col = vec4(0.0);
  if (s.y > 1.05) col = upstairs(s, b, px);

  if (b.x != 0.0) {
    if (s.y < 1.4) col += neighbour(s, b, px);
  } else {
    float size = 0.2;
    vec2 q = vec2(s.x / size + 0.73, (0.86 - s.y) / size);
    float buzz = step(0.55, fract(iTime / 6.0)) * step(fract(iTime / 6.0), 0.68);
    float flicker = 1.0 - buzz * step(0.4, ghee_hash(floor(iTime * 24.0)));
    float tw = 0.075 * size;
    float tpx = px;
    float d = min(min(letter_r(q), letter_a(q - vec2(0.85, 0.0))), letter_m(q - vec2(1.75, 0.0)));
    col += tube(min(d, letter_n(q - vec2(3.46, 0.0))) * size, tw, tpx, word, 1.0);
    col += tube(letter_e(q - vec2(2.75, 0.0)) * size, tw, tpx, word, flicker);

    vec2 bq = q - vec2(-1.95, 0.0);
    col += tube(cup(bq) * size, tw, tpx, bowl, 1.0);
    col += tube(sticks(bq) * size, tw, tpx, chopsticks, 1.0);
    float step_on = floor(fract(iTime / 2.4) * 4.0);
    for (int i = 0; i < 3; i++) {
      float on = step(float(i) + 0.5, step_on);
      col += tube(wisp(bq, -0.42 + 0.3 * float(i)) * size, tw, tpx, steam, on);
    }

    float door = box(s, vec2(-0.5, 0.03), vec2(0.5, 0.36));
    float noren = step(0.22, s.y) * step(0.025, abs(fract(s.x / 0.333 + 0.5) - 0.5));
    float lamp = 1.0 - smoothstep(0.0, 0.3, s.y - 0.03);
    col += lit(door, px, rooms * mix(0.12 + 0.2 * lamp, 0.03, noren));
    vec2 l = vec2(abs(s.x) - 0.68, s.y - 0.25);
    float lantern = length(l / vec2(0.075, 0.1)) - 1.0;
    float ribs = 0.75 + 0.25 * cos(l.y * 140.0);
    col += lit(lantern * 0.075, px, lanterns * ribs) + vec4(0.0, 0.0, 0.0, 0.5 * exp(-dot(l, l) * 900.0));
  }

  float below = uv.y - base;
  if (below > 0.0) {
    float aspect = iResolution.x / iResolution.y;
    float x = (uv.x - 0.5) * aspect;
    float far = (below - 0.07) / 0.008;
    float near = (below - 0.17) / 0.013;
    far = exp(-far * far);
    near = exp(-near * near);
    float a = (x - iTime * cars) / 2.5;
    float k = floor(a);
    float run = smoothstep(0.0, 0.5, fract(a)) * (1.0 - smoothstep(0.55, 0.6, fract(a))) * step(0.35, ghee_hash(vec2(k, 1.0)));
    float r = (x + iTime * cars * 1.3) / 3.1;
    float back = smoothstep(0.0, 0.1, fract(r)) * (1.0 - smoothstep(0.3, 0.7, fract(r))) * step(0.3, ghee_hash(vec2(floor(r), 2.0)));
    col += vec4(headlights * far * run + taillights * near * back, 0.5 * (far * run + near * back));
  }

  vec2 led = led_at(uv);
  if (led.x > 0.0 && led.x < 56.0 && led.y > 0.0 && led.y < 8.0) {
    float dot_mask = mix(1.0, 1.0 - smoothstep(0.3, 0.42, length(fract(led) - 0.5)), smoothstep(1.5, 2.5, led_pitch()));
    float on = 0.0;
    if (gl_FragCoord.x >= 0.0) on = texture2D(iChannel0, led / vec2(56.0, 8.0)).a;
    col += dot_mask * vec4(leds * mix(0.1, 1.0, on), 0.6 * on);
  }
  gl_FragColor = col;
}
