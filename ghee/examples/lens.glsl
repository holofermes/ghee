// A lens drifts over the front page of an old newspaper, magnifies it up to
// `zoom` times and moves at `speed`. The page is set from hidden text layers,
// so this shader alone shows their words: each pixel finds the point of the
// page that the lens shows there, and the _cover of the layer under that point
// reads its words at the size where the glass sets them down, so an edge stays
// one pixel soft at every magnification. Toward its rim the glass bends
// harder, crowds the words into a ring and splits their colours.
//
//     ghee render --demo lens -o frame.####.png --frames 100 --timestep 0.05
//     ghee render --demo lens --serve

float hash(vec2 p) {
  vec3 q = fract(p.xyx * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
  return 0.5 * noise(p) + 0.3 * noise(p * 2.1 + 5.2) + 0.2 * noise(p * 4.3 + 1.7);
}

float rule(vec2 q, float y, float x0, float x1, float half_width, float px) {
  float across = clamp((half_width - abs(q.y - y)) * px + 0.5, 0.0, 1.0);
  return across * clamp((q.x - x0) * px + 0.5, 0.0, 1.0) * clamp((x1 - q.x) * px + 0.5, 0.0, 1.0);
}

// The ink at page point q, in widths of the frame, with px pixels to the width.
float ink_at(vec2 q, float px) {
  float x0 = 0.045;
  float x1 = 0.955;
  float gutter = 0.022;
  float column = (x1 - x0 - 2.0 * gutter) / 3.0;

  float top = 0.03;
  float mast_em = 0.58 / masthead_size.x;
  vec2 mast = vec2(0.5 - 0.29, top + 0.012);
  float mast_bottom = mast.y + masthead_size.y * mast_em;
  float ear_em = 0.0115;
  float ear_y = 0.5 * (mast.y + mast_bottom) - 0.5 * ear_left_size.y * ear_em;
  float double_rule = mast_bottom + 0.008;
  float date_em = 0.0125;
  vec2 date = vec2(0.5 - 0.5 * dateline_size.x * date_em, double_rule + 0.012);
  float date_bottom = date.y + dateline_size.y * date_em;
  float head_em = 0.855 / headline_size.x;
  vec2 head = vec2(0.5 - 0.4275, date_bottom + 0.014);
  float head_bottom = head.y + headline_size.y * head_em;
  float deck_em = 0.6 / deck_size.x;
  vec2 deck_at = vec2(0.2, head_bottom + 0.002);
  float deck_bottom = deck_at.y + deck_size.y * deck_em;
  float body_em = column / 14.2;
  float body_top = deck_bottom + 0.026;

  float c = 0.0;
  if (q.y < double_rule) {
    if (q.x < mast.x) {
      vec2 at = vec2(x0 + 0.01, ear_y);
      c = ear_left_cover((q - at) / ear_em, ear_em * px);
    } else if (q.x > 1.0 - mast.x) {
      vec2 at = vec2(x1 - 0.01 - ear_right_size.x * ear_em, ear_y);
      c = ear_right_cover((q - at) / ear_em, ear_em * px);
    } else {
      c = masthead_cover((q - mast) / mast_em, mast_em * px);
    }
  } else if (q.y < date_bottom + 0.007) {
    c = dateline_cover((q - date) / date_em, date_em * px);
  } else if (q.y < head_bottom) {
    c = headline_cover((q - head) / head_em, head_em * px);
  } else if (q.y < body_top - 0.01) {
    c = deck_cover((q - deck_at) / deck_em, deck_em * px);
  } else {
    float k = clamp(floor((q.x - x0) / (column + gutter)), 0.0, 2.0);
    vec2 at = vec2(x0 + k * (column + gutter), body_top);
    vec2 p = (q - at) / body_em;
    if (k < 0.5) c = column1_cover(p, body_em * px);
    else if (k < 1.5) c = column2_cover(p, body_em * px);
    else c = column3_cover(p, body_em * px);
  }

  float r = rule(q, top, x0, x1, 0.0012, px);
  r = max(r, rule(q, double_rule - 0.0035, x0, x1, 0.0022, px));
  r = max(r, rule(q, double_rule, x0, x1, 0.0008, px));
  r = max(r, rule(q, date_bottom + 0.007, x0, x1, 0.0008, px));
  r = max(r, rule(q, body_top - 0.012, 0.44, 0.56, 0.0008, px));
  r = max(r, rule(q, mast.y - 0.002, x0, mast.x - 0.02, 0.0006, px) * step(q.y, double_rule));
  for (int k = 1; k < 3; k++) {
    float x = x0 + float(k) * (column + gutter) - 0.5 * gutter;
    r = max(r, clamp((0.0007 - abs(q.x - x)) * px + 0.5, 0.0, 1.0) * step(body_top - 0.004, q.y));
  }
  return max(c, r);
}

float fibres(vec2 q, float px, float scale, float seed) {
  vec2 g = q * scale;
  vec2 cell = floor(g) + seed;
  float a = hash(cell) * 6.2832;
  vec2 dir = vec2(cos(a), sin(a));
  vec2 v = fract(g) - 0.5 - (vec2(hash(cell + 1.3), hash(cell + 7.1)) - 0.5) * 0.3;
  float dist = length(v - dir * clamp(dot(v, dir), -0.32, 0.32));
  float cells = px / scale;
  return clamp((0.035 - dist) * cells + 0.5, 0.0, 1.0) * step(0.5, hash(cell + 3.3));
}

vec3 paper_at(vec2 q, float px, float aspect) {
  vec3 col = paper * (0.94 + 0.08 * fbm(q * 7.0));
  col *= 1.0 - 0.06 * fibres(q, px, 70.0, 0.0) + 0.04 * fibres(q, px, 110.0, 40.0);
  col *= 1.0 - 0.1 * fibres(q, px, 260.0, 80.0) * clamp(px / 1500.0, 0.0, 1.0);
  vec2 g = q * 300.0;
  float speck = hash(floor(g) + 9.0);
  float dot_size = 0.12 + 0.2 * fract(speck * 37.0);
  col *= 1.0 - 0.4 * step(0.991, speck) * clamp((dot_size - length(fract(g) - 0.5)) * px / 300.0 + 0.5, 0.0, 1.0);
  vec2 e = min(q, vec2(1.0, aspect) - q);
  float near_edge = 1.0 - smoothstep(0.0, 0.25, min(e.x, e.y));
  float stain = fbm(q * 4.0 + 17.0) + 0.12 * near_edge;
  float fox = smoothstep(0.68, 0.74, stain);
  float tide = smoothstep(0.68, 0.7, stain) - smoothstep(0.7, 0.74, stain);
  col = mix(col, foxing, 0.16 * fox + 0.16 * tide);
  float spots = smoothstep(0.94, 0.98, noise(q * 70.0 + 3.0)) * smoothstep(0.45, 0.65, noise(q * 5.0));
  col = mix(col, foxing, 0.4 * spots);
  float edge = 1.0 - smoothstep(0.0, 0.07, min(e.x, e.y) + 0.025 * noise(q * 30.0));
  return mix(col, foxing * 0.8, 0.45 * edge);
}

// The page point that the lens shows at q, and its magnification, the
// smaller of its two axes. `bow` sets how hard the glass bends at its rim.
vec2 bend(vec2 q, vec2 centre, float radius, float z, float bow, out float m) {
  vec2 d = q - centre;
  float rho = length(d) / radius;
  m = 1.0;
  if (rho >= 1.0) return q;
  float r4 = rho * rho * rho * rho;
  float sag = 1.0 - sqrt(1.0 - r4);
  float s = rho / z + (1.0 - 1.0 / z) * bow * sag;
  float ds = 1.0 / z + (1.0 - 1.0 / z) * bow * 2.0 * rho * rho * rho / sqrt(max(1.0 - r4, 0.0004));
  m = min(z, min(rho / max(s, 0.0001), 1.0 / ds));
  return centre + d * (s / max(rho, 0.0001));
}

void main() {
  vec2 px = floor(uv * iResolution.xy) + 0.5;
  float width = iResolution.x;
  float aspect = iResolution.y / width;
  vec2 q = px / width;

  float t = iTime * speed;
  vec2 centre = vec2(0.5 + 0.34 * sin(1.2 * t), aspect * (0.5 + 0.3 * sin(2.1 * t + 0.6)));
  float radius = 0.27 * aspect;
  vec2 d = q - centre;
  float rho = length(d) / radius;

  float a = -0.02;
  mat2 turn = mat2(cos(a), sin(a), -sin(a), cos(a));
  vec2 middle = vec2(0.5, 0.5 * aspect);

  float m;
  vec2 page = middle + turn * (bend(q, centre, radius, zoom, 1.3, m) - middle);
  vec3 cover = vec3(0.0);
  if (gl_FragCoord.x >= 0.0) {
    for (int i = 0; i < 3; i++) {
      float bow = i == 1 ? 1.34 : (i == 2 ? 1.26 : 1.3);
      float mi;
      vec2 at = middle + turn * (bend(q, centre, radius, zoom, bow, mi) - middle);
      float c = ink_at(at, width * mi);
      if (i == 0) cover = vec3(c);
      else if (i == 1) cover.r = c;
      else cover.b = c;
      if (rho < 0.4 || rho >= 1.0) break;
    }
  }
  float blot = 0.7 + 0.3 * fbm(page * 30.0);
  float pores = smoothstep(0.8, 0.95, noise(page * 520.0)) * clamp(width * m / 1500.0, 0.0, 1.0);
  cover = (1.0 - pow(1.0 - cover, vec3(1.0 + noise(page * 18.0)))) * (blot - 0.3 * pores);
  vec3 col = mix(paper_at(page, width * m, aspect), ink, cover);

  if (rho < 1.0) {
    float lift = 0.05 * (1.0 - rho * rho);
    col = col * (1.0 + lift);
    col = mix(col, rim * 0.6 + col * 0.25, smoothstep(0.86, 0.99, rho));
    float angle = atan(d.y, d.x);
    float arc = (1.0 - smoothstep(0.2, 1.1, abs(angle + 2.3))) * smoothstep(0.66, 0.74, rho) * (1.0 - smoothstep(0.78, 0.86, rho));
    col = mix(col, vec3(1.0), 0.8 * arc);
    float under = (1.0 - smoothstep(0.0, 0.5, abs(angle - 0.9))) * smoothstep(0.82, 0.87, rho) * (1.0 - smoothstep(0.89, 0.93, rho));
    col = mix(col, vec3(1.0), 0.3 * under);
    float spot = 1.0 - smoothstep(0.04, 0.09, length(d / radius - vec2(-0.38, -0.42)));
    col = mix(col, vec3(1.0), 0.85 * spot);
  } else {
    vec2 below = (q - centre - vec2(0.09, 0.12) * radius) / radius;
    float fall = length(below);
    float shade = (1.0 - smoothstep(0.9, 1.12, fall)) * (0.55 + 0.45 * smoothstep(0.55, 0.95, fall));
    col *= 1.0 - 0.35 * shade;
    float caustic = 1.0 - smoothstep(0.0, 0.45, length(below - vec2(0.12, 0.16)));
    col += vec3(0.16, 0.12, 0.06) * caustic * shade;
  }
  col = mix(col, rim * 0.5, clamp((rho - 1.0) * radius * width + 0.5, 0.0, 1.0) * clamp((1.0 + 1.5 / (radius * width) - rho) * radius * width, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}
