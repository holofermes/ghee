float fbm(vec2 p) {
  return 0.5 * ghee_noise(p) + 0.3 * ghee_noise(p * 2.1 + 5.2) + 0.2 * ghee_noise(p * 4.3 + 1.7);
}

float fibres(vec2 q, float px, float scale, float seed) {
  vec2 g = q * scale;
  vec2 cell = floor(g) + seed;
  float a = ghee_hash(cell) * 6.2832;
  vec2 dir = vec2(cos(a), sin(a));
  vec2 v = fract(g) - 0.5 - (vec2(ghee_hash(vec3(cell, 1.0)), ghee_hash(vec3(cell, 2.0))) - 0.5) * 0.3;
  float dist = length(v - dir * clamp(dot(v, dir), -0.32, 0.32));
  float cells = px / scale;
  return clamp((0.035 - dist) * cells + 0.5, 0.0, 1.0) * step(0.5, ghee_hash(vec3(cell, 3.0)));
}

// The paper at page point q, in widths of the frame, with px pixels to the
// width: x is its share of `paper`, y its share of `foxing`.
vec2 paper_at(vec2 q, float px, float aspect) {
  float tone = 0.94 + 0.08 * fbm(q * 7.0);
  tone *= 1.0 - 0.06 * fibres(q, px, 70.0, 0.0) + 0.04 * fibres(q, px, 110.0, 40.0);
  tone *= 1.0 - 0.1 * fibres(q, px, 260.0, 80.0) * clamp(px / 1500.0, 0.0, 1.0);
  vec2 g = q * 300.0;
  float speck = ghee_hash(floor(g) + 9.0);
  float dot_size = 0.12 + 0.2 * fract(speck * 37.0);
  tone *= 1.0 - 0.4 * step(0.991, speck) * clamp((dot_size - length(fract(g) - 0.5)) * px / 300.0 + 0.5, 0.0, 1.0);
  vec2 e = min(q, vec2(1.0, aspect) - q);
  float near_edge = 1.0 - smoothstep(0.0, 0.25, min(e.x, e.y));
  float stain = fbm(q * 4.0 + 17.0) + 0.12 * near_edge;
  float fox = smoothstep(0.68, 0.74, stain);
  float tide = smoothstep(0.68, 0.7, stain) - smoothstep(0.7, 0.74, stain);
  float spots = smoothstep(0.94, 0.98, ghee_noise(q * 70.0 + 3.0)) * smoothstep(0.45, 0.65, ghee_noise(q * 5.0));
  float edge = 1.0 - smoothstep(0.0, 0.07, min(e.x, e.y) + 0.025 * ghee_noise(q * 30.0));
  float stains = 0.16 * fox + 0.16 * tide;
  stains += 0.4 * spots * (1.0 - stains);
  edge *= 0.45;
  return vec2(tone * (1.0 - stains) * (1.0 - edge), stains * (1.0 - edge) + 0.8 * edge);
}

// The ink that coverage c lays down at page point q: blotchy, bled, and
// pored where magnified.
float inked(float c, vec2 q, float px) {
  float blot = 0.7 + 0.3 * fbm(q * 30.0);
  float pores = smoothstep(0.8, 0.95, ghee_noise(q * 520.0)) * clamp(px / 1500.0, 0.0, 1.0);
  return (1.0 - pow(1.0 - c, 1.0 + ghee_noise(q * 18.0))) * (blot - 0.3 * pores);
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

// The top half holds the page as it lies and the bottom half the page as the
// lens shows it. Each pixel holds its ink, then its paper's shares of `paper`
// (times 0.9) and of `foxing`. The page lies still, so the top half keeps the
// last frame's pixels. It sets anew each pixel that holds no paper yet, as in
// a new buffer, and one band of 8 rows a frame, which brings in new words.
void main() {
  vec2 px = uv * iResolution.xy;
  float width = iResolution.x;
  float height = 0.5 * iResolution.y;
  float aspect = height / width;
  vec4 o = vec4(0.0, 0.0, 0.0, 1.0);
  bool set = false;
  vec2 page;
  float m = 1.0;
  if (px.y < height) {
    if (gl_FragCoord.x >= 0.0) o = texture2D(iChannel0, px / iResolution.xy);
    set = o.g < 0.01 || floor(mod(float(iFrame) + 0.5, ceil(height / 8.0))) == floor(px.y / 8.0);
    if (set) page = turned(px / width, aspect);
  } else {
    vec2 q = vec2(px.x, px.y - height) / width;
    vec3 lens = lens_now(aspect);
    set = length(q - lens.xy) < 1.02 * lens.z;
    if (set) page = seen_at(q, lens, aspect, m);
  }
  if (set) {
    float c = 0.0;
    if (gl_FragCoord.x >= 0.0) c = ink_at(page, width * m);
    o.rgb = vec3(inked(c, page, width * m), paper_at(page, width * m, aspect) * vec2(0.9, 1.0));
  }
  gl_FragColor = o;
}
