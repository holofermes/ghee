float box(vec2 p, vec2 lo, vec2 hi) {
  vec2 d = max(lo - p, p - hi);
  return 1.0 - smoothstep(-0.02, 0.02, max(d.x, d.y));
}

vec3 roof_color(float k) {
  vec3 c = terracotta;
  c = mix(c, slate, step(0.3, k));
  c = mix(c, mustard, step(0.5, k));
  c = mix(c, teal, step(0.68, k));
  return mix(c, cherry, step(0.85, k));
}

vec3 car(vec2 q, float along, float lane, float dir, vec3 col) {
  float s = along * dir - iTime * traffic * (0.8 + 0.4 * ghee_hash(vec2(lane, 1.0)));
  float slot = floor(s / 1.7);
  float x = fract(s / 1.7) * 1.7;
  float here = step(0.45, ghee_hash(vec2(slot, lane)));
  float body = box(vec2(x, q.y), vec2(0.0, -0.1), vec2(0.5, 0.1)) * here;
  float glass = box(vec2(x, q.y), vec2(0.14, -0.07), vec2(0.36, 0.07));
  vec3 paint = vec3(ghee_hash(vec2(slot, lane + 3.0)), ghee_hash(vec2(slot, lane + 5.0)), ghee_hash(vec2(slot, lane + 9.0)));
  paint = mix(paint, vec3(0.95), step(0.7, ghee_hash(vec2(slot, lane + 11.0))));
  vec3 c = mix(col, vec3(0.08), box(vec2(x, q.y), vec2(0.04, -0.07), vec2(0.54, 0.13)) * here * 0.5);
  return mix(c, mix(paint, vec3(0.15, 0.2, 0.25), glass), body);
}

void main() {
  vec2 p = uv * 32.0;
  vec2 lot = floor(p);
  vec2 f = p - lot;
  vec4 plan = texture2D(iChannel0, uv);
  vec2 inblock = mod(lot, 4.0) + f;

  vec3 col = grass * (0.95 + 0.08 * ghee_hash(floor(p * 3.0)));
  float rail = step(15.0, lot.y) * step(lot.y, 15.0);
  vec2 street = vec2(min(inblock.x, 4.0 - inblock.x), min(inblock.y, 4.0 - inblock.y));
  float road = 1.0 - smoothstep(0.24, 0.27, min(street.x, street.y));
  road *= 1.0 - rail;

  if (rail > 0.5) {
    col = mix(gravel, gravel * 0.7, step(0.5, fract(p.x * 6.0)) * 0.3);
    float sleepers = step(0.6, fract(p.x * 5.0)) * box(f, vec2(-1.0, 0.36), vec2(2.0, 0.66));
    col = mix(col, vec3(0.35, 0.27, 0.2), sleepers);
    float rails = box(f, vec2(-1.0, 0.4), vec2(2.0, 0.44)) + box(f, vec2(-1.0, 0.58), vec2(2.0, 0.62));
    col = mix(col, vec3(0.75, 0.75, 0.78), rails);
    float x = p.x - mod(iTime * train_speed + 8.0, 64.0) + 16.0;
    float coach = step(0.0, x) * step(x, 7.2) * step(0.12, fract(x / 1.2) * 1.2);
    float body = coach * box(f, vec2(-1.0, 0.18), vec2(2.0, 0.62));
    float top = coach * box(f, vec2(-1.0, 0.1), vec2(2.0, 0.44));
    float windows = step(0.5, fract(x * 3.0)) * box(f, vec2(-1.0, 0.5), vec2(2.0, 0.57));
    col = mix(col, mix(train, vec3(0.15, 0.2, 0.3), windows), body);
    col = mix(col, train * 1.25 + 0.1, top);
  } else if (road > 0.5) {
    col = asphalt;
    vec2 cross = vec2(step(street.y, 0.27) * step(0.27, street.x), step(street.x, 0.27) * step(0.27, street.y));
    float zebra = (cross.x * step(street.x, 0.55) * step(0.5, fract(inblock.y * 10.0))
                 + cross.y * step(street.y, 0.55) * step(0.5, fract(inblock.x * 10.0))) * step(0.3, max(street.x, street.y));
    col = mix(col, vec3(0.92), zebra);
    float horiz = step(street.y, 0.27) * step(0.55, street.x);
    float vert = step(street.x, 0.27) * step(0.55, street.y);
    float dy = (inblock.y < 2.0 ? inblock.y : inblock.y - 4.0);
    float dx = (inblock.x < 2.0 ? inblock.x : inblock.x - 4.0);
    col = mix(col, vec3(0.9, 0.85, 0.6), horiz * step(0.5, fract(p.x * 2.0)) * (1.0 - smoothstep(0.01, 0.02, abs(dy))));
    col = mix(col, vec3(0.9, 0.85, 0.6), vert * step(0.5, fract(p.y * 2.0)) * (1.0 - smoothstep(0.01, 0.02, abs(dx))));
    float hl = floor((lot.y + 2.0) / 4.0);
    float vl = floor((lot.x + 2.0) / 4.0);
    col = horiz > 0.5 ? car(vec2(0.0, dy - 0.12 * sign(dy)), p.x, hl * 2.0 + step(0.0, dy), sign(dy), col) : col;
    col = vert > 0.5 ? car(vec2(0.0, dx - 0.12 * sign(dx)), p.y, vl * 2.0 + step(0.0, dx) + 50.0, sign(dx), col) : col;
  } else if (plan.r > 0.5 || plan.g > 0.86) {
    vec2 c = vec2(0.3 + 0.4 * plan.b, 0.6 + 0.15 * plan.g);
    float trunk = 0.15 + 0.15 * plan.b;
    vec2 shore = (inblock - 2.0) / vec2(1.1, 0.8);
    float water = (1.0 - smoothstep(0.95, 1.0, length(shore))) * plan.r;
    col *= 1.0 - 0.35 * (1.0 - smoothstep(0.2, 0.26, length((f - c - vec2(0.13, 0.05)) * vec2(1.0, 1.6))));
    vec2 q = f - c + vec2(0.0, trunk);
    float crown = (1.0 - smoothstep(0.21, 0.24, length(q))) * (1.0 - water);
    vec3 leaf = mix(foliage, foliage * 1.45 + 0.04, smoothstep(0.08, -0.16, q.x + q.y));
    col = mix(col, vec3(0.3, 0.22, 0.15), box(f, c - vec2(0.025, trunk), c + vec2(0.025, 0.0)) * (1.0 - water));
    col = mix(col, leaf, crown);
    vec3 ripple = mix(pond, pond * 1.3 + 0.1, step(0.85, fract(dot(inblock, vec2(1.0, 3.0)) * 2.0 + iTime * 0.3)) * 0.4);
    col = mix(col, mix(ripple, sand, smoothstep(0.82, 0.95, length(shore))), water);
  } else {
    float h = 0.1 + 0.22 * plan.g;
    float turn = step(0.6, fract(plan.b * 7.0));
    vec2 lo = vec2(0.12 + 0.12 * fract(plan.g * 5.0), 0.42 + 0.1 * fract(plan.b * 3.0));
    vec2 hi = vec2(0.88 - 0.12 * fract(plan.b * 11.0), 0.86);
    col *= 1.0 - 0.4 * box(f, lo + vec2(0.06 + h * 0.4, 0.02), hi + vec2(0.06 + h * 0.4, 0.04));
    float wall = box(f, vec2(lo.x, hi.y - h), hi);
    float win = step(0.45, fract((f.x - lo.x) * 9.0)) * box(f, vec2(lo.x + 0.05, hi.y - h * 0.7), vec2(hi.x - 0.05, hi.y - h * 0.35));
    col = mix(col, mix(walls, walls * 0.45, win), wall);
    vec2 rlo = lo - vec2(0.03, h + 0.05);
    vec2 rhi = hi - vec2(-0.03, h);
    float roof = box(f, rlo, rhi);
    float lit = mix(step((rlo.y + rhi.y) * 0.5, f.y), step((rlo.x + rhi.x) * 0.5, f.x), turn);
    vec3 tile = roof_color(plan.b) * mix(0.7, 1.1, lit);
    tile *= 0.93 + 0.07 * step(0.5, fract(mix(f.x, f.y, turn) * 14.0));
    col = mix(col, tile, roof);
  }
  float shade = 0.0;
  if (gl_FragCoord.x >= 0.0) shade = texture2D(iChannel1, uv).r;
  gl_FragColor = vec4(col * (1.0 - cloud_shade * shade), 1.0);
}
