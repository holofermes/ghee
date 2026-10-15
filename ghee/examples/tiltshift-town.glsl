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

float coach(float x, float y) {
  return step(0.0, x) * step(x, 7.2) * step(0.12, fract(x / 1.2) * 1.2) * box(vec2(0.0, y), vec2(-1.0, 0.36), vec2(1.0, 0.64));
}

void main() {
  vec2 p = uv * 32.0;
  vec2 lot = floor(p);
  vec2 f = p - lot;
  vec4 plan = texture2D(iChannel0, uv);
  vec2 block = mod(lot, 4.0);
  vec2 inblock = block + f;

  float a = radians(yaw);
  vec2 away = vec2(sin(a), cos(a));
  vec2 side = vec2(cos(a), -sin(a));
  float lift = cos(radians(tilt)) / max(sin(radians(tilt)), 0.2);
  vec2 sun = -0.8 * side - 0.3 * away;
  vec2 light = -normalize(sun);

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
    vec2 up = away * clamp(0.24 * lift, 0.0, 0.3);
    col = mix(col, col * 0.6, coach(x - sun.x * 0.15, f.y - sun.y * 0.15));
    float side_wall = max(coach(x, f.y), coach(x - up.x * 0.5, f.y - up.y * 0.5));
    float band = coach(x - up.x * 0.6, f.y - up.y * 0.6) * (1.0 - coach(x - up.x * 0.8, f.y - up.y * 0.8));
    float windows = band * step(0.5, fract((x - up.x * 0.7) * 3.0));
    col = mix(col, mix(train * 0.8, vec3(0.15, 0.2, 0.3), windows), side_wall);
    col = mix(col, train * 1.25 + 0.1, coach(x - up.x, f.y - up.y));
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
  } else {
    vec2 lo_lot = mix(vec2(0.05), vec2(0.32), step(block, vec2(0.5)));
    vec2 hi_lot = mix(vec2(0.95), vec2(0.68), step(2.5, block));
    vec2 mid = (lo_lot + hi_lot) * 0.5;
    vec2 room = (hi_lot - lo_lot) * 0.5;
    if (plan.r > 0.5 || plan.g > 0.86) {
      vec2 base = mid + (vec2(plan.b, plan.g) - 0.5) * max(room - 0.22, 0.0);
      float trunk = 0.15 + 0.15 * plan.b;
      vec2 c = clamp(base + away * trunk * lift, lo_lot + 0.2, hi_lot - 0.2);
      vec2 shore = (inblock - 2.0) / vec2(1.1, 0.8);
      float water = (1.0 - smoothstep(0.95, 1.0, length(shore))) * smoothstep(0.85, 0.95, plan.r);
      col *= 1.0 - 0.35 * (1.0 - smoothstep(0.2, 0.26, length(f - base - sun * 0.2)));
      vec2 q = f - c;
      float crown = (1.0 - smoothstep(0.19, 0.22, length(q))) * (1.0 - water);
      vec3 leaf = mix(foliage, foliage * 1.45 + 0.04, smoothstep(-0.04, 0.12, dot(q, light)));
      vec2 stem = c - base;
      float t = clamp(dot(f - base, stem) / max(dot(stem, stem), 1e-4), 0.0, 1.0);
      col = mix(col, vec3(0.3, 0.22, 0.15), (1.0 - smoothstep(0.02, 0.035, length(f - base - stem * t))) * (1.0 - water));
      col = mix(col, leaf, crown);
      float ring = fract(length(shore) * 2.5 - iTime * 0.25);
      vec3 ripple = mix(pond, pond * 1.25 + 0.08, smoothstep(0.3, 0.5, ring) * smoothstep(0.7, 0.5, ring) * 0.5);
      col = mix(col, mix(ripple, sand, smoothstep(0.82, 0.95, length(shore))), water);
    } else {
      float h = 0.16 + 0.3 * plan.g;
      float turn = step(0.6, fract(plan.b * 7.0));
      vec2 size = vec2(0.22 + 0.1 * fract(plan.g * 5.0), 0.15 + 0.08 * fract(plan.b * 3.0));
      size = min(mix(size, size.yx, turn), room - 0.04);
      vec2 rise = away * h * lift;
      vec2 fit = 2.0 * (room - size - 0.03) / (abs(rise) + 1e-4);
      rise *= min(1.0, min(fit.x, fit.y));
      vec2 lo = mid - rise * 0.5 - size;
      vec2 hi = mid - rise * 0.5 + size;
      vec2 fall = sun * (0.08 + 0.5 * h);
      fall = clamp(fall, lo_lot - 0.05 - lo, hi_lot + 0.05 - hi);

      vec2 s = fall + vec2(1e-4);
      vec2 k0 = (f - hi) / s;
      vec2 k1 = (f - lo) / s;
      vec2 kin = min(k0, k1);
      vec2 kout = max(k0, k1);
      float shadow = step(max(max(kin.x, kin.y), 0.0), min(min(kout.x, kout.y), 1.0));
      col *= 1.0 - 0.4 * shadow;

      vec2 r = rise + vec2(1e-4);
      vec2 face = mix(hi, lo, step(0.0, r));
      vec2 kw = (f - face) / r;
      vec2 at = f.yx - kw * r.yx;
      float wall_x = step(0.0, kw.x) * step(kw.x, 1.0) * step(lo.y, at.x) * step(at.x, hi.y);
      float wall_y = step(0.0, kw.y) * step(kw.y, 1.0) * step(lo.x, at.y) * step(at.y, hi.x);
      float k = mix(kw.y, kw.x, wall_x);
      float u = mix(at.y, at.x, wall_x);
      vec2 n = mix(vec2(0.0, -sign(r.y)), vec2(-sign(r.x), 0.0), wall_x);
      float win = step(0.45, fract((u - mix(lo.x, lo.y, wall_x)) * 9.0)) * step(0.3, k) * step(k, 0.7);
      vec3 wall = walls * (0.85 + 0.25 * dot(n, light));
      col = mix(col, mix(wall, wall * 0.45, win), max(wall_x, wall_y));

      vec2 g = f - rise;
      float roof = box(g, lo - 0.03, hi + 0.03);
      vec2 ridge = away * 0.08 * lift * min(1.0, min(fit.x, fit.y));
      float across = mix(g.y - (lo.y + hi.y) * 0.5 - ridge.y, g.x - (lo.x + hi.x) * 0.5 - ridge.x, turn);
      vec2 slope = mix(vec2(0.0, sign(across)), vec2(sign(across), 0.0), turn);
      vec3 tile = roof_color(plan.b) * (0.9 + 0.25 * dot(slope, light));
      tile *= 0.93 + 0.07 * step(0.5, fract(mix(g.x, g.y, turn) * 14.0));
      col = mix(col, tile, roof);
    }
  }
  float shade = 0.0;
  if (gl_FragCoord.x >= 0.0) shade = texture2D(iChannel1, uv).r;
  gl_FragColor = vec4(col * (1.0 - cloud_shade * shade), 1.0);
}
