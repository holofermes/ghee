// bars: a value per cell of a grid, each cell a field of sliding stripes.
//
// The pass declares the array `amplitudes`, 16 floats, and amplitudes_at(i)
// gives cell i its value. With no program to feed it, ghee feeds a stand-in
// wave. The grid splits the frame in four, two of the quarters in four, and
// two of those cells in four again, so the first values take the big cells.
// A cell's stripes widen with its value and fill it solid past `peak`. Every
// `hold` seconds the grid splits anew: its cells close into their centres one
// by one, and the cells of the next grid open out of theirs. `rate` sets the
// pace of the stripes and of the splits together. At 1 the stripes slide
// `thick_speed` and `fine_speed` stripes a second and a grid holds `hold`
// seconds, and at 0.1 everything runs ten times slower.
//
//     ghee render --demo bars -o frame.####.png --frames 60
//     ghee render --demo bars --filters dither -o dither.png

float rank(float v, float a, float b) {
  return v - step(a, v - 0.5) - step(b, v - 0.5);
}

vec3 pick(float k) {
  vec3 c = mix(ink1, ink2, step(1.0, k));
  c = mix(c, ink3, step(2.0, k));
  return mix(c, ink4, step(3.0, k));
}

void main() {
  vec2 px = uv * iResolution.xy;
  float m = margin * min(iResolution.x, iResolution.y);
  vec2 box = iResolution.xy - 2.0 * m;
  vec2 p = (px - m) / box;
  float frame = step(0.0, p.x) * step(p.x, 1.0) * step(0.0, p.y) * step(p.y, 1.0);
  p = clamp(p, 0.0, 0.9999);

  float t = iTime * rate;
  float grid = t / hold + 0.5;
  float seed = floor(grid);
  float since = fract(grid) * hold;

  float a = floor(ghee_hash(vec2(seed, 1.0)) * 4.0);
  float b = mod(a + 1.0 + floor(ghee_hash(vec2(seed, 2.0)) * 3.0), 4.0);
  float c = floor(ghee_hash(vec2(seed, 3.0)) * 8.0);
  float d = mod(c + 1.0 + floor(ghee_hash(vec2(seed, 4.0)) * 7.0), 8.0);

  vec2 q = floor(p * 2.0);
  float qi = q.x + 2.0 * q.y;
  vec2 local = fract(p * 2.0);
  float split1 = (1.0 - abs(sign(qi - a))) + (1.0 - abs(sign(qi - b)));
  float band = rank(qi, a, b);
  float depth = 1.0;

  vec2 q1 = floor(local * 2.0);
  float n1 = step(min(a, b) + 0.5, qi) * 4.0 + q1.x + 2.0 * q1.y;
  float split2 = (1.0 - abs(sign(n1 - c))) + (1.0 - abs(sign(n1 - d)));
  vec2 local1 = fract(local * 2.0);
  band = mix(band, 2.0 + rank(n1, c, d), split1);
  local = mix(local, local1, split1);
  depth += split1;

  vec2 q2 = floor(local * 2.0);
  float n2 = step(min(c, d) + 0.5, n1) * 4.0 + q2.x + 2.0 * q2.y;
  float deep = split1 * split2;
  band = mix(band, 8.0 + n2, deep);
  local = mix(local, fract(local * 2.0), deep);
  depth += deep;

  vec2 cell = box / exp2(depth);

  float h1 = ghee_hash(vec3(band, seed, 1.0));
  float h2 = ghee_hash(vec3(band, seed, 2.0));
  float h3 = ghee_hash(vec3(band, seed, 3.0));
  float wide = step(0.3 + 0.2 * (depth - 1.0), h2);
  float period = mix(fine, thick, wide) * iResolution.y;
  float angle = floor(h1 * 4.0) * 0.7853982;
  vec2 dir = vec2(cos(angle), sin(angle));
  float speed = mix(fine_speed, thick_speed, wide) * (step(0.5, h3) * 2.0 - 1.0);
  float s = dot(px, dir) / period - t * speed;

  float wait = 0.1 * ghee_hash(vec3(band, seed, 4.0));
  float open = smoothstep(wait, wait + 0.1, since) * smoothstep(wait, wait + 0.1, hold - since);
  vec2 edge = (0.5 * open - abs(local - 0.5)) * cell;
  float inside = clamp(min(edge.x, edge.y) - 0.5 * gap + 0.5, 0.0, 1.0);

  float level = 0.0;
  if (gl_FragCoord.x >= 0.0) level = gain * amplitudes_at(int(band));
  float duty = clamp(level / peak, 0.0, 1.0);
  float stripe = clamp((0.5 * duty - abs(fract(s) - 0.5)) * period + 0.5, 0.0, 1.0);
  float solid = smoothstep(peak - 0.01, peak, level);
  float cover = mix(stripe, 1.0, solid) * inside * frame;

  vec3 ink = pick(floor(ghee_hash(vec3(band, seed, 5.0)) * 4.0));
  gl_FragColor = vec4(mix(paper, ink, cover), 1.0);
}
