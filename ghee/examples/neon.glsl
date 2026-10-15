// neon: a ramen shop's neon sign at night, in a row of shopfronts on a wet
// street. Each pass draws at its own size and is read its own way, and the
// last pass composes them.
//
// - The common GLSL, neon-row.glsl, lays the row out from the frame's
//   shape: the ramen shop in the middle and as many neighbours as the width
//   holds, the row as tall as the wall in a wide frame, and under upper
//   floors in a narrower one.
// - `ticker` holds the scrolling menu, a text layer, at 56x8 pixels at any
//   frame size. `sign` lights one LED of the board for each of its pixels,
//   and `"sampling": "nearest"` hands each LED its pixel whole. Read
//   `"linear"`, each LED blends the pixels around it and the letters smear.
// - `sign` draws the tubes, the LEDs and the lit windows sharp, at the
//   frame's size.
// - `glow` blurs `sign` at a quarter of the frame, 80x60 pixels in a 320x240
//   frame, and `halo` blurs `glow` again at an eighth, 40x30 pixels. Both are
//   read `"linear"`, so their few pixels blend into a smooth glow around the
//   tubes and a wide halo that lights the walls. Read nearest, the glow shows
//   in squares four and eight frame pixels wide.
// - `street` draws one tile of wet cobbles at 256x256 at any frame size, and
//   `main` lays it in perspective under the wall, where a far frame pixel
//   covers many of the tile's pixels. Read `"trilinear"`, ghee also builds
//   the tile's mipmaps after each frame, copies halved again and again down
//   to one pixel, and a far frame pixel reads the copy at its own scale, so
//   the far cobbles stay calm. Read linear, they turn to jagged noise.
//   `main` mirrors every other tile, so the tile's coordinate never jumps and
//   trilinear draws no seam.
// - `main` lights the shopfronts with `glow` and `halo`, adds them over
//   them, and mirrors it all in the wet street. Over the kerb it draws the
//   passing cars as a long exposure shows them, thin streaks of headlights
//   and taillights, mirrored in the street too. `traffic` sets how many
//   cars pass, none at 0.
// - `rain` draws drops and their splashes on the street over `main`, lit by
//   `halo`. Hide it with its eye in the console for a dry street.
//
// The scene param `base`, where the wall meets the street, reaches every
// pass, and the row, `sign`, `main` and `rain` read it.
//
//     ghee render --demo neon -o frame.####.png --frames 60
//     ghee render --demo neon --size 256x64 --serve

float inside(float d, float px) {
  return 1.0 - smoothstep(-px, px, d);
}

vec3 wall(vec2 at) {
  vec2 s = row_at(at);
  vec3 b = shop_at(s);
  vec2 p = vec2(b.y, s.y);
  float px = 1.0 / (row_height() * iResolution.y);
  float ramen = 1.0 - clamp(abs(b.x), 0.0, 1.0);

  vec2 k = s / vec2(0.17, 0.07);
  k.x += 0.5 * mod(floor(k.y), 2.0);
  vec2 f = fract(k);
  float m = smoothstep(0.0, 0.05, f.x) * (1.0 - smoothstep(0.95, 1.0, f.x));
  m *= smoothstep(0.0, 0.12, f.y) * (1.0 - smoothstep(0.88, 1.0, f.y));
  vec3 bricks = mix(mortar, brick * (0.6 + 0.8 * ghee_hash(floor(k))) * (1.25 - 0.5 * f.y * f.y), m);

  vec2 t = vec2(b.y / 0.15, s.y / 0.11);
  vec2 g = fract(t);
  float seam = smoothstep(0.0, 0.06, g.x) * smoothstep(0.0, 0.08, g.y);
  vec3 panels = facade * (0.6 + 0.5 * ghee_hash(floor(t) + b.x * 31.0)) * (0.55 + 0.45 * seam);
  vec3 col = mix(panels, bricks, ramen);

  float upper = smoothstep(1.0, 1.0 + px, s.y);
  float slab = 1.0 - smoothstep(0.0, 0.04, abs(fract((s.y - 1.08) / 0.55) - 0.05));
  col = mix(col, facade * (0.45 + 0.3 * slab), upper);
  float edge = 1.0 - smoothstep(0.03, 0.03 + px, b.y) * (1.0 - smoothstep(b.z - 0.03 - px, b.z - 0.03, b.y));
  col *= 1.0 - 0.5 * edge;

  float kind = floor(ghee_hash(vec2(b.x, 3.0)) * 4.0);
  float shutter = 1.0 - clamp(abs(kind - 1.0), 0.0, 1.0);
  float slats = 0.7 + 0.3 * smoothstep(0.2, 0.5, fract(s.y / 0.025));
  float front = inside(box(p, vec2(0.08, 0.055), vec2(0.82, 0.6)), px);
  float hole = (1.0 - ramen) * (1.0 - upper) * max(inside(box(p, vec2(0.1, 0.7), vec2(0.8, 0.92)), px), front * (1.0 - shutter));
  hole = max(hole, ramen * max(inside(box(led_at(at), vec2(-0.5), vec2(56.5, 8.5)), 1.0 / led_pitch()), inside(box(s, vec2(-0.5, 0.03), vec2(0.5, 0.36)), px)));
  col = mix(col, facade * 1.6 * slats, (1.0 - ramen) * (1.0 - upper) * front * shutter);
  float canopy = (1.0 - ramen) * inside(box(p, vec2(0.06, 0.615), vec2(0.84, 0.665)), px);
  col = mix(col, facade * 1.8, canopy);
  col = mix(col, vec3(0.008), hole);

  float ledge = smoothstep(base - 0.03, base - 0.025, at.y);
  return mix(col, kerb * (0.6 + 0.6 * smoothstep(base - 0.03, base - 0.012, at.y)), ledge);
}

vec2 ripples(vec2 g) {
  vec2 cell = floor(g);
  float h = ghee_hash(cell);
  float age = fract(iTime * 0.8 + h * 7.0);
  vec2 d = g - cell - 0.3 - 0.4 * vec2(h, fract(h * 13.0));
  float r = length(d) - age * 0.3;
  float ring = 0.3 * sin(r * 50.0) * exp(-abs(r) * 20.0) * (1.0 - age);
  return d / max(r, 0.001) * ring;
}

// One lane of cars' lights as a long exposure draws them: each car a thin
// streak, brightest at its head. The streaks' brightness along x.
float streaks(float x, float dir, float seed) {
  float period = 6.0;
  float u = x * dir - iTime * car_speed + seed * 3.7;
  float k = floor(u / period);
  float h = ghee_hash(vec2(k, seed));
  float len = 1.4 + 1.4 * fract(h * 17.0);
  float behind = period * (0.5 + 0.5 * fract(h * 31.0)) - (u - k * period);
  float px = 1.0 / iResolution.y;
  float along = smoothstep(-px, px, behind) * (1.0 - smoothstep(0.3 * len, len, behind));
  return along * (0.6 + 0.4 * exp(-behind * 6.0)) * step(h, traffic) * 2.5;
}

float across(float dy, float width) {
  return exp(-dy * dy / (width * width)) + 0.08 * exp(-abs(dy) * iResolution.y / 4.0);
}

void main() {
  float aspect = iResolution.x / iResolution.y;
  vec2 at = uv;
  float below = uv.y - base;
  vec4 st = vec4(0.5, 0.5, 1.0, 0.5);
  float wet = 0.0;
  float shine = 0.0;
  vec3 above = vec3(0.0);
  vec3 road_light = vec3(0.0);
  float road = 1.0 - base;
  float far = base - 0.02;
  float near = base + 0.1 * road;
  float width = max(0.6 / iResolution.y, 0.003);
  float x = (uv.x - 0.5) * aspect;
  float back = streaks(x, -1.0, 1.0);
  float front = streaks(x, 1.0, 2.0);
  if (below > 0.0) {
    float z = 0.2 / (below + 0.3);
    vec2 ground = vec2((uv.x - 0.5) * aspect * z, z);
    vec2 g = mod(ground * vec2(6.0, 4.0), 2.0);
    st = texture2D(iChannel3, 1.0 - abs(1.0 - g));
    float puddle = smoothstep(0.0, 0.08, 1.0 - length((ground - vec2(0.02, 0.38)) / vec2(0.15, 0.07)) + 0.4 * ghee_noise(ground * 14.0) - 0.2);
    vec2 n = mix((st.rg * 2.0 - 1.0) * sign(1.0 - g), ripples(ground * 6.0), puddle);
    float tilt = (st.a - 0.5) * st.b * (1.0 - puddle);
    at = vec2(uv.x + n.x * 0.02, 2.0 * base - uv.y + n.y * 0.06 + tilt * 0.1);
    wet = mix(mix(0.9, 0.3, st.b), 0.9, puddle);
    shine = mix(0.25 + 0.75 * clamp(-n.y, 0.0, 1.0), 0.0, puddle);
    above = texture2D(iChannel2, vec2(uv.x, base - 0.2)).rgb * halo_gain * exp(-below * 6.0);
    float dr = 3.0 * width + 0.01 * abs(n.y);
    float smear = 0.05 * road;
    float rf = uv.y - 2.0 * base + far + n.y * 0.01;
    float rn = uv.y - near - 0.12 * road + n.y * 0.01;
    road_light = wet * (taillights * back * (across(rf, dr) + 0.4 * across(rf - smear, smear)) + headlights * front * (across(rn, dr) + 0.4 * across(rn - smear, smear)));
    road_light += texture2D(iChannel1, uv).rgb * glow_gain;
  }

  vec4 sign = texture2D(iChannel0, at);
  vec3 light = texture2D(iChannel1, at).rgb * glow_gain;
  if (gl_FragCoord.x >= 0.0) light += texture2D(iChannel2, at).rgb * halo_gain;
  vec3 neon = sign.rgb + sign.a * core + light;

  vec3 col = wall(at) * (ambient + light * 3.0) + neon;
  if (below > 0.0) {
    vec3 stone = asphalt * (0.5 + st.a) * (ambient + above * 2.0) * st.b * (0.5 + shine);
    col = stone + mix(light, neon, 0.15 + 0.85 * (1.0 - st.b)) * wet;
    col *= smoothstep(0.0, 0.01, below) * 0.6 + 0.4;
    col += road_light;
  }
  col += taillights * back * across(uv.y - far, width) + headlights * front * across(uv.y - near, width);
  gl_FragColor = vec4(1.0 - exp(-col * exposure), 1.0);
}
