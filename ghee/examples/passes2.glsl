// passes2: a neon sign at night. Each pass draws at its own size and is read
// its own way, and the last pass composes them.
//
// - `ticker` holds the scrolling menu, a text layer, at 56x8 pixels at any
//   frame size. `sign` lights one LED of the board for each of its pixels,
//   and `"sampling": "nearest"` hands each LED its pixel whole. Read
//   `"linear"`, each LED blends the pixels around it and the letters smear.
// - `sign` draws the tubes and the LEDs sharp, at the frame's size.
// - `glow` blurs `sign` at a quarter of the frame, 80x60 pixels in a 320x240
//   frame, and `halo` blurs `glow` again at an eighth, 40x30 pixels. Both are
//   read `"linear"`, so their few pixels blend into a smooth glow around the
//   tubes and a wide halo that lights the wall. Read nearest, the glow shows
//   in squares four and eight frame pixels wide.
// - `street` draws one tile of wet cobbles at 256x256 at any frame size, and
//   `main` lays it in perspective under the wall, where a far frame pixel
//   covers many of the tile's pixels. Read `"trilinear"`, ghee also builds
//   the tile's mipmaps after each frame, copies halved again and again down
//   to one pixel, and a far frame pixel reads the copy at its own scale, so
//   the far cobbles stay calm. Read linear, they turn to jagged noise.
//   `main` mirrors every other tile, so the tile's coordinate never jumps and
//   trilinear draws no seam.
// - `main` lights the wall with `glow` and `halo`, adds them over it, and
//   mirrors it all in the wet street.
// - `rain` is hidden with `"visible": false`, so the frame is `main`. Show it
//   with its eye in the console.
//
// The scene param `base`, where the wall meets the street, reaches every
// pass, and `main` and `rain` read it.
//
//     ghee render --demo passes2 -o frame.####.png --frames 60
//     ghee render --demo passes2 --serve

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

vec3 wall(vec2 at, float aspect) {
  vec2 b = vec2(at.x * aspect, at.y) / vec2(0.1, 0.042);
  b.x += 0.5 * mod(floor(b.y), 2.0);
  vec2 f = fract(b);
  float m = smoothstep(0.0, 0.05, f.x) * (1.0 - smoothstep(0.95, 1.0, f.x));
  m *= smoothstep(0.0, 0.12, f.y) * (1.0 - smoothstep(0.88, 1.0, f.y));
  vec3 col = mix(mortar, brick * (0.6 + 0.8 * hash(floor(b))) * (0.75 + 0.5 * f.y * f.y), m);
  vec2 box = abs((at - vec2(0.5, 0.505)) * vec2(aspect, 1.0)) - vec2(0.375, 0.062);
  col = mix(col, vec3(0.01), 1.0 - smoothstep(0.0, 0.004, max(box.x, box.y)));
  float ledge = smoothstep(base - 0.03, base - 0.025, at.y);
  return mix(col, kerb * (0.6 + 0.6 * smoothstep(base - 0.03, base - 0.012, at.y)), ledge);
}

vec2 ripples(vec2 g) {
  vec2 cell = floor(g);
  float h = hash(cell);
  float age = fract(iTime * 0.8 + h * 7.0);
  vec2 d = g - cell - 0.3 - 0.4 * vec2(h, fract(h * 13.0));
  float r = length(d) - age * 0.3;
  float ring = 0.3 * sin(r * 50.0) * exp(-abs(r) * 20.0) * (1.0 - age);
  return d / max(r, 0.001) * ring;
}

void main() {
  float aspect = iResolution.x / iResolution.y;
  vec2 at = uv;
  float below = uv.y - base;
  vec4 st = vec4(0.5, 0.5, 1.0, 0.5);
  float wet = 0.0;
  float shine = 0.0;
  vec3 above = vec3(0.0);
  if (below > 0.0) {
    float z = 0.2 / (below + 0.3);
    vec2 ground = vec2((uv.x - 0.5) * aspect * z, z);
    vec2 g = mod(ground * vec2(6.0, 4.0), 2.0);
    st = texture2D(iChannel3, 1.0 - abs(1.0 - g));
    float puddle = smoothstep(0.0, 0.08, 1.0 - length((ground - vec2(0.02, 0.38)) / vec2(0.15, 0.07)) + 0.4 * noise(ground * 14.0) - 0.2);
    vec2 n = mix((st.rg * 2.0 - 1.0) * sign(1.0 - g), ripples(ground * 6.0), puddle);
    float tilt = (st.a - 0.5) * st.b * (1.0 - puddle);
    at = vec2(uv.x + n.x * 0.02, 2.0 * base - uv.y + n.y * 0.06 + tilt * 0.1);
    wet = mix(mix(0.9, 0.3, st.b), 0.9, puddle);
    shine = mix(0.25 + 0.75 * clamp(-n.y, 0.0, 1.0), 0.0, puddle);
    above = texture2D(iChannel2, vec2(uv.x, base - 0.2)).rgb * halo_gain * exp(-below * 6.0);
  }

  vec4 sign = texture2D(iChannel0, at);
  vec3 light = texture2D(iChannel1, at).rgb * glow_gain;
  if (gl_FragCoord.x >= 0.0) light += texture2D(iChannel2, at).rgb * halo_gain;
  vec3 neon = sign.rgb + sign.a * core + light;

  vec3 col = wall(at, aspect) * (ambient + light * 3.0) + neon;
  if (below > 0.0) {
    vec3 stone = asphalt * (0.5 + st.a) * (ambient + above * 2.0) * st.b * (0.5 + shine);
    col = stone + mix(light, neon, 0.15 + 0.85 * (1.0 - st.b)) * wet;
    col *= smoothstep(0.0, 0.01, below) * 0.6 + 0.4;
  }
  gl_FragColor = vec4(1.0 - exp(-col * exposure), 1.0);
}
