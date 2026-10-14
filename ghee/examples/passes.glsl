// passes: a risograph print of a city at dusk. Each ink is a plate, a pass
// at its own size and read its own way, and the last pass prints them.
//
// - `windows` draws one tile of windows at 256x256 at any frame size, and
//   `blue_plate` and `yellow_plate` repeat it on every building, smaller on
//   the far ones, where a frame pixel covers many of the tile's pixels. Read
//   `"trilinear"`, ghee also builds the tile's mipmaps after each frame,
//   copies halved again and again down to one pixel, and a far frame pixel
//   reads the copy at its own scale, so the far windows print as a soft
//   tint. Read `"linear"`, they break into specks. The plates mirror every
//   other tile, so the tile's coordinate never jumps and trilinear draws no
//   seam.
// - `pink_plate` draws the sky at an eighth of the frame, 40x30 pixels in a
//   320x240 frame, and `main` prints each of its pixels as one halftone dot,
//   eight frame pixels wide. `"sampling": "nearest"` hands the whole dot
//   one value, so every dot prints round. Read linear, the value slides
//   across the dot, and the dots along an edge print lopsided.
// - `blue_plate` and `yellow_plate` draw the city, the birds, the moon and
//   the water at the frame's size, and `main` prints the blue in lines.
// - `press` draws how much ink the drum lays down at an eighth of the frame,
//   new for each print, and `main` thins each ink where it is low. Read
//   linear, its few pixels blend into the soft patches of a real print. Read
//   nearest, they print as squares eight frame pixels wide.
// - `main` lays the three inks over the paper, each moved by its own shift,
//   and the pink and the yellow by `jitter` too, a new offset for every
//   print, so the plates sit out of register the way a riso prints them.
// - `grain` is hidden with `"visible": false`, so the frame is `main`. Show
//   it with its eye in the console for the paper's fibres and the specks
//   where the ink missed.
//
// The scene params reach every pass: `horizon` places the city, `rate` sets
// how many prints a second the animation shows, and `paper` is the paper's
// colour.
//
//     ghee render --demo passes -o frame.####.png --frames 60
//     ghee render --demo passes --serve

float halftone(vec2 at, vec2 cells) {
  float v = texture2D(iChannel0, at).r;
  float r = sqrt(v) * 0.72;
  float aa = 0.7 * cells.x / iResolution.x;
  return 1.0 - smoothstep(r - aa, r + aa, length(fract(at * cells) - 0.5));
}

void main() {
  vec2 px = 1.0 / iResolution.xy;
  float t = print_time();
  vec2 shake = (vec2(ghee_hash(vec2(t, 1.0)), ghee_hash(vec2(t, 2.0))) - 0.5) * jitter;
  vec2 shake2 = (vec2(ghee_hash(vec2(t, 3.0)), ghee_hash(vec2(t, 4.0))) - 0.5) * jitter;

  float pink_ink = halftone(uv - (pink_shift + shake) * px, iChannelResolution[0].xy);
  float blue_tone = texture2D(iChannel1, uv - blue_shift * px).r;
  vec3 press = texture2D(iChannel3, uv).rgb;
  float yellow_ink = 0.0;
  if (gl_FragCoord.x >= 0.0) yellow_ink = texture2D(iChannel2, uv - (yellow_shift + shake2) * px).r;

  float line = 0.1 + 0.8 * fract((gl_FragCoord.y - blue_shift.y) / 3.0);
  float blue_ink = smoothstep(line - 0.1, line + 0.1, blue_tone);
  pink_ink *= 1.0 - patchy * press.r;
  blue_ink *= 1.0 - patchy * press.g;
  yellow_ink *= 1.0 - patchy * press.b;

  vec3 col = paper;
  col *= mix(vec3(1.0), pink, pink_ink);
  col *= mix(vec3(1.0), blue, blue_ink);
  col *= mix(vec3(1.0), yellow, yellow_ink);
  gl_FragColor = vec4(col, 1.0);
}
