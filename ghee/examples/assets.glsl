// assets: one image read by a shader and drawn by an image layer, under
// text layers.
//
// The scene declares stamp.png as the asset `stamp`. This shader draws a
// diagonal checker that slides by `drift` pixels a second and wobbles by
// `wobble` pixels, puts a small berry from `stamp_tex`, the asset's sampler,
// in each `blush` square, `berry` pixels wide at `pattern` strength, and
// lights a twinkling sparkle in some `cream` squares. The image layer
// `picture` draws the same asset three times its size, and its pixels stay
// square: an asset samples nearest unless it sets "sampling": "linear". The
// text layers draw over both: `title` in a pixel font from fonts/, with a
// gradient, a stroke and a shadow, and `caption` in ghee's pixel font with
// a wave. The console edits each layer on its own.
//
//     ghee render --demo assets -o assets.png
//     ghee render --demo assets --serve

void main() {
  vec2 px = uv * iResolution.xy + vec2(iTime * drift, 0.0);
  px += wobble * sin(px.yx * 0.035 + iTime * 1.6);
  vec2 d = vec2(px.x + px.y, px.x - px.y) / (tile * 1.4142);
  vec2 cell = floor(d);
  float square = mod(cell.x + cell.y, 2.0);
  vec3 col = mix(cream, blush, square);

  // stamp.png's edge pixels are transparent, so a clamped read past the
  // berry's box draws nothing.
  vec2 middle = vec2(cell.x + cell.y + 1.0, cell.x - cell.y) * 0.5 * tile * 1.4142;
  vec2 q = (px - middle) / berry + 0.5;
  vec4 s = texture2D(stamp_tex, clamp(q, 0.0, 1.0));
  col = mix(col, s.rgb, s.a * square * pattern);

  float h = ghee_hash(cell);
  float r = floor(5.9 * max(sin(iTime * 1.8 + h * 40.0), 0.0));
  vec2 k = abs(floor((px - middle) / 3.0 + 0.5));
  float arm = clamp((1.0 - min(k.x, k.y)) * 8.0, 0.0, 1.0) * clamp((r - max(k.x, k.y)) * 8.0, 0.0, 1.0);
  float core = clamp((1.5 - max(k.x, k.y)) * 8.0, 0.0, 1.0) * clamp((r - 3.5) * 8.0, 0.0, 1.0);
  float star = max(arm, core) * (1.0 - square) * clamp((0.5 - h) * 50.0, 0.0, 1.0);
  gl_FragColor = vec4(mix(col, sparkle, star), 1.0);
}
