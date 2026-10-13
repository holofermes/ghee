// assets: an image asset, read by a shader and drawn by a layer, under text
// layers.
//
// The scene declares stamp.png as the asset `stamp`. This shader draws a
// diagonal checker that slides, and puts a faint berry from `stamp_tex`, the
// asset's sampler, in each `blush` square. The image layer `picture` draws
// the same asset three times its size, and its pixels stay square: an asset
// samples nearest unless it sets "sampling": "linear". The text layers draw
// over both, `title` in a gradient with a stroke and a shadow, `caption`
// with a wave. The console edits each layer on its own.
//
//     ghee render --demo assets -o assets.png
//     ghee render --demo assets --serve

void main() {
  vec2 px = uv * iResolution.xy + vec2(iTime * drift, 0.0);
  // The checker's squares stand on a corner: d counts them along the two
  // diagonals.
  vec2 d = vec2(px.x + px.y, px.x - px.y) / (tile * 1.4142);
  vec2 cell = floor(d);
  float square = mod(cell.x + cell.y, 2.0);
  vec3 col = mix(cream, blush, square);

  // A berry of `berry` pixels stands upright in the middle of each square.
  // stamp.png's edge pixels are transparent, so a clamped read past the
  // berry's box draws nothing.
  vec2 middle = vec2(cell.x + cell.y + 1.0, cell.x - cell.y) * 0.5 * tile * 1.4142;
  vec2 q = (px - middle) / berry + 0.5;
  vec4 s = texture2D(stamp_tex, clamp(q, 0.0, 1.0));
  gl_FragColor = vec4(mix(col, s.rgb, s.a * square * pattern), 1.0);
}
