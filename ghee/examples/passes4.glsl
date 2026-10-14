// passes4: a town in tilt-shift, the photo trick that blurs all but one row
// of a real street until it looks like a toy model. Each pass draws at its
// own size and is read its own way, and the last pass composes them.
//
// - `lots` draws the town's plan at 32x32 pixels at any frame size, one
//   pixel for each lot: a park or a house, its size and its roof colour.
//   `town` reads it, and `"sampling": "nearest"` hands each lot its pixel
//   whole. Read `"linear"`, each house blends with its neighbours, and the
//   roofs smear into each other's colours and shapes.
// - `clouds` draws the shadows of passing clouds at 64x64, and `town` darkens
//   the board under them. Read `"linear"`, its few pixels blend into soft
//   shadows. Read nearest, they show as squares eight board pixels wide.
// - `town` draws the board seen from above at 512x512, with its cars and
//   its train, and `scene` lays it in perspective, where a far frame pixel
//   covers many of the board's pixels. Read `"trilinear"`, ghee also builds
//   the board's mipmaps after each frame, copies halved again and again
//   down to one pixel, and a far frame pixel reads the copy at its own
//   scale, so the far rows stay calm. Read linear, they turn jagged.
// - `scene` draws that perspective view sharp, at the frame's size.
// - `blur` blurs `scene` at 128x128 and is read trilinear too. `main` reads
//   it with a bias that grows with the distance from the `focus` row, and
//   the bias picks smaller mipmaps, so the blur grows with that distance.
//   Read linear, the bias does nothing and every blurred row is as soft as
//   the next.
// - `main` mixes `scene` and `blur` by that distance and lifts the colours,
//   so the town reads as a model on a table.
// - `photo` is hidden with `"visible": false`, so the frame is `main`. Show
//   it with its eye in the console for a lens's dark corners and grain.
//
//     ghee render --demo passes4 -o frame.####.png --frames 60
//     ghee render --demo passes4 --serve

void main() {
  float away = smoothstep(band, band + fade, abs(uv.y - focus));
  vec2 shrink = iResolution.xy / iChannelResolution[1].xy;
  float bias = log2(min(shrink.x, shrink.y)) + depth_blur * away * away;
  vec3 sharp = texture2D(iChannel0, uv).rgb;
  vec3 soft = texture2D(iChannel1, uv, bias).rgb;
  vec3 col = mix(sharp, soft, away);
  col = mix(vec3(dot(col, vec3(0.299, 0.587, 0.114))), col, saturation);
  gl_FragColor = vec4((col - 0.5) * contrast + 0.5, 1.0);
}
