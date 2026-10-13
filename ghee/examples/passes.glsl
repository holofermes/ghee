// passes: several passes, each drawn at its own size and read its own way,
// composed by the last one.
//
// - `sun` draws the striped sun at a quarter of the frame, 80x60 pixels in a
//   320x240 frame. Its `"sampling": "nearest"` hands every pass that reads
//   it whole pixels, so the sun shows as blocks four frame pixels wide.
// - `soft` blurs `sun` at an eighth of the frame, 40x30 pixels, and `main`
//   lays it under the sun as its shadow, moved by `offset`. Read nearest, as
//   here, each of its pixels shows as a block eight frame pixels wide. Read
//   `"linear"`, the blocks blend into one smooth shadow.
// - `ground` draws one tile of the grid at 256x256 at any frame size, and
//   `main` repeats it in perspective below the horizon, where a far frame
//   pixel covers many of the tile's pixels. Read linear, as here, each frame
//   pixel blends the four tile pixels around one point, and the far rows
//   flicker as they scroll. Read `"trilinear"`, ghee also builds the tile's
//   mipmaps after each frame, copies halved again and again down to one
//   pixel, and a far frame pixel reads the copy at its own scale, so the far
//   rows fade into a calm haze. `main` mirrors every other tile, so the
//   tile's coordinate never jumps and trilinear draws no seam.
// - `main` composes the three: the sky, the sun over its shadow, and the
//   grid.
// - `grain` is hidden with `"visible": false`, so the frame is `main`. Show
//   it with its eye in the console.
//
// The scene param `horizon` reaches every pass.
//
//     ghee render --demo passes -o frame.####.png --frames 60
//     ghee render --demo passes --serve

void main() {
  float sun = texture2D(iChannel0, uv).r;
  float shade = texture2D(iChannel1, uv - offset).r;

  float below = max(uv.y - horizon, 0.002);
  float z = 0.25 / below;
  vec2 g = vec2((uv.x - 0.5) * z + 0.5, z + iTime * speed);
  float grid = texture2D(iChannel2, 1.0 - abs(1.0 - mod(g, 2.0))).r;

  vec3 col = mix(sky, shadow_color, shade * shadow);
  col = mix(col, sun_color, sun);
  vec3 ground = mix(ground_color, grid_color, grid);
  ground = mix(sky, ground, smoothstep(0.0, 0.25, uv.y - horizon));
  gl_FragColor = vec4(mix(col, ground, smoothstep(-0.002, 0.002, uv.y - horizon)), 1.0);
}
