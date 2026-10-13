// passes: several passes, each drawn at its own size and read its own way,
// composed by the last one.
//
// - `sun` draws at a quarter of the frame, and `"sampling": "nearest"` hands
//   every pass that reads it whole pixels: the sun keeps its blocky edge.
// - `soft` draws at an eighth of the frame and blurs `sun`. Its linear
//   sampling smooths it out when `main` reads it at the frame's size, as the
//   sun's shadow.
// - `ground` draws a grid at 256x256, and `"sampling": "trilinear"` builds
//   its mip chain each frame, so the far rows that `main` reads small stay
//   calm.
// - `main` composes the three: the sky, the sun over its shadow, and the
//   grid laid out in perspective below the horizon.
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
  float grid = texture2D(iChannel2, fract(g)).r;

  vec3 col = mix(sky, shadow_color, shade * shadow);
  col = mix(col, sun_color, sun);
  vec3 ground = mix(ground_color, grid_color, grid);
  ground = mix(sky, ground, smoothstep(0.0, 0.25, uv.y - horizon));
  gl_FragColor = vec4(mix(col, ground, smoothstep(-0.002, 0.002, uv.y - horizon)), 1.0);
}
