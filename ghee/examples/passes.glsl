// passes: several passes, each drawn at its own size and read its own way,
// composed by the last one.
//
// - `sun` draws at a quarter of the frame, and `"sampling": "nearest"` hands
//   every pass that reads it whole pixels: the sun stays blocky.
// - `halo` draws at an eighth of the frame and blurs `sun`. Its linear
//   sampling smooths it out when `main` reads it at the frame's size.
// - `ground` draws a grid at 256x256, and `"sampling": "trilinear"` builds
//   its mip chain each frame, so the far rows that `main` reads small stay
//   calm.
// - `main` composes the three: the sky, the sun in its halo, and the grid
//   laid out in perspective below the horizon.
// - `scanlines` is hidden with `"visible": false`, so the frame is `main`.
//   Show it with its eye in the console.
//
// The scene param `horizon` reaches every pass.
//
//     ghee render --demo passes -o frame.####.png --frames 60
//     ghee render --demo passes --serve

void main() {
  float sun = texture2D(iChannel0, uv).r;
  float halo = texture2D(iChannel1, uv).r;

  // Each row under the horizon is a distance on the ground, and the grid
  // scrolls toward the viewer.
  float below = max(uv.y - horizon, 0.002);
  float z = 0.25 / below;
  vec2 g = vec2((uv.x - 0.5) * z + 0.5, z + iTime * speed);
  float grid = texture2D(iChannel2, fract(g)).r;

  vec3 sky = mix(sky_top, sky_bottom, clamp(uv.y / horizon, 0.0, 1.0));
  sky = mix(sky + sun_color * halo * glow, sun_color, sun);
  vec3 ground = mix(ground_color, grid_color, grid);
  ground = mix(sky_bottom, ground, smoothstep(0.0, 0.3, uv.y - horizon));
  gl_FragColor = vec4(mix(sky, ground, smoothstep(-0.002, 0.002, uv.y - horizon)), 1.0);
}
