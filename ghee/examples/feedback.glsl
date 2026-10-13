// feedback: a pass that reads its own last frame.
//
// The pass `trail` reads itself through iChannel0, so each frame starts from
// the frame before, scaled toward the centre by `shrink` and dimmed by
// `decay`. `copies` times a second it adds the outline of a rounded triangle
// that turns and breathes, and the older copies sink inward and fade. The
// pass `image` reads `trail` after the loop, gives each level of brightness
// one of four colours, and draws the outline as it is now on top: a colour
// step inside the loop would compound every frame. The common block holds
// the outline, which both passes draw, from the scene params.
//
//     ghee render --demo feedback -o frame.####.png --frames 120
//     ghee render --demo feedback --serve

void main() {
  vec2 aspect = vec2(iResolution.x / iResolution.y, 1.0);
  vec2 p = (uv - 0.5) * aspect;

  // The decay and the shrink count in 60ths of a second, so the trail keeps
  // its shape at any frame rate.
  float steps = iTimeDelta * 60.0;
  float prev = texture2D(iChannel0, p * pow(shrink, steps) / aspect + 0.5).r;
  // The buffer is 8-bit: less than one level off each frame and a dim pixel
  // never reaches black.
  prev = max(prev * pow(decay, steps) - 1.5 / 255.0 * steps, 0.0);

  // 1 in the first frame of each copy's interval, 0 in the others
  float stamp = clamp((iTimeDelta * copies - fract(iTime * copies)) * 1000.0, 0.0, 1.0);
  gl_FragColor = vec4(vec3(max(prev, outline(p) * stamp)), 1.0);
}
