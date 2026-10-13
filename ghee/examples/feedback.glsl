// feedback: a pass that reads its own last frame.
//
// The pass `trail` reads itself through iChannel0, so each frame starts from
// the frame before, turned by `spin`, scaled by `zoom` about the centre and
// dimmed by `decay`, and draws a wave on top of it. The old waves spiral out
// behind the new one. The pass `image` reads `trail` after the loop and gives
// it colour: a colour curve inside the loop would compound every frame.
//
//     ghee render --demo feedback -o frame.####.png --frames 120
//     ghee render --demo feedback --serve

float wave(float x, float t) {
  return 0.22 * sin(x * 4.4 + t) + 0.08 * sin(x * 9.7 - t * 0.6);
}

void main() {
  vec2 aspect = vec2(iResolution.x / iResolution.y, 1.0);
  vec2 p = (uv - 0.5) * aspect;

  // The decay, the turn and the zoom count in 60ths of a second, so the wake
  // keeps its shape at any frame rate.
  float steps = iTimeDelta * 60.0;
  float s = sin(spin * steps), c = cos(spin * steps);
  vec2 q = mat2(c, -s, s, c) * p * pow(zoom, steps);
  float prev = texture2D(iChannel0, q / aspect + 0.5).r;
  // The buffer is 8-bit: less than one level off each frame and a dim pixel
  // never reaches black.
  prev = max(prev * pow(decay, steps) - 1.5 / 255.0 * steps, 0.0);

  float y = 0.5 + wave(uv.x, iTime * speed);
  float crest = smoothstep(0.012, 0.0, abs(uv.y - y));
  gl_FragColor = vec4(vec3(max(prev, crest)), 1.0);
}
