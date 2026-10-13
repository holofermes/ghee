// feedback: a pass that reads its own last frame.
//
// The pass `trail` reads itself through iChannel0, so each frame starts from
// the frame before, scaled toward the centre by `shrink`, turned by `twist`
// degrees a second and dimmed by `decay`, with black past its edges.
// `copies` times a second it adds the outline of a polygon with `corners`
// corners that turns and breathes. The older copies sink inward and fade,
// and as each one landed at a turn of its own, their corners wind into a
// spiral. Each pixel of `trail` holds the age of its copy in red, 1 as the
// copy lands and down to 0, and in green how much line covers it. The pass
// `image` reads `trail` after the loop, greys each copy by its age, and
// draws the outline as it is now on top, with a glow. The common block holds
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
  float a = radians(twist) * iTimeDelta;
  vec2 q = mat2(cos(a), sin(a), -sin(a), cos(a)) * p * pow(shrink, steps) / aspect + 0.5;
  vec2 inside = clamp((0.5 - abs(q - 0.5)) * iResolution.xy + 0.5, 0.0, 1.0);
  vec2 prev = texture2D(iChannel0, q).rg * inside.x * inside.y;
  // The buffer is 8-bit: less than one level off each frame and a dim pixel
  // never reaches black.
  float fade = max(1.5 * steps, 1.0) / 255.0;
  float age = max(prev.r * pow(decay, steps) - fade, 0.0);
  float ink = max(prev.g - fade, 0.0);

  float stamp = clamp((iTimeDelta * copies - fract(iTime * copies)) * 1000.0, 0.0, 1.0);
  float line = outline(edge(p)) * stamp;
  gl_FragColor = vec4(mix(age, 1.0, smoothstep(0.0, 0.6, line)), max(ink, line), 0.0, 1.0);
}
