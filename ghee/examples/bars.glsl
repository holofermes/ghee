// bars: a bar for each value of an array that a program feeds each frame.
//
// The pass declares the array `amplitudes`, 16 floats, and amplitudes_at(i)
// gives column i its value. With no program to feed it, ghee feeds a
// stand-in wave. A bar past `peak` turns `peak_color`.
//
//     ghee render --demo bars -o frame.####.png --frames 60
//     ghee render --demo bars --filters dither -o dither.png

void main() {
  vec2 p = vec2(uv.x, 1.0 - uv.y);  // y climbs off the floor
  float fi = p.x * float(amplitudes_count);
  int i = int(min(fi, float(amplitudes_count - 1)));
  float level = gain * amplitudes_at(i);

  vec2 px = 1.0 / iResolution.xy;
  float base = 0.1;
  float top = base + 0.8 * min(level, 1.0);
  float half_px = 0.5 * float(amplitudes_count) * px.x;
  float column = 1.0 - smoothstep(0.25 - half_px, 0.25 + half_px, abs(fract(fi) - 0.5));
  float body = column * smoothstep(base - px.y, base, p.y) * (1.0 - smoothstep(top - px.y, top, p.y));
  float tip = body * smoothstep(top - cap - px.y, top - cap, p.y);
  float rule = 1.0 - smoothstep(1.5 * px.y, 2.5 * px.y, abs(p.y - base + px.y));
  float over = smoothstep(peak - 0.01, peak, level);

  vec3 col = mix(paper, ink, rule);
  col = mix(col, mix(ink, peak_color, over), body);
  col = mix(col, cap_color, tip);
  gl_FragColor = vec4(col, 1.0);
}
