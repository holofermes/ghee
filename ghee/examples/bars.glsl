// bars: a bar for each value of an array that a program feeds each frame.
//
// The pass declares the array `amplitudes`, 16 floats. amplitudes_at(i)
// gives a column its own value, and amplitudes_sample(x) reads the same
// array interpolated, for the glow behind the bars. With no program to feed
// it, ghee feeds a stand-in wave. host.py feeds it from Python.
//
//     ghee render --demo bars -o frame.####.png --frames 60
//     ghee render --demo bars --filters dither -o dither.png

vec3 palette(float t) {
    vec3 a = vec3(0.42, 0.40, 0.52);
    vec3 b = vec3(0.42, 0.38, 0.45);
    vec3 c = vec3(1.00, 1.05, 0.95);
    vec3 d = vec3(0.10, 0.42, 0.72);
    return a + b * cos(6.28318 * (c * t + d));
}

vec3 tone(vec3 col, float black, float exposure) {
    col = vec3(1.0) - exp(-col * exposure);
    col = max(col - black, 0.0) / max(1.0 - black, 1e-3);
    return pow(col, vec3(1.0 / 2.2));
}

void main() {
  vec2 p = vec2(uv.x, 1.0 - uv.y);  // y climbs off the floor
  float fi = p.x * float(amplitudes_count);
  int i = int(min(fi, float(amplitudes_count - 1)));
  float h = gain * amplitudes_at(i);
  float curve = gain * amplitudes_sample(p.x);

  float column = smoothstep(0.46, 0.32, abs(fract(fi) - 0.5));
  float body = column * (1.0 - smoothstep(h - 0.004, h + 0.004, p.y));
  float cap = body * smoothstep(h - 0.10, h, p.y);
  float halo = exp(-abs(p.y - curve) * 22.0);

  vec3 col = palette(0.18 + 0.30 * p.x + 0.06 * curve);
  col *= body * (0.18 + 0.7 * p.y) + cap * 1.2 + halo * glow;
  gl_FragColor = vec4(tone(col, 0.02, 1.6), 1.0);
}
