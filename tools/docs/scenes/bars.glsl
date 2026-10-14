void main() {
  float x = uv.x * float(levels_count);
  float level = levels_at(int(x));
  float bar = step(1.0 - uv.y, level) * step(0.2, fract(x));
  gl_FragColor = vec4(vec3(bar), 1.0);
}
