void main() {
  float depth = 1.0 / (1.08 - uv.y);
  vec2 t = 0.6 * vec2((uv.x - 0.5) * depth, depth);
  gl_FragColor = vec4(texture2D(iChannel0, abs(fract(0.5 * t) * 2.0 - 1.0)).rgb, 1.0);
}
