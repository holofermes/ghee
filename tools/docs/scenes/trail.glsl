void main() {
  vec2 spot = 0.5 * iResolution.xy + 24.0 * vec2(cos(iTime * 3.0), sin(iTime * 3.0));
  vec3 last = texture2D(iChannel0, uv).rgb;
  vec3 now = vec3(1.0, 0.55, 0.1) * step(length(gl_FragCoord.xy - spot), 4.0);
  gl_FragColor = vec4(max(last * 0.92 - 1.0 / 255.0, now), 1.0);
}
