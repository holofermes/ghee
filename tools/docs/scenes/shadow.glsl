void main() {
  float disc = texture2D(iChannel0, uv).r;
  float shadow = texture2D(iChannel0, uv - vec2(6.0) / iResolution.xy).r;
  vec3 color = mix(vec3(0.85), vec3(0.35), shadow);
  gl_FragColor = vec4(mix(color, vec3(1.0, 0.55, 0.1), disc), 1.0);
}
