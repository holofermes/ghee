void main() {
  float d = length(gl_FragCoord.xy - 0.5 * iResolution.xy);
  gl_FragColor = vec4(vec3(clamp(20.0 - d, 0.0, 1.0)), 1.0);
}
