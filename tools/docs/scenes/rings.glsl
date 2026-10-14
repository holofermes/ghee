void main() {
  float d = length(gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
  gl_FragColor = vec4(vec3(step(0.5, fract(d * rings))), 1.0);
}
