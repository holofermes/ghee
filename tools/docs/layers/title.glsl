void main() {
  float px = 48.0;
  vec2 at = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
  vec2 p = (at - 0.5 * (iResolution.xy - title_size * px)) / px;
  gl_FragColor = vec4(vec3(title_cover(p, px)), 1.0);
}
