void main() {
  float px = 48.0;
  vec2 p = (uv * iResolution.xy - 0.5 * (iResolution.xy - title_size * px)) / px;
  gl_FragColor = vec4(vec3(title_cover(p, px)), 1.0);
}
