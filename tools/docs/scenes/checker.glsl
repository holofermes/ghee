void main() {
  vec2 cell = floor(gl_FragCoord.xy / 8.0);
  gl_FragColor = vec4(vec3(mod(cell.x + cell.y, 2.0)), 1.0);
}
