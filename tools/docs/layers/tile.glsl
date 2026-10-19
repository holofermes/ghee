void main() {
  gl_FragColor = texture2D(wifi_tex, fract(gl_FragCoord.xy / wifi_size));
}
