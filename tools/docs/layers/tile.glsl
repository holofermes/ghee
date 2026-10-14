void main() {
  gl_FragColor = texture2D(wifi_tex, fract(uv * iResolution.xy / wifi_size));
}
