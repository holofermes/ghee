void main() {
  vec3 col = mix(vec3(0.05, 0.04, 0.09), vec3(0.15, 0.10, 0.20), uv.y);

  vec2 span = size * (stamp_size / stamp_size.y) * (iResolution.y / iResolution.xy);
  vec2 center = vec2(0.5 + bob * 0.015 * sin(iTime * 0.9 * speed),
                     0.5 - bob * 0.04 * sin(iTime * 1.6 * speed));
  vec2 q = (uv - center) / span + 0.5;

  vec4 s = texture2D(stamp_tex, q);
  s.a *= step(abs(q.x - 0.5), 0.5) * step(abs(q.y - 0.5), 0.5);
  gl_FragColor = vec4(mix(col, s.rgb, s.a), 1.0);
}
