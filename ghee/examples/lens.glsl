// A lens drifts over a page of words and magnifies them up to `zoom` times.
// Each pixel finds the point of the page that the lens shows there, and
// page_cover reads the words at that point, at the pixels per em that the
// lens gives it, so an edge stays one pixel soft at every magnification.
void main() {
  // VC4 interpolates v_texCoord up to 0.04 px off the pixel centre.
  vec2 px = floor(v_texCoord * resolution) + 0.5;

  // The page fills 0.9 of the frame, centred, at ppem pixels per em.
  float ppem = 0.9 * min(resolution.x / page_size.x, resolution.y / page_size.y);
  vec2 corner = 0.5 * (resolution - page_size * ppem);

  // The lens wanders over the page, its radius a third of the frame height.
  vec2 centre = resolution * (0.5 + vec2(0.3 * sin(time * 0.4), 0.25 * sin(time * 0.7)));
  float radius = resolution.y / 3.0;
  vec2 d = px - centre;
  float r = min(length(d) / radius, 1.0);

  // The magnification m falls from zoom at the centre to 1 at the rim, and
  // the pixel shows the page at centre + d / m.
  float f = 1.0 - r * r;
  float m = 1.0 + (zoom - 1.0) * f * f;
  vec2 p = (centre + d / m - corner) / ppem;

  // The lens stretches the page by m around its centre, and by `along`
  // along its radius. The coverage ramps over one pixel at the smaller of
  // the two.
  float along = m * m / (1.0 + (zoom - 1.0) * f * (1.0 + 3.0 * r * r));
  float cover = page_cover(p, ppem * min(m, along));

  float inside = step(length(d), radius);
  float rim = clamp(1.5 - abs(length(d) - radius), 0.0, 1.0);
  vec3 col = mix(mix(paper, glass, inside), ink, cover);
  gl_FragColor = vec4(mix(col, ink, 0.6 * rim), 1.0);
}
