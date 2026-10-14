float print_time() {
  return floor(iTime * rate) / rate;
}

// The skyline at a point: x is 1 on a far building, 2 on a near one and 0
// in the sky, yz the point on the window tile.
vec3 skyline(vec2 at) {
  float aspect = iResolution.x / iResolution.y;
  vec3 hit = vec3(0.0);
  for (int i = 0; i < 2; i++) {
    float layer = float(i);
    float width = 0.09 + 0.11 * layer;
    float x = at.x * aspect / width + print_time() * (0.04 + 0.06 * layer) + 17.0 * layer;
    vec2 cell = vec2(floor(x), layer);
    float f = fract(x);
    float wide = 0.65 + 0.3 * ghee_hash(cell);
    float left = (1.0 - wide) * ghee_hash(cell + 3.0);
    float tall = (0.28 - 0.1 * layer) * (0.45 + 0.55 * ghee_hash(cell + 7.0));
    float tier = step(0.55, ghee_hash(cell + 11.0)) * step(left + wide * 0.3, f) * step(f, left + wide * 0.7);
    float top = horizon - tall - tier * (0.07 - 0.03 * layer);
    if (step(left, f) * step(f, left + wide) * step(top, at.y) * step(at.y, horizon) > 0.5) {
      vec2 tile = vec2((f - left) * width, horizon - at.y + 0.01) / (0.1 + 0.22 * layer) + ghee_hash(cell + 5.0) * 4.0;
      hit = vec3(layer + 1.0, tile);
    }
  }
  return hit;
}
