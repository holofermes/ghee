// The flow is the curl of a stream function: a steady drift plus two
// crossed sine waves that slide in time. A curl has no divergence, so the
// particles swirl without bunching up. The stream function falls to zero at
// the surface, which turns the wave into a wall the current runs along.

float psi(vec2 h) {
  float f = 6.2831853 / swirl_size;
  vec2 q = h * f;
  float t = iTime * 0.1;
  float n = sin(q.x + t) * cos(q.y * 1.1 - t * 0.7)
          + 0.5 * sin(q.x * 2.3 - t * 1.3) * cos(q.y * 1.9 + t * 0.9);
  float free = drift.x * h.y - drift.y * h.x + n * swirl / f;
  return free * smoothstep(0.0, 0.25, depth(vec2(h.x / ASPECT, h.y)));
}

vec2 velocity(vec2 h) {
  float e = 0.004;
  float p0 = psi(h);
  return vec2(psi(h + vec2(0.0, e)) - p0, p0 - psi(h + vec2(e, 0.0))) / e;
}

void main() {
  vec2 p = unpack_pos(texture2D(iChannel0, uv).rgb);
  float age = 1.0;
  if (gl_FragCoord.x >= 0.0) age = unpack_age(texture2D(iChannel1, uv).rg);
  bool born = iTime < 0.1 || age >= lifespan(uv);
  if (!born) {
    float s = pow(2.0, (hash(uv * 9.3) - 0.5) * 2.0 * speed_spread);
    vec2 asp = vec2(ASPECT, 1.0);
    p += velocity(p * asp) / asp * (iTimeDelta * speed * s);
    born = p.x < -0.25 || p.x > 1.25 || p.y > 1.25 || depth(p) < -0.02;
  }
  if (born) {
    float seed = fract(iTime * 0.618) * 17.0;
    p.x = mix(-0.05, 1.05, hash(uv * 3.1 + seed));
    p.y = mix(waterline(p.x) + 0.04, 1.02, hash(uv * 7.7 + seed));
  }
  gl_FragColor = vec4(pack_pos(p), 1.0);
}
