float seg(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

float half_ring(vec2 p, vec2 c, float r, vec2 side) {
  vec2 q = p - c;
  vec2 e = vec2(side.y, -side.x) * r;
  return dot(q, side) > 0.0 ? abs(length(q) - r) : min(length(q - e), length(q + e));
}

float letter_r(vec2 p) {
  float d = seg(p, vec2(0.0, 0.0), vec2(0.0, 1.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.27, 0.0)));
  d = min(d, half_ring(p, vec2(0.27, 0.25), 0.25, vec2(1.0, 0.0)));
  d = min(d, seg(p, vec2(0.0, 0.5), vec2(0.27, 0.5)));
  return min(d, seg(p, vec2(0.22, 0.5), vec2(0.55, 1.0)));
}

float letter_a(vec2 p) {
  float d = seg(p, vec2(0.0, 1.0), vec2(0.3, 0.0));
  d = min(d, seg(p, vec2(0.3, 0.0), vec2(0.6, 1.0)));
  return min(d, seg(p, vec2(0.11, 0.64), vec2(0.49, 0.64)));
}

float letter_m(vec2 p) {
  float d = seg(p, vec2(0.0, 1.0), vec2(0.0, 0.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.36, 0.62)));
  d = min(d, seg(p, vec2(0.36, 0.62), vec2(0.72, 0.0)));
  return min(d, seg(p, vec2(0.72, 0.0), vec2(0.72, 1.0)));
}

float letter_e(vec2 p) {
  float d = seg(p, vec2(0.0, 0.0), vec2(0.0, 1.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.46, 0.0)));
  d = min(d, seg(p, vec2(0.0, 0.5), vec2(0.38, 0.5)));
  return min(d, seg(p, vec2(0.0, 1.0), vec2(0.46, 1.0)));
}

float letter_n(vec2 p) {
  float d = seg(p, vec2(0.0, 1.0), vec2(0.0, 0.0));
  d = min(d, seg(p, vec2(0.0, 0.0), vec2(0.58, 1.0)));
  return min(d, seg(p, vec2(0.58, 1.0), vec2(0.58, 0.0)));
}

float cup(vec2 p) {
  float d = half_ring(p, vec2(0.0, 0.42), 0.62, vec2(0.0, 1.0));
  d = min(d, seg(p, vec2(-0.62, 0.42), vec2(0.62, 0.42)));
  return min(d, seg(p, vec2(-0.22, 1.16), vec2(0.22, 1.16)));
}

float sticks(vec2 p) {
  float d = seg(p, vec2(0.28, 0.3), vec2(0.98, -0.48));
  return min(d, seg(p, vec2(0.42, 0.34), vec2(1.1, -0.4)));
}

float wisp(vec2 p, float x) {
  float y = clamp(p.y, -0.7, 0.16);
  return length(vec2(p.x - x - 0.07 * sin(y * 7.0 + 1.0), p.y - y));
}

float hash(float n) {
  return fract(sin(n * 91.345) * 47453.5453);
}

vec4 tube(float d, float px, vec3 color, float on) {
  float body = 1.0 - smoothstep(0.075 - px, 0.075 + px, d);
  float core = exp(-d * d / 0.0015);
  return body * vec4(color * mix(0.1, 1.0, on), core * on);
}

void main() {
  vec2 p = (uv - vec2(0.5, 0.0)) * vec2(iResolution.x / iResolution.y, 1.0);
  vec2 q = (p - vec2(-0.11, 0.2)) / 0.15;
  float px = 1.0 / (0.15 * iResolution.y);
  vec4 col = vec4(0.0);

  if (q.y > -1.2 && q.y < 1.6 && q.x > -3.0 && q.x < 4.5) {
    float buzz = step(0.55, fract(iTime / 6.0)) * step(fract(iTime / 6.0), 0.68);
    float flicker = 1.0 - buzz * step(0.4, hash(floor(iTime * 24.0)));
    float d = min(min(letter_r(q), letter_a(q - vec2(0.85, 0.0))), letter_m(q - vec2(1.75, 0.0)));
    col += tube(min(d, letter_n(q - vec2(3.46, 0.0))), px, word, 1.0);
    col += tube(letter_e(q - vec2(2.75, 0.0)), px, word, flicker);

    vec2 b = q - vec2(-1.95, 0.0);
    col += tube(cup(b), px, bowl, 1.0);
    col += tube(sticks(b), px, chopsticks, 1.0);
    float step_on = floor(fract(iTime / 2.4) * 4.0);
    for (int i = 0; i < 3; i++) {
      float on = step(float(i) + 0.5, step_on);
      col += tube(wisp(b, -0.42 + 0.3 * float(i)), px, steam, on);
    }
  }

  vec2 led = (uv - vec2(0.5, 0.505)) * vec2(iResolution.x / iResolution.y, 1.0) / 0.0125 + vec2(28.0, 4.0);
  if (led.x > 0.0 && led.x < 56.0 && led.y > 0.0 && led.y < 8.0) {
    float dot_mask = 1.0 - smoothstep(0.3, 0.42, length(fract(led) - 0.5));
    float lit = 0.0;
    if (gl_FragCoord.x >= 0.0) lit = texture2D(iChannel0, led / vec2(56.0, 8.0)).a;
    col += dot_mask * vec4(leds * mix(0.1, 1.0, lit), 0.6 * lit);
  }
  gl_FragColor = col;
}
