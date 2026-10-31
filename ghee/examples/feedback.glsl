// feedback: particles drift in a current under a slow wave, and a pass that
// reads its own last frame keeps their trails.
//
// The pass `state` holds one particle per texel, 192 in a 16 by 12 grid, and
// moves each along the curl of a stream function (feedback-flow.glsl). `prev`
// keeps last frame's positions, `life` each particle's age, and `attr` the
// step each one took and how bright it is.
//
// This pass, `trails`, reads itself through iChannel2: each frame starts
// from the frame before, halved every `trail_length` seconds, and draws each
// particle's step as a short stroke on top. The trail behind a particle is
// only the buffer's memory, so at a trail_length of zero the dots are left
// alone. `image` colours the trails, from `tail` when dim to `head` when
// bright, with a `glow` around them, under the wave's `surface` line.
//
//     ghee render --demo feedback -o frame.####.png --frames 300
//     ghee render --demo feedback --serve

const int GW = 16;
const int GH = 12;
const vec2 GRID = vec2(16.0, 12.0);
const float DSPAN = 0.0625;
const float JUMP = 0.03;

float seg(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / (dot(ba, ba) + 1e-9), 0.0, 1.0);
  return length(pa - ba * h);
}

// A particle far from this pixel is rejected on its coarse position, before
// the rest of its stroke is worked out.
#define STROKE(cell, texel) \
  away = uva - coarse_pos(texel) * asp; \
  if (dot(away, away) < reach2) { \
    vec2 p1 = unpack_pos(texel); \
    vec3 a = texture2D(iChannel1, cell).rgb; \
    vec2 p0 = p1 - (a.rg - 0.5) * DSPAN; \
    ink = max(ink, a.b * smoothstep(w, w * 0.35, seg(uva, p0 * asp, p1 * asp))); \
  }

void main() {
  vec2 asp = vec2(ASPECT, 1.0);
  vec2 uva = vec2(uv.x, 1.0 - uv.y) * asp;

  // The fade dithers below one level, so an 8-bit buffer fades smoothly and
  // reaches black.
  float steps = iTimeDelta * 60.0;
  float last = 0.0;
  if (gl_FragCoord.x >= 0.0) last = texture2D(iChannel2, uv).r;
  float ink = max(last * pow(0.5, iTimeDelta / max(trail_length, 0.001))
                  - steps * 0.6 / 255.0 + (hash(uv + fract(iTime) * 113.1) - 0.5) / 255.0, 0.0);

  float w = 0.75 * dot_size / iResolution.y;
  float reach = w + (JUMP + 0.006) * length(asp);
  float reach2 = reach * reach;

  vec2 away;
  for (int ix = 0; ix < GW; ix++) {
    for (int iy = 0; iy < GH; iy += 4) {
      vec2 c0 = (vec2(float(ix), float(iy)) + 0.5) / GRID;
      vec2 c1 = (vec2(float(ix), float(iy + 1)) + 0.5) / GRID;
      vec2 c2 = (vec2(float(ix), float(iy + 2)) + 0.5) / GRID;
      vec2 c3 = (vec2(float(ix), float(iy + 3)) + 0.5) / GRID;
      vec3 s0 = vec3(0.0), s1 = vec3(0.0), s2 = vec3(0.0), s3 = vec3(0.0);
      if (gl_FragCoord.x >= 0.0) {
        s0 = texture2D(iChannel0, c0).rgb;
        s1 = texture2D(iChannel0, c1).rgb;
        s2 = texture2D(iChannel0, c2).rgb;
        s3 = texture2D(iChannel0, c3).rgb;
      }
      STROKE(c0, s0);
      STROKE(c1, s1);
      STROKE(c2, s2);
      STROKE(c3, s3);
    }
  }
  gl_FragColor = vec4(ink, 0.0, 0.0, 1.0);
}
