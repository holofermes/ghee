// A lens drifts over the front page of an old newspaper, magnifies it up to
// `zoom` times and moves at `speed`. The page is set from hidden text layers,
// so this scene alone shows their words: each pixel finds the point of the
// page that the lens shows there, and the _cover of the layer under that point
// reads its words at the size where the glass sets them down, so an edge stays
// one pixel soft at every magnification. Toward its rim the glass bends
// harder, crowds the words into a ring and splits their colours.
//
// The page is three quarters as tall as it is wide. A frame wider than that,
// such as a 256x64 panel, shows a strip of the page across its width and
// scrolls down it `scroll` times a minute, a fraction of a row a frame, back
// to the top in a loop, while the lens glides along the strip.
//
// Pass `print`, in lens-print.glsl, sets the page as it lies and as the lens
// shows it: its paper, and the ink that the words lay on it. This pass colours
// the page and draws the glass.
//
//     ghee render --demo lens -o frame.####.png --frames 100 --timestep 0.05
//     ghee render --demo lens --serve

// The print at p, in its pixels from its top left, y down.
vec4 print_at(vec2 p) {
  vec2 size = iChannelResolution[0].xy;
  return texture2D(iChannel0, vec2(p.x, size.y - p.y) / size);
}

// The ink under the lens at frame point p, in pixels, blended between pixel
// centres.
float under_at(vec2 p) {
  vec2 b = floor(p - 0.5);
  vec2 f = p - 0.5 - b;
  b.y = clamp(b.y, 0.0, iResolution.y - 2.0) + iResolution.y;
  float c00 = print_at(b + 0.5).r;
  float c10 = print_at(b + vec2(1.5, 0.5)).r;
  float c01 = print_at(b + vec2(0.5, 1.5)).r;
  float c11 = print_at(b + 1.5).r;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
  // The page lays out from the frame's top left, y down.
  vec2 px = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
  float width = iResolution.x;
  float aspect = iResolution.y / width;
  vec2 q = px / width;
  vec2 s = scrolled(width, aspect, iTime);
  vec3 lens = lens_now(aspect);
  vec2 centre = lens.xy;
  float radius = lens.z;
  vec2 d = q - centre;
  float rho = length(d) / radius;

  vec3 col;
  if (rho < 1.0) {
    vec4 page = vec4(0.0);
    vec2 fringe = vec2(0.0);
    if (gl_FragCoord.x >= 0.0) {
      page = print_at(px + vec2(0.0, iResolution.y));
      fringe = vec2(page.r);
      if (rho >= 0.4) {
        float r4 = rho * rho * rho * rho;
        float ds = 1.0 / zoom + (1.0 - 1.0 / zoom) * bow * 2.0 * rho * rho * rho / sqrt(max(1.0 - r4, 0.0004));
        // Red and blue take the ink where a bow 0.12 larger or smaller would
        // show it, this many pixels out or in.
        float shift = 0.12 * (1.0 - 1.0 / zoom) * (1.0 - sqrt(1.0 - r4)) / ds * radius * width;
        vec2 outward = d / max(length(d), 0.0001) * shift;
        fringe = vec2(under_at(px + outward), under_at(px - outward));
      }
    }
    col = mix(paper * (page.g / 0.9) + foxing * page.b, ink, vec3(fringe.x, page.r, fringe.y));
    float lift = 0.05 * (1.0 - rho * rho);
    col = col * (1.0 + lift);
    col = mix(col, rim * 0.6 + col * 0.25, smoothstep(0.86, 0.99, rho));
    float angle = atan(d.y, d.x);
    float arc = (1.0 - smoothstep(0.2, 1.1, abs(angle + 2.3))) * smoothstep(0.66, 0.74, rho) * (1.0 - smoothstep(0.78, 0.86, rho));
    col = mix(col, vec3(1.0), 0.8 * arc);
    float under = (1.0 - smoothstep(0.0, 0.5, abs(angle - 0.9))) * smoothstep(0.82, 0.87, rho) * (1.0 - smoothstep(0.89, 0.93, rho));
    col = mix(col, vec3(1.0), 0.3 * under);
    float spot = 1.0 - smoothstep(0.04, 0.09, length(d / radius - vec2(-0.38, -0.42)));
    col = mix(col, vec3(1.0), 0.85 * spot);
  } else {
    vec4 page = vec4(0.0);
    float k = floor(s.x);
    float i = floor(px.y);
    vec2 here = vec2(px.x, mod(i + k, iResolution.y) + 0.5);
    vec2 next = vec2(px.x, mod(i + k + 1.0, iResolution.y) + 0.5);
    if (i > iResolution.y - 1.5) next.y = 2.0 * iResolution.y - 0.5;
    if (gl_FragCoord.x >= 0.0) {
      page = mix(print_at(here), print_at(next), s.x - k);
    }
    col = mix(paper * (page.g / 0.9) + foxing * page.b, ink, page.r);
    vec2 below = (q - centre - vec2(0.09, 0.12) * radius) / radius;
    float fall = length(below);
    float shade = (1.0 - smoothstep(0.9, 1.12, fall)) * (0.55 + 0.45 * smoothstep(0.55, 0.95, fall));
    col *= 1.0 - 0.35 * shade;
    float caustic = 1.0 - smoothstep(0.0, 0.45, length(below - vec2(0.12, 0.16)));
    col += vec3(0.16, 0.12, 0.06) * caustic * shade;
  }
  col = mix(col, rim * 0.5, clamp((rho - 1.0) * radius * width + 0.5, 0.0, 1.0) * clamp((1.0 + 1.5 / (radius * width) - rho) * radius * width, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}
