#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 v_texCoord;
uniform vec2 resolution;
uniform float time;
uniform float DT;
uniform float frame;
#define gl_FragCoord vec4(v_texCoord * resolution, 0.0, 1.0)
uniform sampler2D amplitudes_tex;
const int amplitudes_size = 16;
uniform float gain;
uniform float glow;
float amplitudes_at(int i0){ return texture2D(amplitudes_tex, vec2((float(i0)+0.5)/float(amplitudes_size), 0.5)).r; }
float amplitudes_sample(float t){ return texture2D(amplitudes_tex, vec2(t, 0.5)).r; }

vec3 palette(float t) {
    vec3 a = vec3(0.42, 0.40, 0.52);
    vec3 b = vec3(0.42, 0.38, 0.45);
    vec3 c = vec3(1.00, 1.05, 0.95);
    vec3 d = vec3(0.10, 0.42, 0.72);
    return a + b * cos(6.28318 * (c * t + d));
}

vec3 tone(vec3 col, float black, float exposure) {
    col = vec3(1.0) - exp(-col * exposure);
    col = max(col - black, 0.0) / max(1.0 - black, 1e-3);
    return pow(col, vec3(1.0 / 2.2));
}

// amplitudes_at(i) gives a column its own value; amplitudes_sample(x) reads the
// same host array interpolated, for the glow that runs behind the bars.
void main() {
  vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y);  // y climbs off the floor
  float fi = uv.x * float(amplitudes_size);
  int i = int(min(fi, float(amplitudes_size - 1)));
  float h = gain * amplitudes_at(i);
  float curve = gain * amplitudes_sample(uv.x);

  float column = smoothstep(0.46, 0.32, abs(fract(fi) - 0.5));
  float body = column * (1.0 - smoothstep(h - 0.004, h + 0.004, uv.y));
  float cap = body * smoothstep(h - 0.10, h, uv.y);
  float halo = exp(-abs(uv.y - curve) * 22.0);

  vec3 col = palette(0.18 + 0.30 * uv.x + 0.06 * curve);
  col *= body * (0.18 + 0.7 * uv.y) + cap * 1.2 + halo * glow;
  gl_FragColor = vec4(tone(col, 0.02, 1.6), 1.0);
}
