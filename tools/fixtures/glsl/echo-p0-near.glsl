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
uniform sampler2D iChannel0;
uniform float decay;
uniform float zoom;
uniform float spin;
uniform float hue;


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


void main() {
  vec2 aspect = vec2(resolution.x / resolution.y, 1.0);
  vec2 p = (v_texCoord - 0.5) * aspect;

  // the last frame, read turned and slightly wider than it was drawn, so every
  // echo in the buffer takes one more step out along a spiral. The buffer is
  // 8-bit: subtract less than one LSB and dim pixels never reach black.
  float steps = DT * 60.0;
  float s = sin(spin * steps), c = cos(spin * steps);
  vec2 q = mat2(c, -s, s, c) * p * pow(zoom, steps);
  vec3 prev = texture2D(iChannel0, q / aspect + 0.5).rgb;
  prev = max(prev * pow(decay, steps) - 1.5 / 255.0 * steps, 0.0);

  // two sources on opposite sides, so the buffer winds two arms
  vec2 e = vec2(0.10, 0.0);
  float src = smoothstep(0.018, 0.0, min(distance(p, e), distance(p, -e)));
  vec3 ink = palette(hue + 0.12 * sin(time * 6.0));
  gl_FragColor = vec4(prev + ink * src * (1.0 - prev), 1.0);
}
