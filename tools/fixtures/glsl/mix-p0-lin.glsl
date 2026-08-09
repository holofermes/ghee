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


vec3 palette(float t) {
    vec3 a = vec3(0.42, 0.40, 0.52);
    vec3 b = vec3(0.42, 0.38, 0.45);
    vec3 c = vec3(1.00, 1.05, 0.95);
    vec3 d = vec3(0.10, 0.42, 0.72);
    return a + b * cos(6.28318 * (c * t + d));
}

float wave(float x, float t) {
    return 0.28 * sin(x * 4.4 + t) + 0.10 * sin(x * 9.7 - t * 0.6);
}

vec3 tone(vec3 col, float black, float exposure) {
    col = vec3(1.0) - exp(-col * exposure);
    col = max(col - black, 0.0) / max(1.0 - black, 1e-3);
    return pow(col, vec3(1.0 / 2.2));
}


void main() {
  vec2 uv = v_texCoord;
  vec3 col = vec3(0.0);
  for (int k = 0; k < 3; k++) {
    float o = 0.28 * (float(k) - 1.0);
    float d = uv.y - 0.5 - o - 0.45 * wave(uv.x + o, time);
    col += palette(0.16 + 0.10 * float(k) + 0.16 * uv.x) * exp(-d * d * 900.0);
  }
  gl_FragColor = vec4(col, 1.0);
}
