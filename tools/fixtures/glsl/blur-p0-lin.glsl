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

vec3 tone(vec3 col, float black, float exposure) {
    col = vec3(1.0) - exp(-col * exposure);
    col = max(col - black, 0.0) / max(1.0 - black, 1e-3);
    return pow(col, vec3(1.0 / 2.2));
}


void main() {
  vec2 p = (v_texCoord - 0.5) * vec2(resolution.x / resolution.y, 1.0);
  vec3 col = vec3(0.0);
  for (int k = 0; k < 3; k++) {
    float f = float(k);
    float a = time * (0.5 + 0.19 * f) + 2.1 * f;
    vec2 c = vec2(0.30 * cos(a), 0.23 * sin(a * 1.4));
    col += palette(0.16 + 0.31 * f) * (1.0 - 0.30 * f)
         * exp(-abs(distance(p, c) - 0.11) * 30.0);
  }
  gl_FragColor = vec4(col, 1.0);
}
