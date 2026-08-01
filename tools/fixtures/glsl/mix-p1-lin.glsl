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
  vec2 p = (v_texCoord - 0.5) * vec2(resolution.x / resolution.y, 1.0);
  float d = length(p);
  float r = 0.5 + 0.5 * sin(d * 34.0 - time * 2.2);
  gl_FragColor = vec4(palette(0.30 + 0.30 * d) * pow(r, 4.0) * exp(-d * 2.4), 1.0);
}
