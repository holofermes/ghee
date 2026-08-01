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
uniform sampler2D iChannel0;
uniform float hue;
uniform float exposure;


float wave(float x, float t) {
    return 0.28 * sin(x * 4.4 + t) + 0.10 * sin(x * 9.7 - t * 0.6);
}

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
  // buf carries one number per pixel; this pass is where it becomes a picture.
  // Feeding a curve back into a loop compounds it every frame, so the tone
  // curve has to live out here, after the feedback.
  float v = texture2D(iChannel0, v_texCoord).r;
  vec3 col = palette(hue + 0.28 * v) * v;
  col += vec3(1.0, 0.72, 0.42) * smoothstep(0.7, 1.0, v) * 0.5;
  gl_FragColor = vec4(tone(col, 0.04, exposure), 1.0);
}
