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
uniform float exposure;
uniform float black;
uniform float vignette;


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
  vec2 uv = v_texCoord;
  vec3 col = texture2D(iChannel0, uv).rgb;

  // nothing downstream of here feeds back, so this is where a curve is safe
  float vig = smoothstep(0.85, 0.25, distance(uv, vec2(0.5)));
  gl_FragColor = vec4(tone(col * mix(1.0, vig, vignette), black, exposure), 1.0);
}
