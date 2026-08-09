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
uniform float drift;
uniform float speed;


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
  vec2 uv = v_texCoord;

  // buf reads itself = feedback, and reading it off-centre is what makes the
  // wake travel. The buffer is 8-bit: subtract less than one LSB and dim
  // pixels never reach black.
  float steps = DT * 60.0;
  float prev = texture2D(iChannel0, uv + vec2(0.0, drift * steps)).r;
  prev = max(prev * pow(decay, steps) - 1.5 / 255.0 * steps, 0.0);

  float y = 0.5 + wave(uv.x, time * speed);
  float crest = smoothstep(0.012, 0.0, abs(uv.y - y));

  gl_FragColor = vec4(vec3(prev + crest * (1.0 - prev)), 1.0);
}
