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
uniform float glow;
uniform float speed;
uniform float hue;


vec3 palette(float t) {
    vec3 a = vec3(0.42, 0.40, 0.52);
    vec3 b = vec3(0.42, 0.38, 0.45);
    vec3 c = vec3(1.00, 1.05, 0.95);
    vec3 d = vec3(0.10, 0.42, 0.72);
    return a + b * cos(6.28318 * (c * t + d));
}


void main() {
  vec2 uv = v_texCoord;
  vec2 p = (uv - 0.5) * vec2(resolution.x / resolution.y, 1.0);

  float t = time * speed;
  vec2 head = 0.34 * vec2(cos(t * 1.7), sin(t * 1.1));
  float d = distance(p, head);
  vec3 ink = palette(hue + 0.16 * sin(time * 0.5))
           * (smoothstep(0.028, 0.0, d) + 0.4 * exp(-d * 26.0));

  // channel 0 is this pass, so this fetch is the previous frame. The buffer
  // is 8-bit: subtract less than one LSB and dim pixels never reach black.
  float steps = DT * 60.0;
  vec3 prev = max(texture2D(iChannel0, uv).rgb * pow(decay, steps) - 1.5 / 255.0 * steps, 0.0);

  // screen rather than add: crossings climb toward the ink, never to white
  gl_FragColor = vec4(prev + ink * glow * (1.0 - prev), 1.0);
}
