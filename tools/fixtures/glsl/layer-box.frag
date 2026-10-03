#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 v_texCoord;
uniform vec2 resolution;
varying vec4 ghee_v0;
varying vec4 ghee_v4;
void main() {
    // The sample moves to the pixel centre, as the slug fill's does.
    vec2 px = v_texCoord * resolution;
    vec2 d = floor(px) + 0.5 - px;
    vec2 u = ghee_v0.xy + vec2(dot(ghee_v4.xy, d), dot(ghee_v4.zw, d));
    // The box's coverage, antialiased over a pixel at each edge.
    vec2 edge = clamp(min(u, 2047.0 - u) * ghee_v0.zw + 0.5, 0.0, 1.0);
    float a = edge.x * edge.y;
    // Yellow where (x - y) / the stripe width has an even floor, x and y in
    // em from the box's bottom left.
    vec2 box = vec2(0.625, 0.75);
    vec2 em = u * box / 2047.0;
    float s = (em.x - em.y) / 0.125 - 0.5;
    float t = abs(fract(s * 0.5) * 2.0 - 1.0) - 0.5;
    // The em spanned by a pixel along each axis of the box.
    vec2 fw = box / (2047.0 * ghee_v0.zw);
    float yellow = clamp(t * 0.125 / (fw.x + fw.y) + 0.5, 0.0, 1.0);
    gl_FragColor = vec4(vec3(1.0, 0.8, 0.0) * (yellow * a), a);
}