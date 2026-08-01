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
uniform float amount;
uniform float levels;
uniform float color;
uniform float scale;


float bayer(vec2 p){
    float x = mod(p.x, 4.0); float y = mod(p.y, 4.0); float v;
    if (y < 1.0){ if(x<1.0)v=0.0; else if(x<2.0)v=8.0; else if(x<3.0)v=2.0; else v=10.0; }
    else if (y < 2.0){ if(x<1.0)v=12.0; else if(x<2.0)v=4.0; else if(x<3.0)v=14.0; else v=6.0; }
    else if (y < 3.0){ if(x<1.0)v=3.0; else if(x<2.0)v=11.0; else if(x<3.0)v=1.0; else v=9.0; }
    else { if(x<1.0)v=15.0; else if(x<2.0)v=7.0; else if(x<3.0)v=13.0; else v=5.0; }
    return (v + 0.5) / 16.0;
}
void main(){
    vec2 uv = gl_FragCoord.xy / resolution;
    vec3 src = texture2D(iChannel0, uv).rgb;
    vec3 base = mix(vec3(dot(src, vec3(0.299, 0.587, 0.114))), src, color);
    float steps = max(levels - 1.0, 1.0);
    float d = bayer(floor(gl_FragCoord.xy / scale));
    vec3 q = clamp(floor(base * steps + d) / steps, 0.0, 1.0);
    gl_FragColor = vec4(mix(src, q, amount), 1.0);
}
