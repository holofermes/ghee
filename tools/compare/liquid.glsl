// pygame's liquid.py: each block of the screen copies a block of the bitmap
// from a spot shifted by a sine of anim and of the block's position, anim
// growing by 0.02 a frame. pygame takes the shifted spot as whole pixels,
// cut toward zero. VC4 divides through an approximate reciprocal, so a
// pixel's block is found from its centre.

void main() {
  vec2 p = floor(uv * iResolution.xy);
  vec2 b = floor((p + 0.5) / blocksize) * blocksize;
  float anim = 0.02 * (float(iFrame) + first + 1.0);
  vec2 wave = vec2(sine(anim + b.x * freq), sine(anim + b.y * freq));
  vec2 src = floor(b + wave * amplitude + shift) + p - b;
  gl_FragColor = texture2D(liquid_tex, (src + 0.5) / liquid_size);
}
