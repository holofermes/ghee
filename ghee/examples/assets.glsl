// assets: an image asset, read by a shader and drawn by a layer, under text
// layers.
//
// The scene declares stamp.png as the asset `stamp`. This shader tiles it
// over the background through `stamp_tex`, the asset's sampler, and
// `stamp_size`, its size in pixels. The image layer `picture` draws the same
// asset three times its size, and its pixels stay square: an asset samples
// nearest unless it sets "sampling": "linear". The text layers draw over
// both, `title` in a gradient with a stroke and a shadow, `caption` with a
// glow. The console edits each layer on its own.
//
//     ghee render --demo assets -o assets.png
//     ghee render --demo assets --serve

void main() {
  vec3 col = mix(top, bottom, uv.y);
  vec2 px = uv * iResolution.xy + iTime * vec2(drift, 0.5 * drift);
  vec4 s = texture2D(stamp_tex, fract(px / (stamp_size * tile)));
  gl_FragColor = vec4(mix(col, s.rgb, s.a * pattern), 1.0);
}
