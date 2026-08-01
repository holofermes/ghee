"""GLSL the bundled demos share: a palette, a wave, and the tone curve.

A scene picks these up through its `common` block, which ghee prepends to every
pass after the preamble.

    Scene(common=look.PALETTE + look.TONE, passes=[...])
"""

# A cosine palette: deep blue through cyan and violet, warm at the far end.
# t wraps, so a phase can just keep climbing.
PALETTE = """
vec3 palette(float t) {
    vec3 a = vec3(0.42, 0.40, 0.52);
    vec3 b = vec3(0.42, 0.38, 0.45);
    vec3 c = vec3(1.00, 1.05, 0.95);
    vec3 d = vec3(0.10, 0.42, 0.72);
    return a + b * cos(6.28318 * (c * t + d));
}
"""

# Two sines, the second nearly but deliberately not a harmonic of the first and
# running the other way, so the crest never repeats on the beat.
WAVE = """
float wave(float x, float t) {
    return 0.28 * sin(x * 4.4 + t) + 0.10 * sin(x * 9.7 - t * 0.6);
}
"""

# Highlights roll off, the toe crushes to true black, then an sRGB encode.
# Without that last step the dark end reads as mud.
TONE = """
vec3 tone(vec3 col, float black, float exposure) {
    col = vec3(1.0) - exp(-col * exposure);
    col = max(col - black, 0.0) / max(1.0 - black, 1e-3);
    return pow(col, vec3(1.0 / 2.2));
}
"""
