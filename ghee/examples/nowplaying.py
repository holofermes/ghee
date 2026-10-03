#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["ghee-render"]
# ///
"""Now playing on a small panel: the artist on top, the title below it.

Both lines take their size from the pass, so the same scene fills a 128x32
SSD1306 and a 256x64 SSD1322. The artist's line takes the top ARTIST of the
pass and shrinks when its words are wider than the pass. The title takes
the rest. A title that fits sits centred. A longer title holds at the left
edge for HOLD seconds, then scrolls left at `speed` em a second. Two layers
draw the title GAP em apart, so the second copy follows the first and the
loop shows no seam. The title moves by whole pixels, so on a 1-bit panel its
letters keep their pixels as they scroll.

The tick hook stands in for a music player. It steps through PLAYLIST and
sets the words of the texts `artist` and `title`, and ghee lays them out
again with no recompile. It also sets `since`, the time the track changed,
so a new title starts its scroll from the left edge. Put a player's own
track change in now_playing().

    ./nowplaying.py --display ssd1306        # the 128x32 I2C OLED
    ./nowplaying.py --display kms            # a panel with a kernel driver
    ./nowplaying.py -o np.####.png --size 256x64 --frames 300 --timestep 0.05
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

PLAYLIST = [
    ("Zoë Quartet", "Slow Rivers"),
    ("The Lanterns", "Night Train to the Coast (Extended Mix)"),
    ("Mira & the Late Hours", "Café at Six"),
]
TRACK_SECONDS = 12.0  # how long each entry of PLAYLIST plays

COMMON = """
// The artist's share of the pass height, from the top.
const float ARTIST = 0.4;
// How long a long title holds at the left edge, in seconds.
const float HOLD = 2.0;
// The space between the two copies of a long title, in em.
const float GAP = 2.0;

// Pixels per em of the artist: its line fills the share, or the pass width
// when the words are wider.
float artist_px() {
    return min(ARTIST * resolution.y / artist_size.y,
               resolution.x / max(artist_size.x, 0.001));
}

// The pass pixel of point em of copy k of the title, 0 or 1.
vec2 title_place(vec2 em, float k) {
    float px = (1.0 - ARTIST) * resolution.y / title_size.y;
    float w = title_size.x * px;
    float y = floor(ARTIST * resolution.y);
    if (w <= resolution.x) {
        // The title fits: copy 0 sits centred, and copy 1 waits right of the pass.
        return vec2(floor(0.5 * (resolution.x - w)) + 2.0 * k * resolution.x, y) + em * px;
    }
    // One cycle holds, then scrolls the title left by its width and the gap,
    // where copy 1 stands at the start of the next cycle.
    float period = w + GAP * px;
    float rate = speed * px;
    float t = mod(time - since, HOLD + period / rate);
    float x = max(t - HOLD, 0.0) * rate;
    return vec2(floor(k * period - x), y) + em * px;
}
"""

ARTIST_PLACE = """
vec2 glyph_place(vec2 em) {
    float px = artist_px();
    vec2 room = vec2(resolution.x, ARTIST * resolution.y);
    return floor(0.5 * (room - artist_size * px)) + em * px;
}
"""

TITLE_PLACE = "vec2 glyph_place(vec2 em) {{ return title_place(em, {k:.1f}); }}"

# A 1-bit panel lights a pixel above half grey, and a stroke thinner than a
# pixel, such as the bar of a small e, can fall under it. This fill grows the
# letters by a tenth of a pixel to keep such strokes.
INK = "void main() { gl_FragColor = vec4(clamp(glyph_dist() + 0.6, 0.0, 1.0)); }"

_now = {"track": -1}


def now_playing(tick):
    """The (artist, title) playing at this tick."""
    return PLAYLIST[int(tick.time // TRACK_SECONDS) % len(PLAYLIST)]


@ghee.tick
def track(tick):
    """Set the words when the track changes."""
    playing = now_playing(tick)
    if playing == _now["track"]:
        return
    _now["track"] = playing
    artist, title = playing
    ghee.set_param("artist", artist)
    ghee.set_param("title", title)
    ghee.set_param("since", tick.time)


def build():
    artist, title = PLAYLIST[0]
    return ghee.Scene(
        common=COMMON,
        params=[
            ghee.Param("speed", "float", 2.0, min=0.5, max=6.0, step=0.1),
            ghee.Param("since", "float", 0.0),
        ],
        texts=[ghee.Text("artist", value=artist), ghee.Text("title", value=title)],
        passes=[
            ghee.Pass(
                "main",
                layers=[
                    ghee.Layer("artist", INK, ARTIST_PLACE),
                    ghee.Layer("title", INK, TITLE_PLACE.format(k=0)),
                    ghee.Layer("title", INK, TITLE_PLACE.format(k=1)),
                ],
            )
        ],
    )


ghee.run(build())
