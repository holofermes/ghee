#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""Now playing on a small panel: the artist on top, the title scrolling below.

The artist takes the top ARTIST of the frame and the title the rest, so the
same scene fills a 128x32 SSD1306 and a 256x64 SSD1322. The pass `top` is as
wide as the frame and as tall as the artist's share, both from
ghee.render_size(), and its text layer `artist` fits the words to it, so a
long name shrinks to the frame's width.
The pass `main` copies `top` into its top rows, pixel for pixel. Its text
layer `title` fits the rows below them, and the scroll effect moves it right
to left by whole pixels, so on a 1-bit panel its letters keep their pixels.

The tick hook stands in for a music player. It steps through PLAYLIST and
sets the words of both layers through `ghee.layers`, and ghee lays them out
again with no recompile. Put a player's own track change in now_playing().

    ghee render --demo nowplaying --display kms        # a panel with a kernel driver
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
ARTIST = 0.4  # the artist's share of the frame's height, from the top

# A 1-bit panel lights a pixel above half grey, and a stroke thinner than a
# pixel, such as the bar of a small e, can fall under it. A white text_stroke
# grows the letters by a hundredth of an em to keep such strokes.
INK = {"width": 0.01, "color": "#ffffff"}

# The pass `top`, TOP pixels, fills the top left of the frame pixel for pixel.
MAIN = """
void main() {
    // VC4 interpolates uv up to 0.04 px off the pixel centre.
    vec2 px = floor(uv * iResolution.xy) + 0.5;
    vec4 c = vec4(0.0);
    if (px.x < TOP.x && px.y < TOP.y) c = texture2D(iChannel0, px / TOP);
    gl_FragColor = vec4(c.rgb, 1.0);
}
"""

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
    ghee.layers.artist.text = artist
    ghee.layers.title.text = title


def build():
    width, height = ghee.render_size()
    rows = round(ARTIST * height)
    artist, title = PLAYLIST[0]
    return ghee.Scene(
        passes=[
            ghee.Pass(
                "top",
                size=(width, rows),
                layers=[ghee.Layer("artist", artist, text_stroke=INK)],
            ),
            ghee.Pass(
                "main",
                f"const vec2 TOP = vec2({width:.1f}, {rows:.1f});\n{MAIN}",
                channels=[ghee.Channel("top")],
                layers=[ghee.Layer("title", title, y=rows, scroll={}, text_stroke=INK)],
            ),
        ],
    )


ghee.run(build())
