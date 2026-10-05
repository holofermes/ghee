#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["ghee-render"]
# ///
"""Now playing on a small panel: the artist on top, the title below it.

The text layer `now` holds both lines and fits them to the pass, so the same
scene fills a 128x32 SSD1306 and a 256x64 SSD1322.

The tick hook stands in for a music player. It steps through PLAYLIST and
sets the words of `now`, and ghee lays them out again with no recompile. Put
a player's own track change in now_playing().

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
    ghee.set_param("now", "\n".join(playing))


def build():
    words = "\n".join(PLAYLIST[0])
    return ghee.Scene(
        passes=[
            ghee.Pass("main", layers=[ghee.Layer("now", words, text_align="center")])
        ],
    )


ghee.run(build())
