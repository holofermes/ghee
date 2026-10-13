#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""Now playing on a small panel, as a cassette player's screen: the title,
the track count, the time played and the track's length, and the tape
running between two turning reels.

The pass `main` draws the reels, the bar between them and the play or pause
sign from its params: `progress` fills the bar, `side` winds the tape from
the left reel onto the right one over the whole playlist, and `spin` turns
each reel. Its text layers, in monogram, carry the words. The layout
follows ghee.render_size(): a 256x64 SSD1322 shows the artist on a line of
its own, a 128x32 SSD1306 runs it after the title.

The tick hook stands in for a music player. It plays a stretch of each track
of PLAYLIST, pauses, and moves on, sets the params, and sets the words of the
layers through `ghee.layers`, which ghee lays out again with no recompile. A
title too long for the screen scrolls. Put a player's own state in
now_playing().

    ghee render --demo nowplaying --display kms        # a panel with a kernel driver
    ./nowplaying.py -o np.####.png --size 256x64 --frames 300 --timestep 0.05
"""

import functools
import math

import ghee

ghee.args()  # the probe, this scene takes no arguments

# artist, title, the track's length and where the stretch played starts, in s
PLAYLIST = [
    ("Zoë Quartet", "Slow Rivers", 221, 63),
    ("The Lanterns", "Night Train to the Coast (Extended Mix)", 412, 190),
    ("Mira & the Late Hours", "Café at Six", 178, 131),
]
PLAY, PAUSE = 10.0, 2.0  # seconds that each track plays, then pauses

FONT = "fonts/Monogram.ttf"
PX = 16  # the font's grid, 16 px to the em: capitals 7 px tall, 4 px down
ORANGE = "#ff8b26"
BROWN = "#683311"
BARE = 0.6  # an empty reel's radius, as a share of a full one's
TAPE = 1.6  # the tape's speed, in a full reel's radii a second

W, H = ghee.render_size()
BIG = H >= 64
R = 10 if BIG else 5  # a reel's radius, from its centre pixel
HUB = 4 if BIG else 2
REEL_Y = H - R - 1
BAR = (2 * R + 5, REEL_Y - (3 if BIG else 2), W - 2 * R - 6, REEL_Y + (3 if BIG else 2))
ICON = 7  # the play and pause signs' height, the capitals'
TITLE = 2 * PX if BIG else PX

MAIN = """
const vec2 LEFT = vec2({left:.1f}, {y:.1f});
const vec2 RIGHT = vec2({right:.1f}, {y:.1f});
const float R = {r:.1f};
const float HUB = {hub:.1f};
const vec4 BAR = vec4({bar});
const vec3 ICON = vec3(0.0, {icon_y:.1f}, {icon:.1f});

void reel(vec2 p, vec2 c, float wound, float turn, inout vec3 col) {{
    vec2 q = p - c;
    float d = length(q);
    float pack = sqrt(mix((HUB + 2.0) * (HUB + 2.0), (R - 1.0) * (R - 1.0), wound));
    float tooth = mod(atan(q.y, q.x) - turn, 2.0944) - 1.0472;
    if (abs(d - R) < 0.5) col = dim;
    if (d > HUB + 0.5 && d < pack - 0.5) col = dim;
    if (abs(d - pack) < 0.5) col = ink;
    if (abs(q.y) < 0.5 && abs(q.x) > pack && abs(q.x) < R + 5.0) col = dim;
    if (abs(d - HUB) < 0.5) col = ink;
    if (d < HUB && d > HUB - 2.0 && abs(d * sin(tooth)) < 0.6) col = ink;
}}

void main() {{
    vec2 px = floor(uv * iResolution.xy) + 0.5;
    vec3 col = background;
    reel(px, LEFT, 1.0 - side, spin.x, col);
    reel(px, RIGHT, side, spin.y, col);
    vec2 b = px - 0.5;
    if (b.x >= BAR.x && b.x <= BAR.z && b.y >= BAR.y && b.y <= BAR.w) {{
        bool edge = b.x == BAR.x || b.x == BAR.z || b.y == BAR.y || b.y == BAR.w;
        float end = BAR.x + 2.0 + floor((BAR.z - BAR.x - 3.0) * progress + 0.5);
        bool fill = b.y >= BAR.y + 2.0 && b.y <= BAR.w - 2.0 && b.x >= BAR.x + 2.0 && b.x < end;
        if (edge || fill) col = ink;
    }}
    vec2 i = b - ICON.xy;
    if (i.y >= 0.0 && i.y < ICON.z && i.x >= 0.0) {{
        float h = ICON.z;
        bool play = i.x <= min(i.y, h - 1.0 - i.y);
        bool pause = i.x < 2.0 || (i.x >= (h + 1.0) * 0.5 && i.x < (h + 1.0) * 0.5 + 2.0);
        if (playing > 0.5 ? play : pause) col = ink;
    }}
    gl_FragColor = vec4(col, 1.0);
}}
"""


def now_playing(t):
    """What the player shows at `t` seconds: the track's number, artist,
    title and length, the seconds into it, whether it plays, and the share of
    the whole playlist played."""
    k, at = divmod(t, PLAY + PAUSE)
    n = int(k) % len(PLAYLIST)
    artist, title, length, start = PLAYLIST[n]
    played = min(at, PLAY)
    total = sum(track[2] for track in PLAYLIST)
    before = sum(track[2] for track in PLAYLIST[:n])
    side = (before + start + played) / total
    return n, artist, title, length, start + played, at < PLAY, side


def clock(s):
    return f"{int(s) // 60}:{int(s) % 60:02}"


@functools.cache
def width(words, size=PX):
    return ghee.measure(words, font_size=size, font_family=FONT)[0]


_now = {"track": None, "time": None, "playing": None, "spin": [0.0, 0.0]}


@ghee.tick
def player(tick):
    """Turn the reels, fill the bar, and set the words when they change."""
    n, artist, title, length, pos, playing, side = now_playing(tick.time)
    if playing:
        for k, wound in enumerate((1 - side, side)):
            radius = math.sqrt(BARE**2 + wound * (1 - BARE**2))
            _now["spin"][k] += tick.dt * TAPE / radius
    ghee.set_param("spin", _now["spin"])
    ghee.set_param("progress", pos / length)
    ghee.set_param("side", side)
    ghee.set_param("playing", 1.0 if playing else 0.0)
    layers = ghee.layers
    if _now["track"] != n:
        _now["track"] = n
        layers.count.text = f"{n + 1:02}/{len(PLAYLIST):02}"
        words = title if BIG else f"{title} - {artist}"
        fits = width(words, TITLE) <= W
        layers.title.text = words
        layers.title.scroll.visible = not fits
        layers.title.x = round((W - width(words, TITLE)) / 2) if fits else 0
        if BIG:
            layers.artist.text = artist.upper()
            layers.artist.x = round((W - width(artist.upper())) / 2)
    shown = (
        f"{clock(pos)} / {clock(length)}" if BIG else f"{clock(pos)}/{clock(length)}"
    )
    if _now["time"] != shown:
        _now["time"] = shown
        layers.time.text = shown
        layers.time.x = W - round(width(shown))
    if _now["playing"] != playing:
        _now["playing"] = playing
        layers.time.blink.visible = not playing


def build():
    text = {"font_family": FONT, "color": ORANGE, "render": "mono"}
    layers = [
        ghee.Layer("count", "01/03", x=ICON + 3, y=-3, font_size=PX, **text),
        ghee.Layer(
            "time",
            "0:00",
            y=-3,
            font_size=PX,
            blink={"rate": 2, "visible": False},
            **text,
        ),
        ghee.Layer(
            "title",
            PLAYLIST[0][1],
            x=0,
            y=14 if BIG else 7,
            font_size=TITLE,
            scroll={"speed": 2.0},
            **text,
        ),
    ]
    if BIG:
        layers.append(ghee.Layer("artist", PLAYLIST[0][0], y=7, font_size=PX, **text))
    shader = MAIN.format(
        left=R + 0.5,
        right=W - R - 0.5,
        y=REEL_Y + 0.5,
        r=R,
        hub=HUB,
        bar=", ".join(f"{v:.1f}" for v in BAR),
        icon_y=1,
        icon=ICON,
    )
    return ghee.Scene(
        passes=[
            ghee.Pass(
                "main",
                shader,
                params=[
                    ghee.Param("background", "color", "#000000"),
                    ghee.Param("ink", "color", ORANGE),
                    ghee.Param("dim", "color", BROWN),
                    ghee.Param("progress", "float", 0.0, min=0.0, max=1.0),
                    ghee.Param("side", "float", 0.0, min=0.0, max=1.0),
                    ghee.Param("spin", "vec2", [0.0, 0.0]),
                    ghee.Param("playing", "float", 1.0, min=0.0, max=1.0),
                ],
                layers=layers,
            ),
        ],
    )


ghee.run(build())
