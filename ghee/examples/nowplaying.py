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
its own, a 128x32 SSD1306 runs it after the title. The scene param
`padding` insets the whole layout from the frame's edges by that many
pixels, and the pass `inset` paints that margin in `background`, over a
scrolling title too.

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
PADDING = 2
ORANGE = "#ff8b26"
BROWN = "#683311"
BARE = 0.6  # an empty reel's radius, as a share of a full one's
TAPE = 1.6  # the tape's speed, in a full reel's radii a second

W, H = ghee.render_size()
BIG = H >= 64
R = 10 if BIG else 5  # a reel's radius, from its centre pixel
HUB = 4 if BIG else 2
BAR = 3 if BIG else 2  # the bar's rows above and below the reels' centre
ICON = 7  # the play and pause signs' height, the capitals'
TITLE = 2 * PX if BIG else PX

MAIN = """
const float R = {r:.1f};
const float HUB = {hub:.1f};
const float BAR = {bar:.1f};
const float ICON = {icon:.1f};

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
    float pad = float(padding);
    vec2 px = uv * iResolution.xy;
    float y = iResolution.y - pad - R - 1.0;
    vec3 col = background;
    reel(px, vec2(pad + R + 0.5, y + 0.5), 1.0 - side, spin.x, col);
    reel(px, vec2(iResolution.x - pad - R - 0.5, y + 0.5), side, spin.y, col);
    vec2 b = px - 0.5;
    vec4 bar = vec4(pad + 2.0 * R + 5.0, y - BAR, iResolution.x - pad - 2.0 * R - 6.0, y + BAR);
    if (b.x >= bar.x && b.x <= bar.z && b.y >= bar.y && b.y <= bar.w) {{
        bool edge = b.x == bar.x || b.x == bar.z || b.y == bar.y || b.y == bar.w;
        float end = bar.x + 2.0 + floor((bar.z - bar.x - 3.0) * progress + 0.5);
        bool fill = b.y >= bar.y + 2.0 && b.y <= bar.w - 2.0 && b.x >= bar.x + 2.0 && b.x < end;
        if (edge || fill) col = ink;
    }}
    vec2 i = b - pad;
    if (i.y >= 0.0 && i.y < ICON && i.x >= 0.0) {{
        float h = ICON;
        bool play = i.x <= min(i.y, h - 1.0 - i.y);
        bool pause = i.x < 2.0 || (i.x >= (h + 1.0) * 0.5 && i.x < (h + 1.0) * 0.5 + 2.0);
        if (playing > 0.5 ? play : pause) col = ink;
    }}
    gl_FragColor = vec4(col, 1.0);
}}
"""

INSET = """
void main() {
    vec2 b = floor(uv * iResolution.xy);
    float pad = float(padding);
    vec2 far = iResolution.xy - pad - 1.0;
    vec3 col = background;
    if (gl_FragCoord.x >= 0.0) col = texture2D(iChannel0, uv).rgb;
    float inside = step(pad, b.x) * step(pad, b.y) * step(b.x, far.x) * step(b.y, far.y);
    gl_FragColor = vec4(mix(background, col, inside), 1.0);
}
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


def rows(pad):
    """The y of the artist's layer and of the title's, which centre their
    capitals between the top line and the reels."""
    top, reels = pad + 7, H - 1 - pad - 2 * R
    if BIG:
        caps = top + (reels - top - 25) // 2
        return caps - 4, caps + 3
    return None, top + (reels - top - 7) // 2 - 4


_now = {
    "track": None,
    "time": None,
    "playing": None,
    "spin": [0.0, 0.0],
    "pad": PADDING,
}


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
    pad = int(tick.params.get("padding", _now["pad"]))
    if _now["pad"] != pad:
        _now.update(pad=pad, track=None, time=None)
        layers.count.x, layers.count.y = pad + ICON + 3, pad - 4
        layers.time.y = pad - 4
        artist_y, layers.title.y = rows(pad)
        if BIG:
            layers.artist.y = artist_y
    if _now["track"] != n:
        _now["track"] = n
        layers.count.text = f"{n + 1:02}/{len(PLAYLIST):02}"
        words = title if BIG else f"{title} - {artist}"
        fits = width(words, TITLE) <= W - 2 * pad
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
        layers.time.x = W - pad - round(width(shown))
    if _now["playing"] != playing:
        _now["playing"] = playing
        layers.time.blink.visible = not playing


def build():
    text = {"font_family": FONT, "color": ORANGE, "render": "mono"}
    top = PADDING - 4
    artist_y, title_y = rows(PADDING)
    layers = [
        ghee.Layer("count", "01/03", x=PADDING + ICON + 3, y=top, font_size=PX, **text),
        ghee.Layer(
            "time",
            "0:00",
            y=top,
            font_size=PX,
            blink={"rate": 2, "visible": False},
            **text,
        ),
        ghee.Layer(
            "title",
            PLAYLIST[0][1],
            x=0,
            y=title_y,
            font_size=TITLE,
            scroll={"speed": 2.0},
            **text,
        ),
    ]
    if BIG:
        layers.append(
            ghee.Layer("artist", PLAYLIST[0][0], y=artist_y, font_size=PX, **text)
        )
    shader = MAIN.format(r=R, hub=HUB, bar=BAR, icon=ICON)
    return ghee.Scene(
        params=[
            ghee.Param("background", "color", "#000000"),
            ghee.Param("padding", "int", PADDING, min=0, max=8),
        ],
        passes=[
            ghee.Pass(
                "main",
                shader,
                params=[
                    ghee.Param("ink", "color", ORANGE),
                    ghee.Param("dim", "color", BROWN),
                    ghee.Param("progress", "float", 0.0, min=0.0, max=1.0),
                    ghee.Param("side", "float", 0.0, min=0.0, max=1.0),
                    ghee.Param("spin", "vec2", [0.0, 0.0]),
                    ghee.Param("playing", "float", 1.0, min=0.0, max=1.0),
                ],
                layers=layers,
            ),
            ghee.Pass("inset", INSET, channels=[ghee.Channel("main")]),
        ],
    )


ghee.run(build())
