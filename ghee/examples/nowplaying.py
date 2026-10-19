#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""Now playing on a small panel, as a cassette player's screen: the title,
the track count, the time played and the track's length, and the tape
running between two turning reels.

The pass `words` holds the text layers, in monogram. The pass `main` shows
the words through a window between two reels and draws the reels, the play or
pause sign and the bar from its params: `progress` fills the bar, `side`
winds the tape from the left reel onto the right one over the whole playlist,
and `spin` turns each reel. The scene param `reels` sizes the reels, 1 as
tall as the room under the top line, and the window takes the width between
them. The layout follows ghee.render_size(): a 256x64 SSD1322 shows the artist
on a line of its own, a 128x32 SSD1306 runs it after the title. The scene
param `padding` insets the whole layout from the frame's edges by that many
pixels, painted in `background`.

The tick hook stands in for a music player. It plays a stretch of each track
of PLAYLIST, pauses, and moves on, sets the params, and sets the words of the
layers through `ghee.layers`, which ghee lays out again with no recompile. A
title too long for the window scrolls behind the reels. Put a player's own
state in now_playing().

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
REELS = 0.8
BAR = 4 if BIG else 3  # the bar's rows above and below its middle row
GAP = 3  # pixels between a reel and the window between the reels
TITLE = 2 * PX if BIG else PX

MAIN = """
const float BAR = {bar:.1f};
const float GAP = {gap:.1f};

void reel(vec2 p, vec2 c, float r, float wound, float turn, inout vec3 col) {{
    vec2 q = p - c;
    float d = length(q);
    float hub = max(2.0, floor(0.4 * r + 0.5));
    float pack = sqrt(mix((hub + 2.0) * (hub + 2.0), (r - 1.0) * (r - 1.0), wound));
    float tooth = mod(atan(q.y, q.x) - turn, 2.0944) - 1.0472;
    float rail = r - BAR;
    if (abs(q.y - rail) < 0.5 && abs(q.x) > sqrt(r * r - rail * rail) && abs(q.x) < r + GAP - 1.0) col = dim;
    if (d < r + 0.5) col = background;
    if (abs(d - r) < 0.5) col = dim;
    if (d > hub + 0.5 && d < pack - 0.5) col = dim;
    if (abs(d - pack) < 0.5) col = ink;
    if (abs(d - hub) < 0.5) col = ink;
    if (d < hub && d > min(hub - 2.0, 0.5 * hub) && abs(d * sin(tooth)) < 0.6) col = ink;
}}

void main() {{
    float pad = float(padding);
    vec2 b = floor(vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y));
    vec2 far = iResolution.xy - pad - 1.0;
    float r = clamp(floor(reels * floor((iResolution.y - 10.0 - 2.0 * pad) / 2.0) + 0.5), 3.0, floor((iResolution.y - 10.0 - 2.0 * pad) / 2.0));
    vec3 col = background;
    if (gl_FragCoord.x >= 0.0) col = texture2D(iChannel0, uv).rgb;
    vec2 window = vec2(pad + 2.0 * r + 1.0 + GAP, far.x - 2.0 * r - 1.0 - GAP);
    if (b.y >= pad + 8.0 && (b.x < window.x || b.x > window.y)) col = background;
    float y = far.y - r;
    reel(b, vec2(pad + r, y), r, 1.0 - side, spin.x, col);
    reel(b, vec2(far.x - r, y), r, side, spin.y, col);
    float h = 2.0 * BAR + 1.0;
    vec2 i = vec2(b.x - window.x, b.y - (far.y + 1.0 - h));
    if (i.y >= 0.0 && i.y < h && i.x >= 0.0) {{
        bool play = i.x <= min(i.y, h - 1.0 - i.y);
        bool pause = i.x < 2.0 || (i.x >= (h + 1.0) * 0.5 - 1.0 && i.x < (h + 1.0) * 0.5 + 1.0);
        if (playing > 0.5 ? play : pause) col = ink;
    }}
    vec4 bar = vec4(window.x + (h + 1.0) * 0.5 + 3.0, far.y + 1.0 - h, window.y, far.y);
    if (b.x >= bar.x && b.x <= bar.z && b.y >= bar.y && b.y <= bar.w) {{
        bool edge = b.x == bar.x || b.x == bar.z || b.y == bar.y || b.y == bar.w;
        float end = bar.x + 2.0 + floor((bar.z - bar.x - 3.0) * progress + 0.5);
        bool fill = b.y >= bar.y + 2.0 && b.y <= bar.w - 2.0 && b.x >= bar.x + 2.0 && b.x < end;
        if (edge || fill) col = ink;
    }}
    float inside = step(pad, b.x) * step(pad, b.y) * step(b.x, far.x) * step(b.y, far.y);
    gl_FragColor = vec4(mix(background, col, inside), 1.0);
}}
"""

WORDS = """
void main() {
    gl_FragColor = vec4(background, 1.0);
}
"""


def radius(reels, pad):
    """A reel's radius from its centre pixel: `reels` of the most that fits
    under the top line."""
    most = (H - 10 - 2 * pad) // 2
    return min(max(math.floor(reels * most + 0.5), 3), most)


def window(reels, pad):
    """The width between the reels, where the words and the bar show."""
    return W - 2 * (pad + 2 * radius(reels, pad) + 1 + GAP)


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
    capitals and the title's descenders between the top line and the bar."""
    top, bar = pad + 9, H - 1 - pad - 2 * BAR
    if BIG:
        caps = top + (bar - top - 29) // 2
        return caps - 4, caps + 3
    return None, top + (bar - top - 9) // 2 - 4


_now = {
    "track": None,
    "time": None,
    "playing": None,
    "spin": [0.0, 0.0],
    "pad": PADDING,
    "reels": REELS,
}


@ghee.tick
def player(tick):
    """Turn the reels, fill the bar, and set the words when they change."""
    n, artist, title, length, pos, playing, side = now_playing(tick.time)
    if playing:
        for k, wound in enumerate((1 - side, side)):
            pack = math.sqrt(BARE**2 + wound * (1 - BARE**2))
            _now["spin"][k] += tick.dt * TAPE / pack
    ghee.set_param("spin", _now["spin"])
    ghee.set_param("progress", pos / length)
    ghee.set_param("side", side)
    ghee.set_param("playing", 1.0 if playing else 0.0)
    layers = ghee.layers
    pad = int(tick.params.get("padding", _now["pad"]))
    reels = float(tick.params.get("reels", _now["reels"]))
    if _now["reels"] != reels:
        _now.update(reels=reels, track=None)
    if _now["pad"] != pad:
        _now.update(pad=pad, track=None, time=None)
        layers.count.x, layers.count.y = pad, pad - 4
        layers.time.y = pad - 4
        artist_y, layers.title.y = rows(pad)
        if BIG:
            layers.artist.y = artist_y
    if _now["track"] != n:
        _now["track"] = n
        layers.count.text = f"{n + 1:02}/{len(PLAYLIST):02}"
        words = title if BIG else f"{title} - {artist}"
        fits = width(words, TITLE) <= window(reels, pad)
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
        ghee.Layer("count", "01/03", x=PADDING, y=top, font_size=PX, **text),
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
    shader = MAIN.format(bar=BAR, gap=GAP)
    return ghee.Scene(
        params=[
            ghee.Param("background", "color", "#000000"),
            ghee.Param("padding", "int", PADDING, min=0, max=8),
            ghee.Param("reels", "float", REELS, min=0.0, max=1.0),
        ],
        passes=[
            ghee.Pass("words", WORDS, layers=layers),
            ghee.Pass(
                "main",
                shader,
                channels=[ghee.Channel("words")],
                params=[
                    ghee.Param("ink", "color", ORANGE),
                    ghee.Param("dim", "color", BROWN),
                    ghee.Param("progress", "float", 0.0, min=0.0, max=1.0),
                    ghee.Param("side", "float", 0.0, min=0.0, max=1.0),
                    ghee.Param("spin", "vec2", [0.0, 0.0]),
                    ghee.Param("playing", "float", 1.0, min=0.0, max=1.0),
                ],
            ),
        ],
    )


ghee.run(build())
