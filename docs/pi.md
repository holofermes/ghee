# Raspberry Pi

ghee runs on every Raspberry Pi, on its GPU. Pi GPUs come in two kinds: a
Pi Zero to Pi 3 has a VideoCore IV, VC4 for short, and a Pi 4, Pi 400 or
Pi 5 has a VideoCore VI or VII, V3D for short. Write a scene for VC4, and it
runs on every Pi.

## The same on every Pi

These hold on every Pi, set by what VC4 can do:

- A pass's GLSL is GLSL ES 1.00, because VC4 runs OpenGL ES 2.0
  ([Passes](scenes.md#passes)).
- A pass's buffer holds 8 bits for each of red, green, blue and alpha, the
  most that VC4 draws.
- A pass reads at most 16 textures, VC4's count.
- `fwidth`, `dFdx` and `dFdy` don't compile in a pass.

## Where the Pis differ

For a scene that must run on every Pi, keep to the VC4 column.

| | VC4: Pi Zero to Pi 3 | V3D: Pi 4, Pi 5 |
| --- | --- | --- |
| a frame, a pass, an image or an [array](scenes.md#arrays) | 2048 pixels a side at most | 4096 pixels a side at most |
| the values of an [array](scenes.md#arrays) that a program feeds | 8 bits each, so a value outside 0 to 1 clamps | 32-bit floats |
| a [data file](scenes.md#data-files), `.raw`, in `f32` | fails to load, so use `u8` | loads, and is read `nearest` |
| GPU memory | set aside at boot ([GPU memory](#gpu-memory)) | the system's RAM |

To fill an HDMI screen larger than 2048 pixels on VC4, draw at 1920x1080 or
less with `--size`, and the screen stretches the frame
([Hardware displays](displays.md#hardware-displays)).

## Shaders on VC4

### Wrong 4x4 blocks

On VC4, a pass that reads a texture and then compares a value, with `step`,
a ternary or `<`, can draw 4x4 blocks of wrong pixels, a new set each frame.
Mesa 26.2.5 and later fix it
([mesa!45037](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/45037)).
To see which Mesa you have, run:

```shell
dpkg -s libgl1-mesa-dri | grep Version
```

On an older Mesa, or for a Pi whose Mesa you don't know, put the pass's last
texture read inside an `if` that always holds:

```glsl
vec4 c = vec4(0.0);
if (gl_FragCoord.x >= 0.0) c = texture2D(iChannel0, uv);
```

ghee's own layers and filters already read this way.

### Copying a pass exactly

On VC4, a `linear` read lands exactly on a pixel only when the buffer's
sides are powers of two. At other sizes, each read mixes in a little of the
neighbouring pixels, and a pass that reads its own last frame adds that up,
frame after frame. To copy a buffer exactly, read it with `"sampling":
"nearest"`, or give it sides that are powers of two
([Size and sampling](scenes.md#size-and-sampling)).

### Pixel positions

On VC4, `gl_FragCoord.xy / iResolution.xy` lands just under the pixel's
centre, because the GPU divides approximately. Use `uv`, which ghee sets on
the exact centre ([Built-in names](scenes.md#built-in-names)).

### sin, cos and division

On VC4, `sin`, `cos` and division are approximate. A smooth picture moves by
one level at most, but these cases move further. In the table, `p` is the
pixel's whole coordinates, `floor(uv * iResolution.xy)`:

| Shader | On VC4 | Use instead |
| --- | --- | --- |
| an edge on `sin` or `cos`, such as `step(0.0, sin(x))` | the edge moves | `sincos` from [trig.glsl](../tools/compare/trig.glsl), pasted into the pass |
| a block, `floor(p / cell)` | a pixel in a block's first column lands in the block before | `floor((p + 0.5) / cell)` |
| noise from `fract(sin(x) * 43758.5453)` | other noise than on every other GPU | `ghee_hash` and `ghee_noise` ([Built-in names](scenes.md#built-in-names)) |

### Text in a shader

On VC4, a shader that calls a text layer's `_cover` more than once fails to
compile. Call it once in each pass, and read the words of a second layer in
a pass of its own ([Text in a shader](layers.md#text-in-a-shader)).

## GPU memory

### On VC4

On a Pi with 512 MiB, such as a Zero 2 W or a Pi 3 A+, set the GPU's
memory, CMA, to 96 MiB. In `/boot/firmware/config.txt`, change the
`dtoverlay=vc4-kms-v3d` line to this one, and reboot:

```
dtoverlay=vc4-kms-v3d,cma-96
```

Raspberry Pi OS sets aside 256 MiB, which leaves a 512 MiB Pi short of
memory: on a Zero 2 W, ghee's first frame took 4 to 6 s, and 0.9 s with
`cma-96`. On a Zero 2 W with `cma-96`, the heaviest bundled demo measured,
`feedback` at 1920x1080, took 73 MiB.

### On V3D

The GPU takes its buffers from the system's RAM, and CMA can stay as it is.
`ghee render --stats` prints the scene's GPU memory, `gpu`, in MiB, when the
render ends.

## A Pi 5's GPU sleeps

The kernel powers a Pi 5's GPU down 50 ms after its last frame. After a
still stretch, the next change reaches the display about 5 ms later, while
the GPU wakes. The board draws about 0.18 W less while the GPU sleeps. To
keep it awake until the next boot:

```shell
echo on | sudo tee /sys/devices/platform/axi/1002000000.v3d/power/control
```
