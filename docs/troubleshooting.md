# Troubleshooting

**`ghee render` says there is no GL device.** Point the engine at the right
node with `--dri` or `$GHEE_DRI`, and add your user to the `video` group, plus
`render` on a desktop.

**`--h264` or the console stream says there is no ffmpeg on PATH.** That board
has no Broadcom encoder, so ghee encodes with ffmpeg and libx264:
`sudo apt install ffmpeg`.

**`--display kms` cannot take DRM master.** ghee needs the card to itself,
and a compositor holds it, almost always a desktop session.

**The console preview falls back to MJPEG in a browser that has WebCodecs.**
H264 playback needs a secure context. `127.0.0.1` is one, and a plain-HTTP LAN
address is not.
