// attachH264(canvas, url, {onconfig, onstats, onfail, startMs}) -> {stop}
// Reads the /h264 framed stream ([u8 type][u32 BE len][payload]; type 0 =
// decoder config JSON, 1 = key AU, 2 = delta AU, AVCC payloads, 3 = the UTF-8
// reason the stream ends), decodes with WebCodecs VideoDecoder, draws
// VideoFrames to the canvas. A stream with no decoded frame startMs after the
// attach (default 5000) counts as failed.
function attachH264(canvas, url, opts) {
  opts = opts || {};
  const startMs = opts.startMs || 5000;
  const ctx = canvas.getContext('2d');
  const abort = new AbortController();
  let decoder = null, reader = null, lastFrame = null, decoded = 0, ts = 0, stopped = false;
  let fpsLast = 0, fpsT = performance.now();
  // aborting the request is what drops this viewer server-side
  function teardown() {
    stopped = true;
    clearTimeout(startTimer);
    abort.abort();
    if (reader) { reader.cancel().catch(() => {}); reader = null; }
    if (decoder) { if (decoder.state !== 'closed') decoder.close(); decoder = null; }
    if (lastFrame) { lastFrame.close(); lastFrame = null; }
  }
  const fail = e => { if (!stopped) { teardown(); opts.onfail && opts.onfail(String(e)); } };
  const startTimer = setTimeout(() => fail('no frame in ' + startMs / 1000 + ' s'), startMs);

  async function configure(cfg) {
    // a mid-stream config (a resync, a size change) replaces the decoder;
    // the old one has to be closed or it keeps its hardware slot
    if (decoder && decoder.state !== 'closed') decoder.close();
    decoder = null;
    canvas.width = cfg.codedWidth; canvas.height = cfg.codedHeight;
    const desc = Uint8Array.from(atob(cfg.description), c => c.charCodeAt(0));
    let dc = {
      codec: cfg.codec,
      description: desc,            // avcC: cross-browser AVCC mode
      codedWidth: cfg.codedWidth,
      codedHeight: cfg.codedHeight,
      optimizeForLatency: true,
      hardwareAcceleration: 'no-preference',
    };
    // the VC4 hardware stream carries no VUI colour description; without the
    // hint browsers guess BT.709 limited
    if (cfg.colorSpace) dc.colorSpace = cfg.colorSpace;
    // Chrome decodes a stream under 360 rows in software (decoder_selector.cc,
    // kSoftwareDecoderHeightCutoff). A hardware decoder can scale the padding
    // of a small stream into the picture (Firefox bug 2054811).
    if (cfg.codedHeight < 360) {
      const sw = Object.assign({}, dc, { hardwareAcceleration: 'prefer-software' });
      const s = await VideoDecoder.isConfigSupported(sw).catch(() => null);
      if (s && s.supported) dc = sw;
    }
    if (stopped) return;
    decoder = new VideoDecoder({
      output: f => {
        decoded++; clearTimeout(startTimer);
        if (lastFrame) lastFrame.close();
        lastFrame = f;
      },
      error: e => fail('decoder error: ' + e.message),
    });
    decoder.configure(dc);
    opts.onconfig && opts.onconfig(cfg);
  }

  function draw() {
    if (stopped) return;
    if (lastFrame) {
      // a dimension that is not a multiple of 16 decodes to a padded frame the
      // SPS crops back; some decoders hand back the padding, so the source
      // rect is explicit rather than the frame's full coded size
      const r = lastFrame.visibleRect ||
        { x: 0, y: 0, width: lastFrame.displayWidth, height: lastFrame.displayHeight };
      if (canvas.width !== lastFrame.displayWidth) canvas.width = lastFrame.displayWidth;
      if (canvas.height !== lastFrame.displayHeight) canvas.height = lastFrame.displayHeight;
      ctx.drawImage(lastFrame, r.x, r.y, r.width, r.height, 0, 0, canvas.width, canvas.height);
    }
    const now = performance.now();
    if (now - fpsT > 500) {   // fps = decoded frames, not canvas redraws
      opts.onstats && opts.onstats({
        decoded, fps: Math.round((decoded - fpsLast) * 1000 / (now - fpsT)),
        queue: decoder ? decoder.decodeQueueSize : 0,
      });
      fpsLast = decoded; fpsT = now;
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);

  (async () => {
    const res = await fetch(url, { signal: abort.signal });
    if (!res.ok) {
      const why = await res.text().catch(() => '');
      throw new Error('h264 stream: HTTP ' + res.status + (why ? ': ' + why : ''));
    }
    reader = res.body.getReader();
    // the bytes of a frame that the last read cut short
    let rest = new Uint8Array(0);
    for (;;) {
      const { value, done } = await reader.read();
      if (done || stopped) { if (!stopped) fail('stream ended'); break; }
      let buf = value;
      if (rest.length) { buf = new Uint8Array(rest.length + value.length); buf.set(rest); buf.set(value, rest.length); }
      let at = 0;
      while (buf.length - at >= 5) {
        const type = buf[at];
        const len = (buf[at + 1] << 24 | buf[at + 2] << 16 | buf[at + 3] << 8 | buf[at + 4]) >>> 0;
        if (buf.length - at < 5 + len) break;
        const payload = buf.subarray(at + 5, at + 5 + len);
        at += 5 + len;
        if (type === 0) await configure(JSON.parse(new TextDecoder().decode(payload)));
        else if (type === 3) { fail(new TextDecoder().decode(payload)); return; }
        else if (decoder && decoder.state === 'configured') {
          decoder.decode(new EncodedVideoChunk({
            type: type === 1 ? 'key' : 'delta',
            timestamp: (ts++) * 33333,
            data: payload,
          }));
        }
        if (stopped) return;
      }
      rest = buf.slice(at);
    }
  })().catch(fail);

  return { stop: teardown };
}
