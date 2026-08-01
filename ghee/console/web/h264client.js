// attachH264(canvas, url, {onconfig, onstats, onfail}) -> {stop}
// Reads the /h264 framed stream ([u8 type][u32 BE len][payload]; type 0 =
// decoder config JSON, 1 = key AU, 2 = delta AU, AVCC payloads), decodes with
// WebCodecs VideoDecoder, draws VideoFrames to the canvas.
function attachH264(canvas, url, opts) {
  opts = opts || {};
  const ctx = canvas.getContext('2d');
  const abort = new AbortController();
  let decoder = null, reader = null, lastFrame = null, decoded = 0, ts = 0, stopped = false;
  let fpsLast = 0, fpsT = performance.now();
  // aborting the request is what drops this viewer server-side
  function teardown() {
    stopped = true;
    abort.abort();
    if (reader) { reader.cancel().catch(() => {}); reader = null; }
    if (decoder) { if (decoder.state !== 'closed') decoder.close(); decoder = null; }
    if (lastFrame) { lastFrame.close(); lastFrame = null; }
  }
  const fail = e => { if (!stopped) { teardown(); opts.onfail && opts.onfail(String(e)); } };

  function configure(cfg) {
    // a mid-stream config (encoder respawn, size change) replaces the decoder;
    // the old one has to be closed or it keeps its hardware slot
    if (decoder && decoder.state !== 'closed') decoder.close();
    canvas.width = cfg.codedWidth; canvas.height = cfg.codedHeight;
    const desc = Uint8Array.from(atob(cfg.description), c => c.charCodeAt(0));
    decoder = new VideoDecoder({
      output: f => { decoded++; if (lastFrame) lastFrame.close(); lastFrame = f; },
      error: e => fail('decoder error: ' + e.message),
    });
    const dc = {
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
    if (!res.ok) throw new Error('h264 stream: HTTP ' + res.status);
    reader = res.body.getReader();
    let buf = new Uint8Array(0);
    const cat = (a, b) => { const o = new Uint8Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
    for (;;) {
      const { value, done } = await reader.read();
      if (done || stopped) { if (!stopped) fail('stream ended'); break; }
      buf = cat(buf, value);
      for (;;) {
        if (buf.length < 5) break;
        const type = buf[0];
        const len = (buf[1] << 24 | buf[2] << 16 | buf[3] << 8 | buf[4]) >>> 0;
        if (buf.length < 5 + len) break;
        const payload = buf.slice(5, 5 + len);
        buf = buf.slice(5 + len);
        if (type === 0) configure(JSON.parse(new TextDecoder().decode(payload)));
        else if (decoder && decoder.state === 'configured') {
          decoder.decode(new EncodedVideoChunk({
            type: type === 1 ? 'key' : 'delta',
            timestamp: (ts++) * 33333,
            data: payload,
          }));
        }
      }
    }
  })().catch(fail);

  return { stop: teardown };
}
