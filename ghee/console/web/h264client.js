// attachH264(canvas, url, {onconfig, onstats, onfail}) -> {stop}
// Reads the /h264 framed stream ([u8 type][u32 BE len][payload]; type 0 =
// decoder config JSON, 1 = key AU, 2 = delta AU, AVCC payloads), decodes with
// WebCodecs VideoDecoder, draws VideoFrames to the canvas.
function attachH264(canvas, url, opts) {
  opts = opts || {};
  const ctx = canvas.getContext('2d');
  let decoder = null, lastFrame = null, decoded = 0, ts = 0, stopped = false;
  let fpsLast = 0, fpsT = performance.now();
  const fail = e => { if (!stopped) { stopped = true; opts.onfail && opts.onfail(String(e)); } };

  function configure(cfg) {
    canvas.width = cfg.codedWidth; canvas.height = cfg.codedHeight;
    const desc = Uint8Array.from(atob(cfg.description), c => c.charCodeAt(0));
    decoder = new VideoDecoder({
      output: f => { decoded++; if (lastFrame) lastFrame.close(); lastFrame = f; },
      error: e => fail('decoder error: ' + e.message),
    });
    decoder.configure({
      codec: cfg.codec,
      description: desc,            // avcC: cross-browser AVCC mode
      codedWidth: cfg.codedWidth,
      codedHeight: cfg.codedHeight,
      optimizeForLatency: true,
      hardwareAcceleration: 'no-preference',
    });
    opts.onconfig && opts.onconfig(cfg);
  }

  function draw() {
    if (stopped) return;
    if (lastFrame) ctx.drawImage(lastFrame, 0, 0, canvas.width, canvas.height);
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
    const res = await fetch(url);
    if (!res.ok) throw new Error('h264 stream: HTTP ' + res.status);
    const reader = res.body.getReader();
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

  return { stop: () => { stopped = true; if (lastFrame) lastFrame.close(); } };
}
