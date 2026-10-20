// attachH264(canvas, url, {onconfig, onstats, onframe, ondata, onfail, startMs}) -> {stop}
// Reads the /h264 framed stream ([u8 type][u32 BE len][payload]; type 0 =
// decoder config JSON, 1 = key AU, 2 = delta AU, AVCC payloads, 3 = the UTF-8
// reason the stream ends, 4 = idle), decodes with WebCodecs VideoDecoder, draws
// VideoFrames to the canvas. onframe runs at the first decoded frame, ondata
// at each read. A stream with no decoded frame startMs after the attach
// (default 5000) counts as failed. onfail(why, dropped): dropped is true when
// the connection failed or ended, false when the server or the decoder refused.
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
  const fail = (e, dropped) => { if (!stopped) { teardown(); opts.onfail && opts.onfail(String(e), !!dropped); } };
  const startTimer = setTimeout(() => fail('no frame in ' + startMs / 1000 + ' s'), startMs);

  async function configure(cfg) {
    // a mid-stream config (a resync, a size change) replaces the decoder;
    // the old one has to be closed or it keeps its hardware slot
    if (decoder && decoder.state !== 'closed') decoder.close();
    decoder = null;
    // a resize clears the canvas, and a restarted stream keeps the last picture
    if (canvas.width !== cfg.codedWidth) canvas.width = cfg.codedWidth;
    if (canvas.height !== cfg.codedHeight) canvas.height = cfg.codedHeight;
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
        if (decoded === 1 && opts.onframe) opts.onframe();
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
    let res;
    try { res = await fetch(url, { signal: abort.signal }); } catch (e) { return fail(e, true); }
    if (!res.ok) {
      const why = await res.text().catch(() => '');
      throw new Error('h264 stream: HTTP ' + res.status + (why ? ': ' + why : ''));
    }
    reader = res.body.getReader();
    let rest = new Uint8Array(0);
    for (;;) {
      let got;
      try { got = await reader.read(); } catch (e) { return fail(e, true); }
      const { value, done } = got;
      if (done || stopped) { if (!stopped) fail('stream ended', true); break; }
      if (opts.ondata) opts.ondata();
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
        else if ((type === 1 || type === 2) && decoder && decoder.state === 'configured') {
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

// The index of the bytes pat in buf at or after from, -1 when absent.
function findBytes(buf, pat, from) {
  for (let i = buf.indexOf(pat[0], from); i >= 0 && i + pat.length <= buf.length; i = buf.indexOf(pat[0], i + 1)) {
    let k = 1;
    while (k < pat.length && buf[i + k] === pat[k]) k++;
    if (k === pat.length) return i;
  }
  return -1;
}

// attachMjpeg(url, {onjpeg, ondata, onfail}) -> {stop}
// Reads the multipart /stream and hands each JPEG to onjpeg as bytes. ondata
// runs at each read, onfail(why) when the stream fails or ends.
function attachMjpeg(url, opts) {
  const abort = new AbortController();
  let stopped = false;
  const stop = () => { stopped = true; abort.abort(); };
  const fail = e => { if (!stopped) { stop(); opts.onfail(String(e)); } };
  (async () => {
    const res = await fetch(url, { signal: abort.signal });
    if (!res.ok) throw new Error('mjpeg stream: HTTP ' + res.status);
    const b = /boundary="?([^";]+)/.exec(res.headers.get('Content-Type') || '');
    if (!b) throw new Error('mjpeg stream: no multipart boundary');
    const enc = new TextEncoder(), gap = enc.encode('\r\n\r\n'), end = enc.encode('\r\n--' + b[1]);
    const reader = res.body.getReader();
    // body: where the part's JPEG starts, -1 while its headers come
    let buf = new Uint8Array(0), body = -1, from = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (stopped) return;
      if (done) throw new Error('stream ended');
      opts.ondata();
      const joined = new Uint8Array(buf.length + value.length);
      joined.set(buf); joined.set(value, buf.length); buf = joined;
      for (;;) {
        if (body < 0) {
          const i = findBytes(buf, gap, from);
          if (i < 0) { from = Math.max(0, buf.length - gap.length + 1); break; }
          body = from = i + gap.length;
        }
        const i = findBytes(buf, end, from);
        if (i < 0) { from = Math.max(body, buf.length - end.length + 1); break; }
        opts.onjpeg(buf.slice(body, i));
        buf = buf.slice(i + 2); body = -1; from = 0;
      }
    }
  })().catch(fail);
  return { stop };
}

// playPreview(img, canvas, base, vid, onchange) -> {stop, source, why}
// Plays a console's preview: its /h264 stream on the canvas when the browser
// decodes H264 (WebCodecs, in a secure context), its /stream MJPEG on the img
// otherwise, and when an H264 stream fails before its first frame. A stream
// that drops, ends or goes quiet after a frame opens again. base is the
// console's origin, "" for the page's own. The page sends the vid's /ping.
// onchange runs when the preview switches and when the picture's size
// arrives. source() gives the kind ("h264", "mjpeg") and the picture's size,
// 0 x 0 before it arrives, why() the reason for MJPEG.
function playPreview(img, canvas, base, vid, onchange) {
  const changed = () => { if (onchange) onchange(); };
  // both streams send something each second
  const QUIET_MS = 3000;
  let lane = null, h264 = false, why = '', stopped = false;
  // shown: the lane drew a frame. tries: the restarts since the last frame.
  let shown = false, tries = 0, retry = 0, quiet = 0;
  // the img's object URLs: the one it shows, the one it decodes
  let url = '', pending = '', next = null, w = 0, h = 0;
  function show(el) {
    img.style.display = el === img ? '' : 'none';
    canvas.style.display = el === canvas ? '' : 'none';
  }
  function drop() {
    clearTimeout(quiet);
    if (lane) { lane.stop(); lane = null; }
  }
  function heard() {
    clearTimeout(quiet);
    if (shown) quiet = setTimeout(restart, QUIET_MS);
  }
  function drew() { shown = true; tries = 0; heard(); }
  function restart() {
    drop();
    clearTimeout(retry);
    if (!stopped) retry = setTimeout(h264 ? playH264 : playMjpeg, Math.min(8000, 250 * 2 ** tries++));
  }
  function playH264() {
    let first = false;
    lane = attachH264(canvas, base + '/h264?vid=' + vid, {
      onconfig: changed,
      ondata: () => { if (first) heard(); },
      onframe: () => { first = true; drew(); },
      onfail: (e, dropped) => {
        lane = null;
        if (dropped && (first || shown)) restart();
        else { drop(); mjpeg('h264 preview dropped: ' + e); }
      },
    });
  }
  // the img decodes one JPEG at a time, and the newest that waits goes next
  function put(jpeg) {
    if (pending) { next = jpeg; return; }
    pending = URL.createObjectURL(new Blob([jpeg], { type: 'image/jpeg' }));
    img.src = pending;
  }
  function decoded(ok) {
    if (ok) {
      if (url) URL.revokeObjectURL(url);
      url = pending;
      if (img.naturalWidth !== w || img.naturalHeight !== h) {
        w = img.naturalWidth; h = img.naturalHeight;
        changed();
      }
    } else URL.revokeObjectURL(pending);
    pending = '';
    if (next) { const j = next; next = null; put(j); }
  }
  function playMjpeg() {
    lane = attachMjpeg(base + '/stream?vid=' + vid, {
      ondata: heard,
      onjpeg: j => { drew(); put(j); },
      onfail: () => { lane = null; restart(); },
    });
    heard();
  }
  function mjpeg(reason) {
    why = reason; h264 = false; shown = false; tries = 0;
    show(img);
    img.onload = () => decoded(true);
    img.onerror = () => decoded(false);
    changed();
    playMjpeg();
  }
  if ('VideoDecoder' in window && window.isSecureContext) {
    h264 = true;
    show(canvas);
    canvas.width = canvas.height = 0;
    playH264();
    changed();
  } else {
    mjpeg(window.isSecureContext ? 'this browser has no WebCodecs VideoDecoder'
      : 'h264 preview needs https or localhost');
  }
  return {
    stop() {
      stopped = true;
      clearTimeout(retry);
      drop();
      img.onload = img.onerror = null;
      img.removeAttribute('src');
      for (const u of [url, pending]) if (u) URL.revokeObjectURL(u);
      url = pending = ''; next = null;
    },
    source() {
      if (canvas.style.display !== 'none') return { kind: 'h264', w: canvas.width, h: canvas.height };
      return { kind: 'mjpeg', w: img.naturalWidth, h: img.naturalHeight };
    },
    why: () => why,
  };
}
