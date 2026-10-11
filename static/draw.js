/* StoryForge: draw your hero.
   A canvas beside the toolbar, driven by pointer events (finger, pen or mouse). Everything drawn
   (marker, crayon, pencil and eraser strokes as point lists; fills as the spot tapped) is kept in
   order, so undo, a turned screen and Done simply draw it again. Done crops the drawing to its edges,
   keeps the transparency (that PNG becomes the hero layered over every background) and sends a
   copy on white for the vision model, which describes the hero for the storyteller. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  const STORAGE_STORY = "storyforge.story"; // the same key the storybook uses to reopen a story
  const STORAGE_LEVEL = "storyforge.level"; // the reading level, shared with the cover
  const GIVE_UP_MS = 45000; // the loading screen never waits longer than this

  const state = {
    // Everything drawn, in order, so undo, a turned screen and Done can redraw it at any size:
    //   { tool: "marker" | "crayon" | "pencil" | "eraser", color, width, points: [[x, y], ...] }
    //   { tool: "fill", color, x, y, box: { x0, y0, x1, y1 } }   (where it was tapped, and what it covered)
    // in CSS pixels of the paper.
    ops: [],
    cleared: null,      // the drawing removed by Clear, so Undo can bring it back
    current: null,      // the stroke being drawn
    pointerId: null,    // only one finger draws at a time
    tool: "marker",
    color: "#2c2340",
    sizeFactor: 1,
    size: { w: 0, h: 0, dpr: 1 },
    sending: false,
  };
  // Everything already drawn, as a picture, so only the stroke in progress is drawn again while
  // drawing (fills are slow to repeat). Rebuilt on undo, Clear and a turned screen.
  const cache = document.createElement("canvas");
  const cctx = cache.getContext("2d");

  /* ───────── tools ───────── */

  // Brush width for each tool, from the chosen size (thin, medium, thick).
  const WIDTH = { marker: 1, crayon: 1.15, pencil: 0.45, eraser: 1.6 };

  function brushWidth(tool) {
    const base = Math.max(10, Math.min(state.size.w, state.size.h) * 0.022) * state.sizeFactor;
    return Math.max(3, base * WIDTH[tool]);
  }

  // Crayon and pencil grain: a small tile of the colour with speckles of lighter alpha, the same
  // every time (a fixed seed), so a stroke looks the same after undo, turning or Done.
  const grainTiles = new Map();
  function grainTile(tool, color) {
    const key = `${tool}${color}`;
    if (grainTiles.has(key)) return grainTiles.get(key);
    const size = 48;
    const tile = document.createElement("canvas");
    tile.width = tile.height = size;
    const t = tile.getContext("2d");
    t.fillStyle = color;
    t.fillRect(0, 0, size, size);
    const img = t.getImageData(0, 0, size, size);
    let seed = tool === "crayon" ? 7 : 11;
    const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    for (let i = 3; i < img.data.length; i += 4) {
      const r = random();
      if (tool === "crayon") img.data[i] = r < 0.2 ? 40 + random() * 90 : r < 0.3 ? 200 : 255;
      else img.data[i] = r < 0.14 ? 150 : 245; // pencil: opaque enough to stay in the cut-out hero
    }
    t.putImageData(img, 0, 0);
    grainTiles.set(key, tile);
    return tile;
  }

  function strokePath(c, pts) {
    c.beginPath();
    c.moveTo(pts[0][0], pts[0][1]);
    // Smooth curves through the midpoints between samples.
    for (let i = 1; i < pts.length - 1; i++) {
      const mx = (pts[i][0] + pts[i + 1][0]) / 2;
      const my = (pts[i][1] + pts[i + 1][1]) / 2;
      c.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
    }
    const last = pts[pts.length - 1];
    c.lineTo(last[0], last[1]);
  }

  function drawOp(c, op) {
    if (op.tool === "fill") { floodFill(c, op.x, op.y, op.color, false); return; }
    c.save();
    c.globalCompositeOperation = op.tool === "eraser" ? "destination-out" : "source-over";
    const paint = op.tool === "crayon" || op.tool === "pencil"
      ? c.createPattern(grainTile(op.tool, op.color), "repeat") : op.color;
    c.strokeStyle = c.fillStyle = op.tool === "eraser" ? "#000" : paint;
    c.lineWidth = op.width;
    c.lineCap = c.lineJoin = "round";
    const pts = op.points;
    if (pts.length === 1) {
      c.beginPath();
      c.arc(pts[0][0], pts[0][1], op.width / 2, 0, Math.PI * 2);
      c.fill();
    } else {
      strokePath(c, pts);
      c.stroke();
    }
    c.restore();
  }

  // Paint bucket: fills the area of similar colour around (x, y), in the canvas's own pixels, with a
  // small tolerance plus one pixel more so the soft edges of lines fill too. With `refuseEdge`, an
  // area that reaches the edge of the paper (the empty paper around the hero) is left alone: the
  // hero would otherwise become a solid rectangle. Returns the filled box in canvas pixels, or null.
  function floodFill(c, x, y, color, refuseEdge) {
    const W = c.canvas.width, H = c.canvas.height;
    const p = c.getTransform().transformPoint(new DOMPoint(x, y));
    const sx = Math.floor(p.x), sy = Math.floor(p.y);
    if (sx < 0 || sy < 0 || sx >= W || sy >= H) return null;
    const img = c.getImageData(0, 0, W, H);
    const d = img.data;
    const s = (sy * W + sx) * 4;
    const seed = [d[s], d[s + 1], d[s + 2], d[s + 3]];
    const rgb = [parseInt(color.slice(1, 3), 16), parseInt(color.slice(3, 5), 16), parseInt(color.slice(5, 7), 16)];
    if (seed[3] > 200 && Math.abs(seed[0] - rgb[0]) + Math.abs(seed[1] - rgb[1]) + Math.abs(seed[2] - rgb[2]) < 30) return null;
    const TOL = 64;
    const mask = new Uint8Array(W * H);
    const like = (i) => {
      const k = i * 4;
      return !mask[i] && Math.abs(d[k] - seed[0]) <= TOL && Math.abs(d[k + 1] - seed[1]) <= TOL
        && Math.abs(d[k + 2] - seed[2]) <= TOL && Math.abs(d[k + 3] - seed[3]) <= TOL;
    };
    let x0 = sx, x1 = sx, y0 = sy, y1 = sy;
    const stack = [sx, sy];
    while (stack.length) {
      const py = stack.pop(), px = stack.pop();
      if (!like(py * W + px)) continue; // already filled from another span
      let l = px;
      while (l > 0 && like(py * W + l - 1)) l--;
      let r = px;
      while (r < W - 1 && like(py * W + r + 1)) r++;
      for (let i = l; i <= r; i++) mask[py * W + i] = 1;
      if (refuseEdge && (l === 0 || r === W - 1 || py === 0 || py === H - 1)) return null;
      x0 = Math.min(x0, l); x1 = Math.max(x1, r); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
      for (const ny of [py - 1, py + 1]) {
        if (ny < 0 || ny >= H) continue;
        let i = l;
        while (i <= r) {
          if (like(ny * W + i)) {
            stack.push(i, ny);
            while (i <= r && like(ny * W + i)) i++;
          }
          i++;
        }
      }
    }
    // One pixel more all round, into the soft edges of the lines around it.
    const [cr, cg, cb] = rgb;
    const paint = (i) => { const k = i * 4; d[k] = cr; d[k + 1] = cg; d[k + 2] = cb; d[k + 3] = 255; };
    for (let yy = Math.max(0, y0 - 1); yy <= Math.min(H - 1, y1 + 1); yy++) {
      for (let xx = Math.max(0, x0 - 1); xx <= Math.min(W - 1, x1 + 1); xx++) {
        const i = yy * W + xx;
        if (mask[i] === 1) { paint(i); continue; }
        if ((xx > 0 && mask[i - 1] === 1) || (xx < W - 1 && mask[i + 1] === 1)
            || (yy > 0 && mask[i - W] === 1) || (yy < H - 1 && mask[i + W] === 1)) paint(i);
      }
    }
    c.putImageData(img, 0, 0);
    return { x0: x0 - 1, y0: y0 - 1, x1: x1 + 2, y1: y1 + 2 };
  }

  /* ───────── paper ───────── */

  function resize() {
    const paper = $("paper");
    const w = paper.clientWidth;
    const h = paper.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const old = state.size;
    if (old.w === w && old.h === h && old.dpr === dpr) return;
    if (old.w && (old.w !== w || old.h !== h)) {
      // Keep the drawing whole and centred when the screen turns or resizes.
      const s = Math.min(w / old.w, h / old.h);
      const ox = (w - old.w * s) / 2;
      const oy = (h - old.h * s) / 2;
      const move = ([x, y]) => [x * s + ox, y * s + oy];
      for (const list of [state.ops, state.cleared || []]) {
        for (const op of list) {
          if (op.tool === "fill") {
            [op.x, op.y] = move([op.x, op.y]);
            const [bx0, by0] = move([op.box.x0, op.box.y0]);
            const [bx1, by1] = move([op.box.x1, op.box.y1]);
            op.box = { x0: bx0, y0: by0, x1: bx1, y1: by1 };
          } else {
            op.width *= s;
            op.points = op.points.map(move);
          }
        }
      }
    }
    state.size = { w, h, dpr };
    canvas.width = cache.width = Math.round(w * dpr);
    canvas.height = cache.height = Math.round(h * dpr);
    rebuild();
  }

  function rebuild() {
    const { dpr } = state.size;
    cctx.setTransform(1, 0, 0, 1, 0, 0);
    cctx.clearRect(0, 0, cache.width, cache.height);
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (const op of state.ops) drawOp(cctx, op);
    show();
  }

  // The finished drawing, then the stroke in progress on top (an eraser stroke erases from it).
  function show() {
    const { dpr } = state.size;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(cache, 0, 0);
    if (state.current) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawOp(ctx, state.current);
    }
  }

  const drawn = () => state.ops.some((op) => op.tool !== "eraser");

  function refreshButtons() {
    const has = drawn();
    $("done").disabled = !has || state.sending;
    $("undo").disabled = !(state.ops.length || state.cleared);
    $("clear").disabled = !state.ops.length;
    $("hint").classList.toggle("gone", has || !!state.current);
  }

  let noteTimer = null;
  function note(text) {
    const box = $("paper-note");
    box.textContent = text;
    box.hidden = false;
    clearTimeout(noteTimer);
    noteTimer = setTimeout(() => { box.hidden = true; }, 2600);
  }

  /* ───────── drawing with pointer events ───────── */

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (state.pointerId !== null || state.sending) return;
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault();
    const [x, y] = point(event);
    if (state.tool === "bucket") {
      const { dpr } = state.size;
      const box = floodFill(cctx, x, y, state.color, true);
      if (!box) {
        if (drawn()) note("Tap inside a shape to fill it.");
        return;
      }
      state.ops.push({ tool: "fill", color: state.color, x, y,
                       box: { x0: box.x0 / dpr, y0: box.y0 / dpr, x1: box.x1 / dpr, y1: box.y1 / dpr } });
      state.cleared = null;
      show();
      refreshButtons();
      return;
    }
    state.pointerId = event.pointerId;
    canvas.setPointerCapture(event.pointerId);
    state.current = { tool: state.tool, color: state.color, width: brushWidth(state.tool), points: [[x, y]] };
    state.cleared = null; // new drawing after a Clear: the old one is gone for good
    show();
    refreshButtons();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (event.pointerId !== state.pointerId || !state.current) return;
    event.preventDefault();
    const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    const pts = state.current.points;
    for (const e of events.length ? events : [event]) {
      const p = point(e);
      const last = pts[pts.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= 1.5) pts.push(p);
    }
    show();
  });

  function finishStroke(event) {
    if (event.pointerId !== state.pointerId) return;
    if (state.current) {
      state.ops.push(state.current);
      drawOp(cctx, state.current);
    }
    state.current = null;
    state.pointerId = null;
    show();
    refreshButtons();
  }
  canvas.addEventListener("pointerup", finishStroke);
  canvas.addEventListener("pointercancel", finishStroke);
  // iOS Safari: no pinch-zoom or page bounce while drawing.
  document.addEventListener("gesturestart", (e) => e.preventDefault());

  /* ───────── tools, sizes, colours, undo, clear ───────── */

  function radio(group, button) {
    for (const b of $(group).querySelectorAll('[role="radio"]')) b.setAttribute("aria-checked", String(b === button));
  }
  for (const button of $("tools").querySelectorAll("button")) {
    button.addEventListener("click", () => { state.tool = button.dataset.tool; radio("tools", button); });
  }
  for (const button of $("sizes").querySelectorAll("button")) {
    button.addEventListener("click", () => { state.sizeFactor = Number(button.dataset.size); radio("sizes", button); });
  }
  for (const button of $("colors").querySelectorAll("button")) {
    button.style.setProperty("--swatch", button.dataset.color);
    button.addEventListener("click", () => {
      state.color = button.dataset.color;
      radio("colors", button);
      // Picking a colour while erasing means "draw with it".
      if (state.tool === "eraser") $("tools").querySelector('[data-tool="marker"]').click();
    });
  }

  function undo() {
    if (state.ops.length) state.ops.pop();
    else if (state.cleared) { state.ops = state.cleared; state.cleared = null; }
    rebuild();
    refreshButtons();
  }
  $("undo").addEventListener("click", undo);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !state.sending
        && !event.target.matches("input, textarea")) { // in the "Who is this?" box, Ctrl+Z undoes typing
      event.preventDefault();
      undo();
    }
    if (event.key === "Escape" && !$("confirm").hidden) $("confirm").hidden = true;
  });

  $("clear").addEventListener("click", () => { $("confirm").hidden = false; $("confirm-no").focus(); });
  $("confirm-no").addEventListener("click", () => { $("confirm").hidden = true; });
  $("confirm-yes").addEventListener("click", () => {
    state.cleared = state.ops;
    state.ops = [];
    $("confirm").hidden = true;
    rebuild();
    refreshButtons();
  });

  /* ───────── Done: hero PNG + description + first page ───────── */

  // The box an op covers; erasing only takes away, so it never widens the hero.
  function opBox(op) {
    if (op.tool === "fill") return op.box;
    const r = op.width / 2;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of op.points) {
      x0 = Math.min(x0, x - r); y0 = Math.min(y0, y - r);
      x1 = Math.max(x1, x + r); y1 = Math.max(y1, y + r);
    }
    return { x0, y0, x1, y1 };
  }

  function box(ops) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const op of ops) {
      const b = opBox(op);
      x0 = Math.min(x0, b.x0); y0 = Math.min(y0, b.y0); x1 = Math.max(x1, b.x1); y1 = Math.max(y1, b.y1);
    }
    return { x0, y0, x1, y1 };
  }

  // The drawing's edges, ignoring stray dots and tiny marks far away from it: one accidental tap
  // in a corner would otherwise shrink the hero (and what the vision model sees) to a speck.
  function bounds() {
    const marks = state.ops.filter((op) => op.tool !== "eraser");
    const small = Math.min(state.size.w, state.size.h) * 0.05;
    const tiny = (op) => { const b = opBox(op); return Math.max(b.x1 - b.x0, b.y1 - b.y0) < small; };
    const main = marks.filter((op) => !tiny(op));
    let b = box(main.length ? main : marks);
    if (main.length) {
      const reach = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 0.25; // eyes, buttons and spots nearby still count
      const near = marks.filter((op) => {
        if (!tiny(op)) return true;
        const t = opBox(op);
        return t.x1 > b.x0 - reach && t.x0 < b.x1 + reach && t.y1 > b.y0 - reach && t.y0 < b.y1 + reach;
      });
      b = box(near);
    }
    const pad = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 0.04 + 2;
    return { x: b.x0 - pad, y: b.y0 - pad, w: b.x1 - b.x0 + 2 * pad, h: b.y1 - b.y0 + 2 * pad };
  }

  // The drawing cropped to its edges, at most `max` pixels on its longer side, drawn again from its
  // ops (fills are repeated at this size). Transparent, or laid on `background` (for the vision
  // model), so erased spots come out as that background and never as black.
  function render(max, background) {
    const b = bounds();
    const scale = Math.min(max / Math.max(b.w, b.h), 3);
    const out = document.createElement("canvas");
    out.width = Math.max(8, Math.round(b.w * scale));
    out.height = Math.max(8, Math.round(b.h * scale));
    const c = out.getContext("2d");
    c.setTransform(scale, 0, 0, scale, -b.x * scale, -b.y * scale);
    for (const op of state.ops) drawOp(c, op);
    if (!background) return out;
    const flat = document.createElement("canvas");
    flat.width = out.width;
    flat.height = out.height;
    const f = flat.getContext("2d");
    f.fillStyle = background;
    f.fillRect(0, 0, flat.width, flat.height);
    f.drawImage(out, 0, 0);
    return flat;
  }

  function showBusy(text, error) {
    $("busy").hidden = false;
    $("busy-text").textContent = text;
    $("busy-error").hidden = !error;
    $("busy-error").textContent = error || "";
    $("busy-retry").hidden = $("busy-close").hidden = !error;
    document.querySelector(".spinner").hidden = !!error;
  }

  async function done() {
    if (!drawn() || state.sending) return;
    state.sending = true;
    refreshButtons();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), GIVE_UP_MS);
    let slow = null;
    try {
      showBusy("Getting to know your hero…");
      slow = setTimeout(() => { $("busy-text").textContent = "Writing the first page…"; }, 5000);
      const payload = {
        hero_png: render(768).toDataURL("image/png"),
        preview_jpeg: render(512, "#ffffff").toDataURL("image/jpeg", 0.85),
        idea: new URLSearchParams(location.search).get("idea") || "",
        who: $("who").value.trim(),
        reading_level: readingLevel(),
      };
      const response = await fetch("/api/stories/drawn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || "Something went wrong. Try again.");
      try { localStorage.setItem(STORAGE_STORY, data.id); } catch (_) { /* private browsing */ }
      $("busy-text").textContent = "Here we go!";
      location.href = `/?story=${encodeURIComponent(data.id)}`;
    } catch (error) {
      state.sending = false;
      refreshButtons();
      const message = error.name === "AbortError"
        ? "That took a little too long. Your drawing is safe."
        : !navigator.onLine ? "You seem to be offline. Check the connection and try again."
        // A dropped connection is a TypeError ("Failed to fetch" in Chrome, "NetworkError…" in Firefox).
        : error instanceof TypeError ? "StoryForge is not answering. Check that it is running, then try again."
        : error.message;
      showBusy("Hmm, that didn't work.", message);
      $("busy-retry").focus();
    } finally {
      clearTimeout(slow);
      clearTimeout(timer);
    }
  }

  /* ───────── reading level ───────── */

  function readingLevel() {
    const checked = document.querySelector('input[name="reading-level"]:checked');
    return checked ? checked.value : "early";
  }
  try {
    const saved = localStorage.getItem(STORAGE_LEVEL);
    const input = saved && document.querySelector(`input[name="reading-level"][value="${saved}"]`);
    if (input) input.checked = true;
  } catch (_) { /* private browsing */ }
  $("levels").addEventListener("change", () => {
    try { localStorage.setItem(STORAGE_LEVEL, readingLevel()); } catch (_) { /* private browsing */ }
  });

  /* ───────── "Who is this?" ───────── */

  $("who-form").addEventListener("submit", (event) => { event.preventDefault(); $("who").blur(); });
  $("done").addEventListener("click", done);
  $("busy-retry").addEventListener("click", done);
  $("busy-close").addEventListener("click", () => { $("busy").hidden = true; });
  // Leaving by accident would lose the drawing.
  window.addEventListener("beforeunload", (event) => {
    if (state.ops.length && !state.sending) event.preventDefault();
  });

  window.addEventListener("resize", resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe($("paper"));
  resize();
  refreshButtons();
})();
