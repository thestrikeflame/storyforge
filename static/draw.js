/* StoryForge: draw your hero.
   A full-screen canvas driven by pointer events (finger, pen or mouse). Strokes are kept as
   point lists, so undo and resizing simply redraw them. Done crops the drawing to its edges,
   keeps the transparency (that PNG becomes the hero layered over every background) and sends a
   copy on white for the vision model, which describes the hero for the storyteller. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  const STORAGE_STORY = "storyforge.story"; // the same key the storybook uses to reopen a story

  const state = {
    strokes: [],        // [{ color, width, points: [[x, y], ...] }] in CSS pixels
    cleared: null,      // the strokes removed by Clear, so Undo can bring them back
    current: null,      // the stroke being drawn
    pointerId: null,    // only one finger draws at a time
    color: "#2c2340",
    size: { w: 0, h: 0, dpr: 1 },
    sending: false,
  };

  /* ───────── paper ───────── */

  function brushWidth() {
    return Math.max(10, Math.min(state.size.w, state.size.h) * 0.022);
  }

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const old = state.size;
    if (old.w && (old.w !== w || old.h !== h)) {
      // Keep the drawing whole and centred when the screen turns or resizes.
      const s = Math.min(w / old.w, h / old.h);
      const ox = (w - old.w * s) / 2;
      const oy = (h - old.h * s) / 2;
      for (const list of [state.strokes, state.cleared || []]) {
        for (const stroke of list) {
          stroke.width *= s;
          stroke.points = stroke.points.map(([x, y]) => [x * s + ox, y * s + oy]);
        }
      }
    }
    state.size = { w, h, dpr };
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    redraw();
  }

  function drawStroke(c, stroke) {
    const pts = stroke.points;
    c.strokeStyle = c.fillStyle = stroke.color;
    c.lineWidth = stroke.width;
    c.lineCap = c.lineJoin = "round";
    if (pts.length === 1) {
      c.beginPath();
      c.arc(pts[0][0], pts[0][1], stroke.width / 2, 0, Math.PI * 2);
      c.fill();
      return;
    }
    // Smooth curves through the midpoints between samples.
    c.beginPath();
    c.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length - 1; i++) {
      const mx = (pts[i][0] + pts[i + 1][0]) / 2;
      const my = (pts[i][1] + pts[i + 1][1]) / 2;
      c.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
    }
    const last = pts[pts.length - 1];
    c.lineTo(last[0], last[1]);
    c.stroke();
  }

  function redraw() {
    const { dpr } = state.size;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (const stroke of state.strokes) drawStroke(ctx, stroke);
    if (state.current) drawStroke(ctx, state.current);
  }

  function refreshButtons() {
    const has = state.strokes.length > 0;
    $("done").disabled = !has || state.sending;
    $("undo").disabled = !(has || state.cleared);
    $("clear").disabled = !has;
    $("hint").classList.toggle("gone", has || !!state.current);
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
    state.pointerId = event.pointerId;
    canvas.setPointerCapture(event.pointerId);
    state.current = { color: state.color, width: brushWidth(), points: [point(event)] };
    state.cleared = null; // new drawing after a Clear: the old one is gone for good
    redraw();
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
    redraw();
  });

  function finishStroke(event) {
    if (event.pointerId !== state.pointerId) return;
    if (state.current) state.strokes.push(state.current);
    state.current = null;
    state.pointerId = null;
    redraw();
    refreshButtons();
  }
  canvas.addEventListener("pointerup", finishStroke);
  canvas.addEventListener("pointercancel", finishStroke);
  // iOS Safari: no pinch-zoom or page bounce while drawing.
  document.addEventListener("gesturestart", (e) => e.preventDefault());

  /* ───────── colours, undo, clear ───────── */

  for (const button of $("colors").querySelectorAll("button")) {
    button.style.setProperty("--swatch", button.dataset.color);
    button.addEventListener("click", () => {
      state.color = button.dataset.color;
      for (const b of $("colors").querySelectorAll("button")) b.setAttribute("aria-checked", String(b === button));
    });
  }

  function undo() {
    if (state.strokes.length) state.strokes.pop();
    else if (state.cleared) { state.strokes = state.cleared; state.cleared = null; }
    redraw();
    refreshButtons();
  }
  $("undo").addEventListener("click", undo);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !state.sending) {
      event.preventDefault();
      undo();
    }
    if (event.key === "Escape" && !$("confirm").hidden) $("confirm").hidden = true;
  });

  $("clear").addEventListener("click", () => { $("confirm").hidden = false; $("confirm-no").focus(); });
  $("confirm-no").addEventListener("click", () => { $("confirm").hidden = true; });
  $("confirm-yes").addEventListener("click", () => {
    state.cleared = state.strokes;
    state.strokes = [];
    $("confirm").hidden = true;
    redraw();
    refreshButtons();
  });

  /* ───────── Done: hero PNG + description + first page ───────── */

  function bounds() {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const s of state.strokes) {
      const r = s.width / 2;
      for (const [x, y] of s.points) {
        x0 = Math.min(x0, x - r); y0 = Math.min(y0, y - r);
        x1 = Math.max(x1, x + r); y1 = Math.max(y1, y + r);
      }
    }
    const pad = Math.max(x1 - x0, y1 - y0) * 0.04 + 2;
    return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
  }

  // The drawing cropped to its edges, at most `max` pixels on its longer side.
  function render(max, background) {
    const b = bounds();
    const scale = Math.min(max / Math.max(b.w, b.h), 3);
    const out = document.createElement("canvas");
    out.width = Math.max(8, Math.round(b.w * scale));
    out.height = Math.max(8, Math.round(b.h * scale));
    const c = out.getContext("2d");
    if (background) { c.fillStyle = background; c.fillRect(0, 0, out.width, out.height); }
    c.setTransform(scale, 0, 0, scale, -b.x * scale, -b.y * scale);
    for (const stroke of state.strokes) drawStroke(c, stroke);
    return out;
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
    if (!state.strokes.length || state.sending) return;
    state.sending = true;
    refreshButtons();
    showBusy("Getting to know your hero…");
    const slow = setTimeout(() => { $("busy-text").textContent = "Writing the first page…"; }, 5000);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);
    try {
      const payload = {
        hero_png: render(768).toDataURL("image/png"),
        preview_jpeg: render(512, "#ffffff").toDataURL("image/jpeg", 0.85),
        idea: new URLSearchParams(location.search).get("idea") || "",
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
        ? "The storyteller took too long to answer. Try again."
        : !navigator.onLine ? "You seem to be offline. Check the connection and try again."
        : error.message === "Failed to fetch" ? "StoryForge is not answering. Check that it is running, then try again."
        : error.message;
      showBusy("Your hero is still here.", message);
      $("busy-retry").focus();
    } finally {
      clearTimeout(slow);
      clearTimeout(timer);
    }
  }

  $("done").addEventListener("click", done);
  $("busy-retry").addEventListener("click", done);
  $("busy-close").addEventListener("click", () => { $("busy").hidden = true; });
  // Leaving by accident would lose the drawing.
  window.addEventListener("beforeunload", (event) => {
    if (state.strokes.length && !state.sending) event.preventDefault();
  });

  window.addEventListener("resize", resize);
  resize();
  refreshButtons();
})();
