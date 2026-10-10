/* StoryForge: brings the child's hero drawing to life on top of each page's background.
   CSS does the moving (see style.css, "Hero animation"); this file only
     - sets the page's action on the picture frame (data-action), restarting it for a new page,
     - re-seeds the SVG "line boil" filter a few times per second, so the drawing wobbles like
       hand-drawn animation,
     - and spawns the matching particles: dust puffs, sparkles, bubbles and zzz.
   With "reduce motion" on, the hero simply stands still: no boil, no particles, no movement. */
(() => {
  "use strict";

  const ACTIONS = ["standing", "walking", "running", "jumping", "waving", "flying", "swimming", "sleeping",
                   "cheering", "dancing", "spinning", "hiding"];
  const BOIL_MS = 160; // about 6 new wobbles per second, like drawings on paper re-traced each frame

  // Particles per action: [kind, every ms, where on the hero (0..1 across, 0..1 down)].
  const PARTICLES = {
    walking: [["dust", 520, [0.5, 1]]],
    running: [["dust", 260, [0.5, 1]]],
    jumping: [["dust", 1600, [0.5, 1], 960]], // a puff each time it lands (60% into its 1.6 s jump)
    swimming: [["bubble", 650, [0.85, 0.25]]],
    sleeping: [["zzz", 1300, [0.75, 0.15]]],
    cheering: [["sparkle", 280, [0.5, 0.15]]],
    dancing: [["sparkle", 900, [0.5, 0.1]]],
  };

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function create(figure, { fx, noise }) {
    let action = null;
    let boilTimer = null;
    let particleTimers = [];

    const moving = () => !reduceMotion.matches && action && figure.dataset.state === "completed" && !document.hidden;

    function boil() {
      clearInterval(boilTimer);
      boilTimer = null;
      if (!noise) return;
      noise.setAttribute("seed", "1");
      if (!moving()) return;
      boilTimer = setInterval(() => noise.setAttribute("seed", String(1 + Math.floor(Math.random() * 9999))), BOIL_MS);
    }

    function spawn(kind, at) {
      const hero = figure.querySelector(".hero");
      if (!hero || !fx || !moving()) return;
      const frame = figure.getBoundingClientRect();
      const box = hero.getBoundingClientRect();
      if (!box.width || !frame.width) return;
      const el = document.createElement("span");
      el.className = `fx fx-${kind}`;
      const jitter = (n) => (Math.random() - 0.5) * n;
      let x = box.left - frame.left + box.width * at[0];
      let y = box.top - frame.top + box.height * at[1];
      if (kind === "sparkle") {
        // Around the hero, not on it, where they show up against the background.
        const side = Math.random() < 0.5 ? -1 : 1;
        x += side * box.width * (0.45 + Math.random() * 0.35);
        y += jitter(box.height * 0.5);
      } else {
        x += jitter(box.width * 0.25);
        y += jitter(6);
      }
      el.style.left = `${(x / frame.width) * 100}%`;
      el.style.top = `${(y / frame.height) * 100}%`;
      el.style.setProperty("--drift", `${jitter(40).toFixed(0)}px`);
      // Which way the hero is facing, so dust trails behind it.
      el.style.setProperty("--behind", getComputedStyle(figure.querySelector(".hero-flip")).transform.startsWith("matrix(-") ? "1" : "-1");
      if (kind === "zzz") el.textContent = "z";
      el.addEventListener("animationend", () => el.remove());
      fx.appendChild(el);
      if (fx.childElementCount > 40) fx.firstElementChild.remove();
    }

    function particles() {
      particleTimers.forEach((t) => { clearInterval(t); clearTimeout(t); });
      particleTimers = [];
      if (fx) fx.textContent = "";
      if (!moving()) return;
      for (const [kind, every, at, offset] of PARTICLES[action] || []) {
        const start = () => particleTimers.push(setInterval(() => spawn(kind, at), every));
        if (offset) particleTimers.push(setTimeout(() => { spawn(kind, at); start(); }, offset));
        else start();
      }
    }

    function refresh() {
      boil();
      particles();
    }

    return {
      // Show this page's action. The same action on a new page starts again from the beginning.
      set(next) {
        action = ACTIONS.includes(next) ? next : "standing";
        figure.removeAttribute("data-action");
        void figure.offsetWidth; // restart the CSS animations
        figure.dataset.action = action;
        refresh();
      },
      // Call when the picture frame changes state (painted, waiting, failed...).
      refresh,
      stop() {
        action = null;
        figure.removeAttribute("data-action");
        refresh();
      },
      init() {
        reduceMotion.addEventListener("change", refresh);
        document.addEventListener("visibilitychange", refresh);
      },
    };
  }

  window.HeroAnimator = { ACTIONS, create };
})();
