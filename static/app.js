/* StoryForge front end. Plain JavaScript, no framework and no build step.
 *
 * Sections:  helpers → cover → microphone → narration → the book → what happens next
 *            → talking to the server → start-up
 *
 * Text that comes from the model or the reader is always inserted with textContent,
 * never as HTML.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const body = document.body;
  const STORAGE_STORY = "storyforge.story";
  const STORAGE_MUTED = "storyforge.muted";

  const state = {
    config: { tts: "browser", server_stt: false, images: true, https_port: null, max_premise_chars: 600, max_action_chars: 300 },
    story: null,          // { id, title, ended, pages: [...] }
    view: 0,              // index of the page being shown
    busy: false,          // a request that changes the story is in flight
    chooseNow: new Set(), // pages where the reader chose not to wait for the picture
    pollTimer: null,
    pollRun: 0,
    pollFailures: 0,
    muted: false,
  };

  /* ───────── helpers ───────── */

  function store(key, value) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch (_) { /* private browsing: carry on without remembering */ }
  }
  function recall(key) {
    try { return localStorage.getItem(key); } catch (_) { return null; }
  }

  async function api(method, path, payload, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeout || 20000);
    let response;
    try {
      response = await fetch(path, {
        method,
        headers: options.raw ? { "Content-Type": "audio/wav" } : payload ? { "Content-Type": "application/json" } : undefined,
        body: options.raw ? payload : payload ? JSON.stringify(payload) : undefined,
        signal: controller.signal,
        cache: "no-store",
      });
    } catch (_) {
      const error = new Error("StoryForge can't be reached. Check that this device is on the same network, then try again.");
      error.network = true;
      throw error;
    } finally {
      clearTimeout(timer);
    }
    let data = null;
    try { data = await response.json(); } catch (_) { /* not JSON */ }
    if (!response.ok) {
      const error = new Error((data && data.message) || "Something went wrong. Try again.");
      error.status = response.status;
      throw error;
    }
    return data;
  }

  let toastTimer = null;
  function toast(message, sticky = false) {
    const el = $("toast");
    clearTimeout(toastTimer);
    if (!message) { el.hidden = true; return; }
    el.textContent = message;
    el.hidden = false;
    if (!sticky) toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
  }

  function setView(name) {
    body.dataset.view = name;
    $("invite").hidden = name !== "invite";
    $("book-row").hidden = name !== "reading";
    $("next").hidden = name !== "reading";
    $("leave").hidden = name !== "reading";
    $("narration-toggle").hidden = name !== "reading" || narrator.mode === "none";
    if (name !== "reading") narrator.stop();
    window.scrollTo(0, 0);
  }

  function friendlyDuration(seconds) {
    if (!seconds) return "";
    if (seconds < 50) return `about ${Math.max(10, Math.round(seconds / 10) * 10)} seconds`;
    const minutes = Math.round(seconds / 60);
    return minutes <= 1 ? "about a minute" : `about ${minutes} minutes`;
  }

  /* ───────── cover ───────── */

  function scatterMotes() {
    const holder = $("motes");
    for (let i = 0; i < 18; i++) {
      const mote = document.createElement("i");
      mote.style.setProperty("--x", `${28 + Math.random() * 44}%`);
      mote.style.setProperty("--d", `${3 + Math.random() * 5}px`);
      mote.style.setProperty("--t", `${7 + Math.random() * 8}s`);
      mote.style.setProperty("--wait", `${1.8 + Math.random() * 9}s`);
      mote.style.setProperty("--drift", `${(Math.random() - 0.5) * 22}vw`);
      holder.appendChild(mote);
    }
  }

  /* ───────── microphone ─────────
   * Two ways to turn speech into text:
   *   "browser"  the browser's own speech recognition (Chrome, Edge, Safari)
   *   "server"   record here, let whisper.cpp on the Jetson transcribe (works offline)
   * Both need a secure page (https or localhost); that is a browser rule. */

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let micBrowserBroken = false; // set when the browser's recognition reports it cannot work

  function micMode() {
    if (!window.isSecureContext) return "insecure";
    if (SpeechRecognition && !micBrowserBroken) return "browser";
    const canRecord = navigator.mediaDevices && navigator.mediaDevices.getUserMedia && (window.AudioContext || window.webkitAudioContext);
    if (state.config.server_stt && canRecord) return "server";
    return "unsupported";
  }

  function micProblem() {
    const mode = micMode();
    if (mode === "insecure") {
      if (state.config.https_port) {
        const url = `https://${location.hostname}:${state.config.https_port}/`;
        return { text: "The microphone only works on the secure address. Open ", link: url, after: " or type instead." };
      }
      return { text: "The microphone needs a secure (https) address, which is not set up yet. Type your idea instead." };
    }
    if (mode === "unsupported") return { text: "This browser can't turn speech into text. Type your idea instead." };
    return null;
  }

  function showMicStatus(el, message, problem = false) {
    el.classList.toggle("problem", problem);
    el.textContent = "";
    if (!message) return;
    if (typeof message === "string") { el.textContent = message; return; }
    el.append(message.text);
    if (message.link) {
      const a = document.createElement("a");
      a.href = message.link;
      a.textContent = message.link.replace(/\/$/, "");
      el.append(a, message.after || "");
    }
  }

  function encodeWav(chunks, sampleRate) {
    const length = chunks.reduce((n, c) => n + c.length, 0);
    const input = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) { input.set(chunk, offset); offset += chunk.length; }
    // Downsample to 16 kHz, which is what whisper expects, by averaging neighbouring samples.
    const target = 16000;
    const ratio = sampleRate / target;
    const outLength = Math.floor(input.length / ratio);
    const view = new DataView(new ArrayBuffer(44 + outLength * 2));
    const text = (pos, s) => { for (let i = 0; i < s.length; i++) view.setUint8(pos + i, s.charCodeAt(i)); };
    text(0, "RIFF"); view.setUint32(4, 36 + outLength * 2, true); text(8, "WAVE"); text(12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, target, true); view.setUint32(28, target * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, outLength * 2, true);
    for (let i = 0; i < outLength; i++) {
      const from = Math.floor(i * ratio);
      const to = Math.min(input.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = from; j < to; j++) sum += input[j];
      const sample = Math.max(-1, Math.min(1, to > from ? sum / (to - from) : 0));
      view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    return view.buffer;
  }

  /** Wire one microphone button to one text field. */
  function attachMic({ button, bars, status, field, maxChars, readyHint }) {
    let listening = false;
    let latched = false;      // a quick tap keeps listening until the next tap
    let pressedAt = 0;
    let session = 0;          // guards against late events from an earlier recording
    let recognition = null;
    let recorder = null;
    let baseText = "";
    let heard = "";

    const joined = (a, b) => (a && b ? `${a.replace(/\s+$/, "")} ${b}` : a || b).slice(0, maxChars());

    function setListening(on) {
      listening = on;
      button.classList.toggle("listening", on);
      if (bars) bars.classList.toggle("on", on);
      if (!on) { latched = false; if (bars) bars.classList.remove("live"); }
    }

    function finished() {
      setListening(false);
      button.classList.remove("working");
      if (heard.trim()) showMicStatus(status, readyHint);
      else if (!status.classList.contains("problem")) showMicStatus(status, "I didn't hear anything. Hold the microphone while you talk.", true);
    }

    function startBrowser(mySession) {
      recognition = new SpeechRecognition();
      recognition.lang = (navigator.language || "").toLowerCase().startsWith("en") ? navigator.language : "en-US";
      recognition.continuous = true;
      recognition.interimResults = true;
      let finalText = "";
      recognition.onresult = (event) => {
        if (mySession !== session) return;
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const piece = event.results[i][0].transcript;
          if (event.results[i].isFinal) finalText = joined(finalText, piece.trim());
          else interim += piece;
        }
        heard = joined(finalText, interim.trim());
        field.value = joined(baseText, heard);
      };
      recognition.onerror = (event) => {
        if (mySession !== session) return;
        const kind = event.error;
        if (kind === "aborted" || kind === "no-speech") return;
        if (kind === "not-allowed" || kind === "service-not-allowed") {
          showMicStatus(status, "The microphone is blocked for this page. Allow it in the browser's address bar, or type instead.", true);
        } else if (kind === "audio-capture") {
          showMicStatus(status, "No microphone was found on this device. Type instead.", true);
        } else if (kind === "network") {
          micBrowserBroken = true; // this browser needs the internet for recognition and has none
          showMicStatus(status, micMode() === "server"
            ? "Speech recognition lost its connection. Hold the microphone again to use the built-in listener."
            : "This browser needs the internet to understand speech. Type instead.", true);
        } else {
          showMicStatus(status, "The microphone stopped unexpectedly. Try again, or type instead.", true);
        }
        setListening(false);
      };
      recognition.onend = () => {
        if (mySession !== session) return;
        if (listening) {
          // Some browsers stop after a pause even though the button is still held: carry on.
          baseText = field.value; heard = heard || ""; finalText = "";
          try { recognition.start(); return; } catch (_) { /* fall through and finish */ }
        }
        if (!status.classList.contains("problem")) finished();
      };
      try {
        recognition.start();
      } catch (_) {
        showMicStatus(status, "The microphone could not start. Try again, or type instead.", true);
        setListening(false);
      }
    }

    async function startServer(mySession) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      } catch (error) {
        if (mySession !== session) return;
        const blocked = error && (error.name === "NotAllowedError" || error.name === "SecurityError");
        showMicStatus(status, blocked
          ? "The microphone is blocked for this page. Allow it in the browser's address bar, or type instead."
          : "No microphone was found on this device. Type instead.", true);
        setListening(false);
        return;
      }
      if (mySession !== session || !listening) { stream.getTracks().forEach((t) => t.stop()); return; }
      const context = new Ctx();
      const source = context.createMediaStreamSource(stream);
      const node = context.createScriptProcessor(4096, 1, 1);
      const silent = context.createGain();
      silent.gain.value = 0;
      const chunks = [];
      const barEls = bars ? [...bars.children] : [];
      if (bars) bars.classList.add("live");
      node.onaudioprocess = (event) => {
        const data = event.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(data));
        let sum = 0;
        for (let i = 0; i < data.length; i += 8) sum += data[i] * data[i];
        const level = Math.min(1, Math.sqrt(sum / (data.length / 8)) * 6);
        barEls.forEach((bar, i) => {
          const shape = 1 - Math.abs(i - (barEls.length - 1) / 2) / barEls.length;
          bar.style.transform = `scaleY(${Math.max(0.15, level * shape * (0.7 + Math.random() * 0.6))})`;
        });
      };
      source.connect(node); node.connect(silent); silent.connect(context.destination);
      const limit = setTimeout(() => { if (mySession === session && listening) stop(); }, 60000);
      recorder = {
        async finish() {
          clearTimeout(limit);
          node.disconnect(); source.disconnect();
          stream.getTracks().forEach((t) => t.stop());
          const rate = context.sampleRate;
          context.close();
          barEls.forEach((bar) => { bar.style.transform = ""; });
          if (!chunks.length) { finished(); return; }
          button.classList.add("working");
          showMicStatus(status, "Turning your words into text");
          try {
            const result = await api("POST", "/api/transcribe", encodeWav(chunks, rate), { raw: true, timeout: 100000 });
            if (mySession !== session) return;
            heard = (result.text || "").trim();
            field.value = joined(baseText, heard);
            finished();
          } catch (error) {
            if (mySession !== session) return;
            button.classList.remove("working");
            showMicStatus(status, error.message, true);
          }
        },
      };
    }

    function start() {
      const problem = micProblem();
      if (problem) { showMicStatus(status, problem, true); return; }
      if (button.classList.contains("working")) return;
      narrator.pause(); // do not record the narrator's own voice
      session += 1;
      baseText = field.value.trim();
      heard = "";
      recorder = null;
      setListening(true);
      showMicStatus(status, "Listening");
      if (micMode() === "browser") startBrowser(session);
      else startServer(session);
    }

    function stop() {
      if (!listening) return;
      setListening(false);
      if (recognition) { try { recognition.stop(); } catch (_) { finished(); } }
      if (recorder) { const r = recorder; recorder = null; r.finish(); }
      else if (micMode() === "server") finished();
    }

    function press() {
      if (listening && latched) { stop(); return; }
      if (listening) return;
      pressedAt = performance.now();
      start();
    }
    function release() {
      if (!listening || latched) return;
      if (performance.now() - pressedAt < 350) {
        latched = true; // it was a tap, not a hold
        showMicStatus(status, "Listening. Tap the microphone again when you have finished.");
      } else {
        stop();
      }
    }

    button.addEventListener("pointerdown", (event) => {
      if (event.button) return;
      event.preventDefault();
      try { button.setPointerCapture(event.pointerId); } catch (_) { /* older browsers */ }
      press();
    });
    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
    button.addEventListener("contextmenu", (event) => event.preventDefault());
    button.addEventListener("keydown", (event) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat) { event.preventDefault(); press(); }
    });
    button.addEventListener("keyup", (event) => {
      if (event.key === " " || event.key === "Enter") { event.preventDefault(); release(); }
    });

    return {
      refresh() {
        const problem = micProblem();
        button.classList.toggle("unavailable", Boolean(problem));
        showMicStatus(status, problem || "", Boolean(problem));
      },
      stop,
    };
  }

  /* ───────── narration ─────────
   * "piper":   the Jetson makes a voice recording for each page and the browser plays it.
   * "browser": the browser's own voice reads the text, sentence by sentence.
   * Falls back from piper to browser automatically, and to silence with a note when neither works. */

  const narrator = {
    mode: "none",
    page: -1,
    playing: false,
    audio: null,
    sentences: [],
    sentenceIndex: 0,
    spokenChars: 0,
    totalChars: 1,
    run: 0,
    watchdog: null,

    init() {
      const canSpeak = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
      if (state.config.tts === "piper") this.mode = "piper";
      else if (state.config.tts === "browser" && canSpeak) this.mode = "browser";
      else this.mode = "none";
      this.canSpeak = canSpeak;
      if (canSpeak) speechSynthesis.getVoices(); // ask early; voices load in the background
    },

    voice() {
      const voices = speechSynthesis.getVoices().filter((v) => v.lang && v.lang.toLowerCase().startsWith("en"));
      return voices.find((v) => v.localService && /natural|samantha|daniel|karen|female/i.test(v.name))
        || voices.find((v) => v.localService) || voices[0] || null;
    },

    start(index, text) {
      this.stop();
      if (this.mode === "none" || state.muted || !text) { renderNarration(); return; }
      this.page = index;
      this.run += 1;
      this.setProgress(0);
      if (this.mode === "piper") this.startAudio(index, text);
      else this.startSpeech(text, 0);
      renderNarration();
    },

    startAudio(index, text) {
      const run = this.run;
      const audio = new Audio(`/api/stories/${state.story.id}/segments/${index}/audio`);
      this.audio = audio;
      audio.addEventListener("timeupdate", () => {
        if (run === this.run && audio.duration) this.setProgress(audio.currentTime / audio.duration);
      });
      audio.addEventListener("ended", () => { if (run === this.run) { this.playing = false; this.setProgress(1); renderNarration(); } });
      audio.addEventListener("error", () => {
        if (run !== this.run) return;
        // The server could not provide a recording: let the browser read instead.
        this.audio = null;
        if (this.canSpeak) { this.mode = "browser"; this.startSpeech(text, 0); }
        else { this.mode = "none"; this.playing = false; }
        renderNarration();
      });
      this.playing = true;
      audio.play().catch(() => { if (run === this.run) { this.playing = false; renderNarration(); } });
    },

    startSpeech(text, fromSentence) {
      this.sentences = text.replace(/\s+/g, " ").match(/[^.!?…]+[.!?…]+["'”’)]*\s*|[^.!?…]+$/g) || [text];
      this.totalChars = Math.max(1, this.sentences.reduce((n, s) => n + s.length, 0));
      this.sentenceIndex = fromSentence;
      this.spokenChars = this.sentences.slice(0, fromSentence).reduce((n, s) => n + s.length, 0);
      this.playing = true;
      try { speechSynthesis.cancel(); } catch (_) { /* nothing was queued */ }
      const run = this.run;
      clearTimeout(this.watchdog);
      let started = false;
      // If no voice starts within a few seconds this browser has no usable voice.
      this.watchdog = setTimeout(() => {
        if (run === this.run && !started && this.playing) { this.unavailable(); }
      }, 6000);
      const speakNext = () => {
        if (run !== this.run) return;
        if (this.sentenceIndex >= this.sentences.length) {
          this.playing = false; this.setProgress(1); renderNarration(); return;
        }
        const sentence = this.sentences[this.sentenceIndex];
        const utterance = new SpeechSynthesisUtterance(sentence);
        const voice = this.voice();
        if (voice) { utterance.voice = voice; utterance.lang = voice.lang; } else utterance.lang = "en-US";
        utterance.rate = 0.94;
        utterance.onstart = () => { started = true; };
        utterance.onboundary = (event) => {
          if (run === this.run) this.setProgress((this.spokenChars + event.charIndex) / this.totalChars);
        };
        utterance.onend = () => {
          if (run !== this.run) return;
          started = true;
          this.spokenChars += sentence.length;
          this.sentenceIndex += 1;
          this.setProgress(this.spokenChars / this.totalChars);
          speakNext();
        };
        utterance.onerror = (event) => {
          if (run !== this.run || event.error === "interrupted" || event.error === "canceled") return;
          if (event.error === "not-allowed") { this.playing = false; renderNarration(); return; } // needs a tap first
          this.unavailable();
        };
        this.current = utterance; // keep a reference; some browsers drop events otherwise
        speechSynthesis.speak(utterance);
      };
      speakNext();
    },

    unavailable() {
      this.run += 1;
      this.playing = false;
      this.mode = "none";
      try { speechSynthesis.cancel(); } catch (_) { /* ignore */ }
      renderNarration();
    },

    pause() {
      if (!this.playing) return;
      this.playing = false;
      if (this.audio) this.audio.pause();
      else if (this.canSpeak) { this.run += 1; clearTimeout(this.watchdog); speechSynthesis.cancel(); }
      renderNarration();
    },

    resume(index, text) {
      if (state.muted || this.mode === "none") return;
      if (this.page !== index || this.finished()) { this.start(index, text); return; }
      if (this.audio) {
        this.playing = true;
        this.audio.play().catch(() => { this.playing = false; renderNarration(); });
      } else {
        this.run += 1;
        this.startSpeech(text, this.sentenceIndex); // carries on from the sentence that was interrupted
      }
      renderNarration();
    },

    finished() {
      if (this.audio) return this.audio.ended;
      return this.sentenceIndex >= this.sentences.length;
    },

    stop() {
      this.run += 1;
      this.playing = false;
      this.page = -1;
      clearTimeout(this.watchdog);
      if (this.audio) { this.audio.pause(); this.audio.removeAttribute("src"); this.audio = null; }
      if (this.canSpeak) { try { speechSynthesis.cancel(); } catch (_) { /* ignore */ } }
      this.sentences = []; this.sentenceIndex = 0;
    },

    setProgress(fraction) {
      const value = Math.max(0, Math.min(1, fraction || 0));
      $("narration-fill").style.transform = `scaleX(${value})`;
      $("narration-track").setAttribute("aria-valuenow", String(Math.round(value * 100)));
    },
  };

  function renderNarration() {
    const page = currentPage();
    const hasText = Boolean(page && page.text_complete && page.text);
    const available = narrator.mode !== "none";
    $("narration").hidden = !hasText || !available || state.muted;
    $("narration-toggle").hidden = body.dataset.view !== "reading" || !available;
    const note = $("narration-note");
    if (hasText && !available) {
      note.textContent = state.config.tts === "off"
        ? "Narration is turned off in the settings."
        : "This browser has no voice for reading aloud, so this story is silent here.";
      note.hidden = false;
    } else {
      note.hidden = true;
    }
    const playingHere = narrator.playing && narrator.page === state.view;
    const playButton = $("narration-play");
    playButton.classList.toggle("playing", playingHere);
    playButton.setAttribute("aria-label", playingHere ? "Pause narration" : "Play narration");
    if (narrator.page !== state.view) narrator.setProgress(0);
    $("narration-toggle").setAttribute("aria-pressed", String(!state.muted));
    $("narration-toggle-label").textContent = state.muted ? "Narration off" : "Narration on";
  }

  /* ───────── the book ───────── */

  const currentPage = () => (state.story ? state.story.pages[state.view] : null);
  const lastIndex = () => (state.story ? state.story.pages.length - 1 : -1);
  const imageSettled = (page) => ["completed", "failed", "skipped", "none"].includes(page.image.status);
  const needsWatching = (page) =>
    page.status === "writing" || (page.status === "ready" && ["queued", "generating"].includes(page.image.status));

  function renderHeader() {
    const story = state.story;
    $("story-title").textContent = story.title || "Your story is beginning";
    const total = story.pages.length;
    $("page-count").textContent = state.view === total - 1 ? `Page ${total}` : `Page ${state.view + 1} of ${total}`;
    document.title = story.title ? `${story.title} · StoryForge` : "StoryForge";
    $("prev-page").disabled = state.view <= 0;
    $("next-page").disabled = state.view >= total - 1;
  }

  let shownText = null;
  function renderWords() {
    const page = currentPage();
    const words = $("words");
    const problem = $("words-problem");
    if (page.status === "failed") {
      words.hidden = true;
      problem.hidden = false;
      $("words-problem-message").textContent = (page.error && page.error.message) || "This page could not be written. Try again.";
      $("words-change").hidden = page.index !== 0;
      shownText = null;
      return;
    }
    words.hidden = false;
    problem.hidden = true;
    words.classList.toggle("opening", page.index === 0);
    words.classList.toggle("writing", page.status === "writing" && !page.text_complete && Boolean(page.text));
    const key = `${state.story.id}:${page.index}:${page.text}`;
    if (key === shownText) return;
    shownText = key;
    if (!page.text) {
      words.replaceChildren(Object.assign(document.createElement("p"), { className: "thinking", textContent: "The storyteller is thinking" }));
      return;
    }
    // Update paragraphs in place so text that is already on the page never flickers.
    const paragraphs = page.text.split(/\n+/).filter(Boolean);
    const existing = [...words.querySelectorAll("p:not(.thinking)")];
    words.querySelectorAll(".thinking").forEach((el) => el.remove());
    paragraphs.forEach((text, i) => {
      if (existing[i]) { if (existing[i].textContent !== text) existing[i].textContent = text; }
      else words.appendChild(Object.assign(document.createElement("p"), { textContent: text }));
    });
    existing.slice(paragraphs.length).forEach((el) => el.remove());
  }

  let shownPicture = null;
  const heroAnimator = window.HeroAnimator
    ? window.HeroAnimator.create($("picture"), { fx: $("hero-fx"), noise: document.getElementById("line-boil-noise") })
    : null;
  if (heroAnimator) heroAnimator.init();
  // Set in JS, not the stylesheet: a url(#id) in an external stylesheet may look for the filter there.
  $("hero-img").style.filter = "url(#line-boil)";

  function renderPicture() {
    const page = currentPage();
    const figure = $("picture");
    const image = page.image;
    let status = image.status;
    if (page.status !== "ready" && status === "none") status = "waiting";
    if (page.status === "ready" && status === "none") status = state.config.images ? "queued" : "skipped";
    if (page.status === "failed") status = "waiting";
    if (page.pictureBroken) status = "failed";

    $("picture-meter-fill").style.transform = `scaleX(${status === "generating" ? image.progress || 0.02 : 0})`;
    $("picture-meter").setAttribute("aria-valuenow", String(Math.round((image.progress || 0) * 100)));

    const key = `${state.story.id}:${page.index}:${page.status}:${status}:${image.url || ""}:${state.story.heroUrl}:${page.hero_action || ""}`;
    if (key === shownPicture) return;
    shownPicture = key;

    const img = $("picture-img");
    const hero = $("hero-img");
    if (status === "completed" && image.url) {
      // Load off-screen first. The page that asked for it may no longer be the one on show.
      const storyId = state.story.id;
      const index = page.index;
      const loader = new Image();
      loader.onload = () => {
        if (!state.story || state.story.id !== storyId || state.view !== index) return;
        img.src = image.url;
        img.alt = image.alt ? `Illustration: your hero in ${image.alt}` : "Illustration for this page";
        // The background is scenery only; the same hero stands in every scene.
        if (hero.getAttribute("src") !== state.story.heroUrl) hero.src = state.story.heroUrl;
        figure.dataset.state = "completed";
        if (heroAnimator) heroAnimator.set(page.hero_action || "standing");
      };
      loader.onerror = () => {
        page.pictureBroken = true;
        if (state.story && state.story.id === storyId && state.view === index) { shownPicture = null; renderPicture(); renderNext(); }
      };
      if (figure.dataset.state === "completed") { figure.dataset.state = "generating"; if (heroAnimator) heroAnimator.stop(); }
      loader.src = image.url;
      return;
    }
    img.removeAttribute("src");
    img.alt = "";
    figure.dataset.state = status;
    if (heroAnimator) heroAnimator.stop();
    if (status === "waiting") {
      $("painting-label").textContent = page.status === "failed" ? "No picture yet" : "The picture comes after the words";
    } else if (status === "queued") {
      $("painting-label").textContent = "Getting the paints ready";
    } else if (status === "generating") {
      $("painting-label").textContent = "Painting this page";
    }
    const eta = friendlyDuration(image.eta_seconds);
    $("painting-eta").textContent = eta ? `Pictures take ${eta} here.` : "";
    if (status === "failed" || status === "skipped") {
      $("unpainted-message").textContent = page.pictureBroken
        ? "The picture could not be shown."
        : image.message || (status === "skipped" ? "Pictures are turned off." : "This page stayed unpainted.");
    }
  }

  /* ───────── what happens next ───────── */

  let shownChoices = null;
  function renderNext() {
    const page = currentPage();
    const wait = $("next-wait");
    const choicesBox = $("next-choices");
    const end = $("the-end");
    const waitButton = $("choose-now");
    const isLatest = state.view === lastIndex();
    wait.hidden = true; choicesBox.hidden = true; end.hidden = true; waitButton.hidden = true;

    if (!isLatest) {
      wait.hidden = false;
      $("next-wait-text").textContent = "You are looking back through the story.";
      waitButton.hidden = false;
      waitButton.textContent = "Go to the latest page";
      waitButton.dataset.mode = "latest";
      return;
    }
    if (page.status === "failed") return;
    if (page.status === "writing") {
      wait.hidden = false;
      $("next-wait-text").textContent = page.text_complete ? "Thinking about what could happen next" : "";
      return;
    }
    if (page.final || state.story.ended) { end.hidden = false; return; }

    const reveal = imageSettled(page) || page.pictureBroken || state.chooseNow.has(page.index) || !state.config.images;
    if (!reveal) {
      wait.hidden = false;
      $("next-wait-text").textContent = "Your choices appear when the picture is painted.";
      waitButton.hidden = false;
      waitButton.textContent = "Choose now";
      waitButton.dataset.mode = "choose";
      return;
    }
    choicesBox.hidden = false;
    const key = `${state.story.id}:${page.index}:${page.choices.join("|")}`;
    if (key !== shownChoices) {
      shownChoices = key;
      const holder = $("choices");
      holder.replaceChildren(...page.choices.map((text) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "choice";
        button.textContent = text;
        button.addEventListener("click", () => { button.classList.add("chosen"); advance("choice", text); });
        return button;
      }));
      $("own-idea").value = "";
      ideaMic.refresh();
      // Bring the new choices into view on small screens, moving as little as possible.
      requestAnimationFrame(() => choicesBox.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    }
    setNextDisabled(state.busy);
  }

  function setNextDisabled(disabled) {
    $("next-choices").querySelectorAll("button, input").forEach((el) => { el.disabled = disabled; });
  }

  function renderAll() {
    if (!state.story) return;
    renderHeader();
    renderWords();
    renderPicture();
    renderNarration();
    renderNext();
  }

  function showPage(index, { turn = false } = {}) {
    if (!state.story) return;
    const target = Math.max(0, Math.min(lastIndex(), index));
    if (target !== state.view || turn) {
      narrator.stop();
      premiseMic.stop(); ideaMic.stop();
    }
    state.view = target;
    $("leave-confirm").hidden = true; $("new-story").hidden = false;
    if (turn) {
      const leaf = $("turn");
      leaf.classList.remove("go");
      void leaf.offsetWidth; // restart the animation
      leaf.classList.add("go");
      leaf.addEventListener("animationend", () => leaf.classList.remove("go"), { once: true });
    }
    renderAll();
  }

  /* ───────── talking to the server ───────── */

  function adoptStory(data) {
    const previous = state.story && state.story.id === data.id ? state.story.pages : [];
    state.story = {
      id: data.id,
      title: data.title,
      premise: data.premise,
      ended: data.ended,
      heroUrl: data.hero_url || "/static/art/hero-placeholder.png",
      pages: data.segments.map((seg, i) => Object.assign({}, previous[i] ? { pictureBroken: previous[i].pictureBroken } : {}, seg)),
    };
    store(STORAGE_STORY, data.id);
  }

  function watch() {
    clearTimeout(state.pollTimer);
    const run = ++state.pollRun;
    const tick = async () => {
      if (run !== state.pollRun || !state.story) return;
      const story = state.story;
      const pending = story.pages.filter(needsWatching);
      if (!pending.length) return;
      try {
        const updates = await Promise.all(pending.map((p) => api("GET", `/api/stories/${story.id}/segments/${p.index}`, null, { timeout: 8000 })));
        if (run !== state.pollRun || state.story !== story) return; // a newer story or page took over
        if (state.pollFailures >= 3) toast("");
        state.pollFailures = 0;
        updates.forEach(applyUpdate);
      } catch (error) {
        if (run !== state.pollRun) return;
        if (error.status === 404) {
          toast("This story is no longer on the device. Start a new journey.", true);
          return;
        }
        state.pollFailures += 1;
        if (state.pollFailures === 3) toast("The connection to StoryForge was lost. Trying again.", true);
      }
      const writing = story.pages.some((p) => p.status === "writing");
      const delay = state.pollFailures ? Math.min(8000, 1000 * state.pollFailures) : writing ? 450 : 1200;
      state.pollTimer = setTimeout(tick, delay);
    };
    state.pollTimer = setTimeout(tick, 250);
  }

  function applyUpdate(update) {
    const story = state.story;
    const old = story.pages[update.index];
    if (!old) return;
    const page = Object.assign({}, update, { pictureBroken: old.pictureBroken && update.image.status === "failed" });
    story.pages[update.index] = page;
    if (update.title) story.title = update.title;
    if (update.ended) story.ended = true;
    if (update.index !== state.view) return;

    renderHeader();
    renderWords();
    renderPicture();
    // Narration begins the moment the page text is complete: before the choices are written
    // and long before the picture is painted.
    if (!old.text_complete && page.text_complete && update.index === lastIndex()) narrator.start(page.index, page.text);
    renderNarration();
    renderNext();
  }

  async function beginStory(premise) {
    if (state.busy) return;
    state.busy = true;
    const submit = $("premise-submit");
    submit.disabled = true;
    submit.textContent = "Opening the book";
    premiseMic.stop();
    try {
      const data = await api("POST", "/api/stories", { premise });
      const old = recall(STORAGE_STORY);
      if (old && old !== data.id) api("POST", `/api/stories/${old}/abandon`).catch(() => {});
      state.chooseNow.clear();
      shownText = shownPicture = shownChoices = null;
      adoptStory(data);
      state.view = 0;
      setView("reading");
      renderAll();
      watch();
      $("words").focus({ preventScroll: true });
    } catch (error) {
      toast(error.message);
    } finally {
      state.busy = false;
      submit.disabled = false;
      submit.textContent = "Begin the story";
    }
  }

  async function advance(kind, action = "") {
    if (state.busy || !state.story) return;
    const page = state.story.pages[lastIndex()];
    if (!page || page.status !== "ready") return;
    state.busy = true;
    setNextDisabled(true);
    narrator.stop();
    ideaMic.stop();
    try {
      const data = await api("POST", `/api/stories/${state.story.id}/continue`, { kind, action, after: page.index });
      adoptStory(data);
      showPage(lastIndex(), { turn: true });
      watch();
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      toast(error.message);
      document.querySelectorAll(".choice.chosen").forEach((el) => el.classList.remove("chosen"));
    } finally {
      state.busy = false;
      setNextDisabled(false);
    }
  }

  async function retryWords() {
    if (state.busy || !state.story) return;
    state.busy = true;
    try {
      const seg = await api("POST", `/api/stories/${state.story.id}/segments/${state.view}/retry`);
      state.story.pages[seg.index] = seg;
      shownText = null;
      renderAll();
      watch();
    } catch (error) {
      toast(error.message);
    } finally {
      state.busy = false;
    }
  }

  async function retryPicture() {
    const page = currentPage();
    if (!page) return;
    const button = $("picture-retry");
    button.disabled = true;
    try {
      if (page.pictureBroken && page.image.status === "completed") {
        page.pictureBroken = false; // the file exists; just try loading it again
        page.image.url += "&again=" + Date.now();
      } else {
        const seg = await api("POST", `/api/stories/${state.story.id}/segments/${page.index}/image/retry`);
        state.story.pages[seg.index] = seg;
      }
      shownPicture = null;
      renderPicture(); renderNext();
      watch();
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  }

  function leaveStory() {
    if (state.story) api("POST", `/api/stories/${state.story.id}/abandon`).catch(() => {});
    clearTimeout(state.pollTimer);
    state.pollRun += 1;
    state.story = null;
    store(STORAGE_STORY, null);
    document.title = "StoryForge";
    $("resume").hidden = true;
    $("premise").value = "";
    openInvite();
  }

  function openInvite() {
    setView("invite");
    premiseMic.refresh();
    const problem = micProblem();
    (problem ? $("premise") : $("premise-mic")).focus({ preventScroll: true });
  }

  async function offerResume() {
    const id = recall(STORAGE_STORY);
    if (!id || !/^[a-f0-9]{16}$/.test(id)) return;
    try {
      const data = await api("GET", `/api/stories/${id}`);
      if (!data.segments.length || data.segments.every((s) => s.status === "failed")) return;
      const button = $("resume");
      button.textContent = data.title ? `Continue “${data.title}”` : "Continue your story";
      button.hidden = false;
      button.onclick = () => openStory(data);
    } catch (_) {
      store(STORAGE_STORY, null);
    }
  }

  function openStory(data) {
    state.chooseNow.clear();
    shownText = shownPicture = shownChoices = null;
    adoptStory(data);
    state.view = lastIndex();
    setView("reading");
    renderAll();
    watch();
  }

  // /draw sends the reader here as /?story=<id> once their hero's story has started.
  async function openFromAddress() {
    const id = new URLSearchParams(location.search).get("story");
    if (!id) return false;
    history.replaceState(null, "", "/"); // a reload later shows the cover with "Continue" instead
    if (!/^[a-f0-9]{16}$/.test(id)) return false;
    try {
      openStory(await api("GET", `/api/stories/${id}`));
      return true;
    } catch (_) {
      toast("That story could not be opened. Start a new one, or try again.", true);
      return false;
    }
  }

  /* ───────── start-up ───────── */

  const premiseMic = attachMic({
    button: $("premise-mic"), bars: $("premise-bars"), status: $("premise-mic-status"), field: $("premise"),
    maxChars: () => state.config.max_premise_chars,
    readyHint: "Got it. Press Begin the story, or hold the microphone to add more.",
  });
  const ideaMic = attachMic({
    button: $("idea-mic"), bars: null, status: $("idea-mic-status"), field: $("own-idea"),
    maxChars: () => state.config.max_action_chars,
    readyHint: "Got it. Press Make it happen.",
  });

  $("start").addEventListener("click", openInvite);
  $("invite-back").addEventListener("click", () => { premiseMic.stop(); setView("landing"); $("start").focus(); });
  $("premise-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const premise = $("premise").value.trim();
    if (premise.length < 3) {
      showMicStatus($("premise-mic-status"), "Describe your story first: speak it, or type a sentence below.", true);
      return;
    }
    beginStory(premise);
  });
  $("premise").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("premise-form").requestSubmit(); }
  });

  $("own-idea-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const idea = $("own-idea").value.trim();
    if (idea.length < 2) {
      showMicStatus($("idea-mic-status"), "Say or type what should happen next.", true);
      $("own-idea").focus();
      return;
    }
    advance("custom", idea);
  });
  $("just-continue").addEventListener("click", () => advance("continue"));
  $("finish-story").addEventListener("click", () => advance("finish"));
  $("choose-now").addEventListener("click", (event) => {
    if (event.currentTarget.dataset.mode === "latest") { showPage(lastIndex()); return; }
    state.chooseNow.add(state.view);
    renderNext();
  });

  $("prev-page").addEventListener("click", () => showPage(state.view - 1));
  $("next-page").addEventListener("click", () => showPage(state.view + 1));
  $("words-retry").addEventListener("click", retryWords);
  $("words-change").addEventListener("click", () => {
    const premise = state.story ? state.story.premise || "" : "";
    leaveStory();
    $("premise").value = premise;
  });
  $("picture-retry").addEventListener("click", retryPicture);

  $("narration-play").addEventListener("click", () => {
    const page = currentPage();
    if (!page) return;
    if (narrator.playing && narrator.page === state.view) narrator.pause();
    else narrator.resume(page.index, page.text);
  });
  $("narration-replay").addEventListener("click", () => {
    const page = currentPage();
    if (page) narrator.start(page.index, page.text);
  });
  $("narration-toggle").addEventListener("click", () => {
    state.muted = !state.muted;
    store(STORAGE_MUTED, state.muted ? "1" : null);
    if (state.muted) narrator.stop();
    else { const page = currentPage(); if (page && page.text_complete) narrator.start(page.index, page.text); }
    renderNarration();
  });

  $("new-story").addEventListener("click", () => {
    $("new-story").hidden = true;
    $("leave-confirm").hidden = false;
    $("leave-no").focus();
  });
  $("leave-no").addEventListener("click", () => { $("leave-confirm").hidden = true; $("new-story").hidden = false; $("new-story").focus(); });
  $("leave-yes").addEventListener("click", leaveStory);
  $("end-new").addEventListener("click", leaveStory);
  $("end-reread").addEventListener("click", () => { showPage(0); window.scrollTo({ top: 0, behavior: "smooth" }); });

  window.addEventListener("pagehide", () => narrator.stop());
  window.addEventListener("offline", () => toast("This device went offline. StoryForge will carry on when the connection returns.", true));
  window.addEventListener("online", () => { toast(""); if (state.story) watch(); });
  document.addEventListener("keydown", (event) => {
    if (body.dataset.view !== "reading" || event.target.matches("input, textarea")) return;
    if (event.key === "ArrowLeft") showPage(state.view - 1);
    if (event.key === "ArrowRight") showPage(state.view + 1);
  });

  (async function boot() {
    state.muted = recall(STORAGE_MUTED) === "1";
    scatterMotes();
    try {
      state.config = Object.assign(state.config, await api("GET", "/api/config"));
    } catch (_) {
      toast("StoryForge is not answering yet. Check that it is running, then reload this page.", true);
    }
    narrator.init();
    if (state.config.landing_art) {
      const photo = $("cover-photo");
      photo.onload = () => { photo.hidden = false; $("cover").classList.add("has-photo"); };
      photo.src = state.config.landing_art;
    }
    if (!(await openFromAddress())) offerResume();
  })();
})();
