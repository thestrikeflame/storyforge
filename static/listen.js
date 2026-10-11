/* StoryForge: talking instead of typing, for the story idea, "Describe your own idea" and /draw's
   "Who is this?". Tap the microphone and speak; tap again, or it stops by itself after 8 seconds.
   The browser records with MediaRecorder and the Jetson writes down what was said (POST /api/listen:
   whisper.cpp there, or Hugging Face's Whisper). The words go into the text box, to check and change
   before anything is sent. Recording needs a secure page (https, or localhost): on plain http the
   microphone stays hidden and the page says why. */
(() => {
  "use strict";

  const MAX_MS = 8000;

  function supported() {
    return Boolean(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia
                   && window.MediaRecorder);
  }

  // What this browser records best: Chrome and Firefox opus in webm, Safari AAC in mp4.
  function recordingType() {
    if (!MediaRecorder.isTypeSupported) return "";
    return ["audio/webm;codecs=opus", "audio/mp4", "audio/webm", "audio/ogg;codecs=opus"]
      .find((type) => MediaRecorder.isTypeSupported(type)) || "";
  }

  /**
   * Wire one microphone button to one text field.
   *   say(text, problem)  shows a short status line next to it
   *   maxChars()          the field's limit
   *   insteadText         the line shown when nothing can turn speech into text
   *   heard(words)        called after the words are in the field
   *   before()            called just before recording (e.g. pause the narration)
   *   onState(mode)       "idle" | "listening" | "working"
   * Returns { stop }: stops listening and throws the recording away.
   */
  function attach({ button, field, say, maxChars, insteadText = "Type your idea instead.", heard, before, onState }) {
    let recorder = null;
    let stream = null;
    let timer = null;
    let chunks = [];
    let run = 0;
    let busy = false;

    function set(mode) {
      button.classList.toggle("listening", mode === "listening");
      button.classList.toggle("working", mode === "working");
      button.setAttribute("aria-pressed", String(mode === "listening"));
      if (onState) onState(mode);
    }

    function release() {
      clearTimeout(timer);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }

    async function start() {
      if (busy) return;
      if (before) before();
      const my = ++run;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch (error) {
        say(error && error.name === "NotAllowedError"
          ? "To talk, allow the microphone. Or type it instead."
          : "The microphone isn't working here. Type it instead.", true);
        return;
      }
      if (my !== run) { release(); return; }
      const type = recordingType();
      chunks = [];
      try {
        recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      } catch (_) {
        release();
        say(insteadText, true);
        return;
      }
      recorder.ondataavailable = (event) => { if (event.data && event.data.size) chunks.push(event.data); };
      recorder.onstop = () => send(my, recorder.mimeType || type || "audio/webm");
      recorder.start();
      set("listening");
      say("Listening… tap the microphone when you're done.");
      timer = setTimeout(finish, MAX_MS);
    }

    function finish() {
      clearTimeout(timer);
      if (recorder && recorder.state !== "inactive") recorder.stop();
      release();
    }

    async function send(my, type) {
      const blob = new Blob(chunks, { type });
      chunks = [];
      if (my !== run) return;
      if (blob.size < 1000) {
        set("idle");
        say("I didn't hear anything. Tap the microphone and try again.", true);
        return;
      }
      busy = true;
      set("working");
      say("Writing down what you said…");
      const controller = new AbortController();
      const giveUp = setTimeout(() => controller.abort(), 30000);
      try {
        const response = await fetch("/api/listen", {
          method: "POST", headers: { "Content-Type": type.split(";")[0] }, body: blob, signal: controller.signal,
        });
        const data = await response.json().catch(() => ({}));
        if (my !== run) return;
        if (!response.ok) throw new Error(data.message || insteadText);
        const words = (data.transcript || "").trim();
        if (!words) {
          say("I didn't catch that. Tap the microphone and try again, or type it.", true);
          return;
        }
        const limit = maxChars ? maxChars() : 600;
        const before = field.value.trim();
        field.value = (before ? `${before} ${words}` : words).slice(0, limit);
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.focus({ preventScroll: true });
        field.setSelectionRange(field.value.length, field.value.length);
        if (heard) heard(words);
      } catch (_) {
        if (my === run) say(insteadText, true);
      } finally {
        clearTimeout(giveUp);
        if (my === run) { busy = false; set("idle"); }
      }
    }

    button.addEventListener("click", () => {
      if (recorder && recorder.state === "recording") finish();
      else start();
    });

    return {
      stop() {
        run += 1; // a recording in flight is thrown away
        if (recorder && recorder.state !== "inactive") recorder.stop();
        release();
        recorder = null;
        busy = false;
        set("idle");
      },
    };
  }

  window.StoryListen = { supported, attach };
})();
