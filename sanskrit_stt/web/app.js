// Records the microphone, cuts speech into chunks at natural pauses, and sends each chunk
// to /api/transcribe as 16 kHz WAV. Results fill in at their place in the transcript, in order.

const RATE = 16000;
const FRAME = 480;              // 30 ms analysis frame
const FRAME_S = FRAME / RATE;
// The model is far more accurate on whole verses than on half-lines (~20% vs ~32% CER on the
// samples), so short pauses only end a chunk once it is long enough; a long pause always does.
const MIN_CHUNK_S = 10;         // a short pause ends the chunk only after this much audio
const MAX_CHUNK_S = 20;         // model was trained on 5-25 s clips
const PAUSE_S = 0.6;            // short pause (between half-verses)
const LONG_PAUSE_S = 1.5;       // speaker stopped: send whatever we have
const PAD_S = 0.3;              // silence kept before/after speech
const MIN_SPEECH_S = 0.4;       // ignore coughs and clicks

const $ = (id) => document.getElementById(id);
const micBtn = $("mic"), statusEl = $("status"), timerEl = $("timer");
const transcriptEl = $("transcript"), copyBtn = $("copy"), clearBtn = $("clear"), toastEl = $("toast");

// ---------- audio processing ----------

/** Box-filter downsampler from the device rate to 16 kHz (adequate for speech). */
class Resampler {
  constructor(inRate) { this.ratio = inRate / RATE; this.pos = 0; this.acc = 0; this.n = 0; }
  push(input) {
    if (this.ratio === 1) return input;
    const out = [];
    for (let i = 0; i < input.length; i++) {
      this.acc += input[i]; this.n++; this.pos++;
      if (this.pos >= this.ratio) { out.push(this.acc / this.n); this.acc = 0; this.n = 0; this.pos -= this.ratio; }
    }
    return Float32Array.from(out);
  }
}

/** Energy-based phrase detector with an adaptive noise floor. */
class Chunker {
  constructor(onChunk, onLevel) {
    this.onChunk = onChunk; this.onLevel = onLevel;
    this.noise = null; this.carry = new Float32Array(0);
    this.frames = []; // { data, speech }
  }

  push(samples) {
    const buf = new Float32Array(this.carry.length + samples.length);
    buf.set(this.carry); buf.set(samples, this.carry.length);
    let i = 0;
    for (; i + FRAME <= buf.length; i += FRAME) this.addFrame(buf.subarray(i, i + FRAME).slice());
    this.carry = buf.slice(i);
  }

  addFrame(data) {
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const rms = Math.sqrt(sum / data.length);
    if (this.noise === null) this.noise = rms;
    // Fall quickly to quieter levels, rise very slowly so long chanting isn't absorbed into the floor.
    this.noise = rms < this.noise ? 0.85 * this.noise + 0.15 * rms : 0.999 * this.noise + 0.001 * rms;
    this.noise = Math.max(this.noise, 0.0005);
    const speech = rms > Math.max(this.noise * 2.5, 0.006);
    this.onLevel(Math.min(1, rms * 9));
    this.frames.push({ data, speech });

    const speechFrames = this.frames.reduce((n, f) => n + f.speech, 0);
    if (speechFrames === 0) {
      // Only leading silence so far: keep a short pre-roll.
      const keep = Math.ceil(PAD_S / FRAME_S);
      if (this.frames.length > keep) this.frames.splice(0, this.frames.length - keep);
      return;
    }

    let silentRun = 0;
    for (let i = this.frames.length - 1; i >= 0 && !this.frames[i].speech; i--) silentRun++;
    const dur = this.frames.length * FRAME_S;

    const pause = silentRun * FRAME_S;
    if (pause >= LONG_PAUSE_S || (dur >= MIN_CHUNK_S && pause >= PAUSE_S)) {
      this.emit(this.frames.length - silentRun + Math.ceil(PAD_S / FRAME_S));
    } else if (dur >= MAX_CHUNK_S) {
      // Cut at the quietest point in the second half instead of mid-syllable.
      let cut = this.frames.length, best = Infinity;
      for (let i = Math.floor(this.frames.length / 2); i < this.frames.length; i++) {
        const e = energy(this.frames[i].data);
        if (e < best) { best = e; cut = i + 1; }
      }
      this.emit(cut);
    }
  }

  emit(cut) {
    const chunk = this.frames.slice(0, cut);
    this.frames = this.frames.slice(cut);
    if (chunk.reduce((n, f) => n + f.speech, 0) * FRAME_S < MIN_SPEECH_S) return;
    const out = new Float32Array(chunk.length * FRAME);
    chunk.forEach((f, i) => out.set(f.data, i * FRAME));
    this.onChunk(out);
  }

  flush() { if (this.frames.length) this.emit(this.frames.length); this.frames = []; }
}

function energy(frame) {
  let s = 0;
  for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
  return s;
}

function encodeWav(samples) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ---------- transcription queue ----------

let queue = Promise.resolve();
let inFlight = 0;

function enqueue(samples) {
  const el = document.createElement("span");
  el.className = "chunk pending";
  transcriptEl.querySelector(".placeholder")?.remove();
  transcriptEl.append(el);
  inFlight++;
  updateUi();
  queue = queue.then(() => send(samples, el));
}

async function send(samples, el) {
  try {
    const form = new FormData();
    form.append("file", encodeWav(samples), "speech.wav");
    const res = await fetch("api/transcribe", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `server returned ${res.status}`);
    if (body.text) {
      el.textContent = body.text;
      el.className = "chunk fresh";
    } else {
      el.remove();
    }
  } catch (err) {
    el.textContent = "(this passage could not be transcribed)";
    el.className = "chunk failed";
    toast(`Transcription failed: ${err.message}`);
  } finally {
    inFlight--;
    updateUi();
  }
}

// ---------- recording ----------

let session = null;

async function start() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("Microphone needs a secure page: open via http://localhost or HTTPS.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch {
    toast("Microphone access was denied.");
    return;
  }
  const ctx = new AudioContext();
  await ctx.resume(); // the permission prompt can consume the click's user activation
  await ctx.audioWorklet.addModule("capture-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "capture-processor");
  const mute = ctx.createGain();
  mute.gain.value = 0; // keeps the graph pulling audio without playing it back

  const resampler = new Resampler(ctx.sampleRate);
  const chunker = new Chunker(enqueue, setLevel);
  node.port.onmessage = (e) => chunker.push(resampler.push(e.data));
  source.connect(node).connect(mute).connect(ctx.destination);

  const startedAt = Date.now();
  const timer = setInterval(() => { timerEl.textContent = clock((Date.now() - startedAt) / 1000); }, 250);
  session = { stream, ctx, node, chunker, timer };
  timerEl.textContent = "00:00";
  updateUi();
}

function stop() {
  if (!session) return;
  const { stream, ctx, node, chunker, timer } = session;
  session = null;
  node.port.onmessage = null;
  chunker.flush();
  clearInterval(timer);
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  setLevel(0);
  updateUi();
}

// ---------- UI ----------

let smoothed = 0;
function setLevel(v) {
  smoothed = v > smoothed ? v : smoothed * 0.85 + v * 0.15;
  micBtn.style.setProperty("--level", smoothed.toFixed(3));
}

function clock(s) {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function finalText() {
  return [...transcriptEl.querySelectorAll(".chunk:not(.pending):not(.failed)")].map((c) => c.textContent).join(" ");
}

function updateUi() {
  const live = session !== null;
  micBtn.classList.toggle("live", live);
  micBtn.setAttribute("aria-pressed", String(live));
  micBtn.setAttribute("aria-label", live ? "Stop listening" : "Start listening");
  timerEl.classList.toggle("on", live);
  if (live) statusEl.textContent = inFlight ? "Listening · writing…" : "Listening…";
  else if (inFlight) statusEl.textContent = "Writing the last words…";
  else statusEl.textContent = finalText() ? "Tap to continue" : "Tap to begin";
  const hasText = transcriptEl.querySelector(".chunk") !== null;
  copyBtn.disabled = !finalText();
  clearBtn.disabled = !hasText || inFlight > 0;
}

let toastTimer;
function toast(msg) {
  toastEl.textContent = msg;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.hidden = true; }, 4000);
}

micBtn.addEventListener("click", () => (session ? stop() : start()));

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(finalText());
    toast("Copied to clipboard.");
  } catch {
    toast("Could not copy. Select the text instead.");
  }
});

clearBtn.addEventListener("click", () => {
  transcriptEl.innerHTML = '<p class="placeholder">Your words will appear here, verse by verse.</p>';
  updateUi();
});

$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("vak-theme", next); } catch {}
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) { e.preventDefault(); micBtn.click(); }
});

updateUi();
