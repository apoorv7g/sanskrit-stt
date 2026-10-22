# Vāk — Sanskrit speech to Devanagari

Speak Sanskrit into the browser; the words appear in Devanagari, verse by verse.
Everything runs locally on CPU.

Uses AI4Bharat's **IndicConformer** Sanskrit model
([`ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large`](https://huggingface.co/ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large), ~120M params, MIT)
via its ONNX export ([`OpenVoiceOS/ai4bharat-indicconformer-sa-onnx`](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-sa-onnx))
and [onnx-asr](https://github.com/istupakov/onnx-asr). No NeMo, PyTorch or GPU needed.

## Run it

Requirements: Python 3.11+, `ffmpeg` on `PATH`, ~600 MB disk for the model.

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/sanskrit-stt serve
```

Open <http://127.0.0.1:8000>, tap the microphone, and speak. The first start downloads the model.

Browsers only allow the microphone on `localhost` or HTTPS, so to use it from another device put
the server behind HTTPS (e.g. a reverse proxy) and start it with `--host 0.0.0.0`.

## How it works

```
browser mic ─▶ AudioWorklet ─▶ 16 kHz ─▶ phrase detector ─▶ WAV chunk ─▶ POST /api/transcribe
                                                                              │
          transcript (in order) ◀── Devanagari ◀── IndicConformer CTC ◀── ffmpeg decode
```

- **Chunking happens in the browser.** An energy-based detector with an adaptive noise floor
  ends a chunk at a short pause once it has ≥10 s of audio, at any pause ≥1.5 s, or at the
  quietest point before 20 s. Chunks are sent one at a time; each shows a placeholder until its
  text arrives, so the transcript stays in order.
- **Why ~10–20 s chunks:** the model is much more accurate on whole verses than on half-lines.
  Feeding the browser chunker the sample verses gave 23% CER with these settings vs 32% when
  cutting at every short pause.
- **The server** decodes any format with ffmpeg; uploads longer than 25 s are split with Silero VAD.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | – | `{status, model}` |
| POST | `/api/transcribe` | multipart `file` (any ffmpeg-readable audio, ≤25 MB) | `{text, segments[{start,end,text}], duration_s, processing_s}` |

## CLI

```bash
sanskrit-stt serve [--host 127.0.0.1] [--port 8000]
sanskrit-stt transcribe recitation.mp3 [--segments]
sanskrit-stt eval [samples/references.tsv]      # character error rate against references
```

Optional environment variables: `SANSKRIT_STT_MODEL` (HF repo id or local model directory),
`SANSKRIT_STT_MAX_SEGMENT_S` (default 25), `SANSKRIT_STT_MIN_SILENCE_MS` (default 1000).

## Accuracy (baseline)

`sanskrit-stt eval` on three Bhagavad Gita verses (i5-12450H, CPU): **mean CER 20.1%** at ~26×
real-time (3.1: 25.3%, 3.3: 14.0%, 3.4: 21.1%). CER ignores spaces because Sanskrit word
boundaries are ambiguous under sandhi. Typical errors are conjuncts and anusvara/visarga.
Three verses from one reciter is a smoke test, not a benchmark: add `file<TAB>reference` rows
for your own recordings to a TSV and run `sanskrit-stt eval path/to/refs.tsv`.

## Layout

```
sanskrit_stt/
  web/            the app: index.html, styles.css, app.js (mic, chunking), capture-worklet.js
  server.py       FastAPI: /api/* + serves web/
  transcriber.py  model loading, VAD splitting
  audio.py        ffmpeg decode → 16 kHz mono float32
  text.py         NFC + Vedic-accent stripping
  metrics.py      CER
  cli.py          serve | transcribe | eval
samples/          CC BY-SA 4.0 Gita recitations + references (see samples/ATTRIBUTION.md)
tests/            pytest; `-m "not model"` skips tests that need the model
```

## Licences

Code: MIT. Model: MIT (AI4Bharat; ONNX export by OpenVoiceOS). Sample audio: CC BY-SA 4.0.
Fonts: Tiro Devanagari Sanskrit and Cormorant Garamond (SIL OFL, via Google Fonts).
