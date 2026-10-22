# sanskrit-stt

Sanskrit speech → Devanagari text, running fully on CPU.

Uses AI4Bharat's **IndicConformer** Sanskrit model
([`ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large`](https://huggingface.co/ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large), ~120M params, MIT)
through its ONNX export
([`OpenVoiceOS/ai4bharat-indicconformer-sa-onnx`](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-sa-onnx), CTC head)
and [onnx-asr](https://github.com/istupakov/onnx-asr). No NeMo, PyTorch or GPU needed.

```
any audio ──ffmpeg──▶ 16 kHz mono ──(> 25 s? Silero VAD split)──▶ IndicConformer CTC ──▶ clean Devanagari
```

## Requirements

- Python 3.11+
- `ffmpeg` on `PATH`
- ~600 MB disk for the model (downloaded once into the Hugging Face cache)

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[server,dev]"   # drop extras you don't need
.venv/Scripts/sanskrit-stt download                       # fetch + warm up the model
```

## Usage

### CLI

```bash
sanskrit-stt transcribe samples/gita_3_4.wav
sanskrit-stt transcribe --segments lecture.mp3      # timestamped segments
sanskrit-stt transcribe --json a.wav b.ogg          # JSON with segments + timings
sanskrit-stt transcribe --keep-accents vedic.wav    # keep Vedic svara marks (॑ ॒)
sanskrit-stt eval                                   # score against samples/references.tsv
```

### Python

```python
from sanskrit_stt import SanskritTranscriber

stt = SanskritTranscriber()
result = stt.transcribe("recitation.mp3")      # path, raw bytes, or 16 kHz float32 array
print(result.text)
for seg in result.segments:
    print(seg.start, seg.end, seg.text)
```

### HTTP API (for the UI)

```bash
sanskrit-stt serve --port 8000
```

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{status, model, quantization}` |
| POST | `/transcribe?keep_accents=false` | multipart `file` (any ffmpeg-readable audio, ≤100 MB) | `{filename, text, segments[{start,end,text}], duration_s, processing_s}` |

Interactive docs at `http://127.0.0.1:8000/docs`. CORS is open for local UI development.

## Configuration

Environment variables (CLI flags override where available):

| Variable | Default | Notes |
|---|---|---|
| `SANSKRIT_STT_MODEL` | `OpenVoiceOS/ai4bharat-indicconformer-sa-onnx` | HF repo id or local directory with `model.onnx`, `vocab.txt`, `config.json` |
| `SANSKRIT_STT_QUANTIZATION` | fp32 | `int8` is smaller but was ~4× slower and less accurate on an i5-12450H |
| `SANSKRIT_STT_MAX_SEGMENT_S` | `25` | longer audio is split with VAD; model was trained on 5–25 s clips |
| `SANSKRIT_STT_MIN_SILENCE_MS` | `1000` | VAD pieces closer than this are merged into ≤25 s chunks |
| `SANSKRIT_STT_VAD_THRESHOLD` | `0.5` | Silero speech threshold |
| `SANSKRIT_STT_SPEECH_PAD_MS` | `200` | padding around VAD segments |
| `SANSKRIT_STT_KEEP_ACCENTS` | `0` | `1` keeps Vedic accent marks |

## Accuracy and speed (baseline)

`sanskrit-stt eval` on the three Bhagavad Gita samples (i5-12450H, CPU, fp32):

| | CER |
|---|---|
| Gita 3.1 | 25.3% |
| Gita 3.3 | 14.0% |
| Gita 3.4 | 21.1% |
| **mean** | **20.1%** at ~26× real-time |

CER is computed with spaces removed: Sanskrit word boundaries are ambiguous under sandhi, so WER
(~100% here) mostly measures spacing, not recognition. Typical errors are conjunct consonants
(`नैष्कर्म्यं` → `नैष्कर्मिणमि`) and anusvara/visarga. The first call after loading is slow
(warm-up); the CLI `eval`/`download` and the server warm up automatically.

Three verses from one reciter is a smoke test, not a benchmark — evaluate on audio that looks
like your real use case by adding rows to a TSV (`file<TAB>reference`) and running
`sanskrit-stt eval path/to/refs.tsv`.

## Project layout

```
sanskrit_stt/
  audio.py        ffmpeg decode → 16 kHz mono float32
  transcriber.py  model loading, VAD splitting, Transcript/Segment
  text.py         NFC, Vedic-accent stripping, scoring normalisation
  metrics.py      WER / CER
  cli.py          transcribe | eval | download | serve
  server.py       FastAPI app
samples/          CC BY-SA 4.0 Gita recitations + references (see samples/ATTRIBUTION.md)
tests/            pytest (tests marked `model` need the downloaded model)
```

```bash
pytest                  # everything (~15 s once the model is cached)
pytest -m "not model"   # fast tests only
```

## Licences

Code: MIT. Model: MIT (AI4Bharat; ONNX export by OpenVoiceOS). Sample audio: CC BY-SA 4.0.
