"""Command line: transcribe files, score against references, pre-download the model, run the API."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sanskrit_stt.config import Settings
from sanskrit_stt.metrics import cer, wer
from sanskrit_stt.transcriber import SanskritTranscriber


def _transcriber(args: argparse.Namespace) -> SanskritTranscriber:
    settings = Settings()
    if getattr(args, "model", None):
        settings.model = args.model
    if getattr(args, "int8", False):
        settings.quantization = "int8"
    return SanskritTranscriber(settings)


def cmd_transcribe(args: argparse.Namespace) -> int:
    stt = _transcriber(args)
    results = []
    for path in args.files:
        result = stt.transcribe(path, keep_accents=args.keep_accents)
        if args.json:
            results.append({"file": str(path), **result.to_dict()})
        elif args.segments:
            print(f"# {path}")
            for seg in result.segments:
                print(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text}")
        else:
            print(result.text if len(args.files) == 1 else f"{path}\t{result.text}")
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score transcripts against a TSV of `file<TAB>reference` (paths relative to the TSV)."""
    stt = _transcriber(args)
    stt.warmup()
    tsv = Path(args.references)
    rows = list(csv.reader(tsv.open(encoding="utf-8"), delimiter="\t"))
    total_cer = total_wer = total_audio = total_proc = 0.0
    for name, reference in rows:
        result = stt.transcribe(tsv.parent / name)
        c, w = cer(reference, result.text), wer(reference, result.text)
        total_cer += c
        total_wer += w
        total_audio += result.duration_s
        total_proc += result.processing_s
        print(f"{name}\tCER {c:6.1%}\tWER {w:6.1%}\t{result.duration_s:5.1f}s in {result.processing_s:.2f}s")
        print(f"  ref: {reference}\n  hyp: {result.text}")
    n = max(len(rows), 1)
    print(f"\nmean CER {total_cer / n:.1%}  mean WER {total_wer / n:.1%}  "
          f"speed {total_audio / max(total_proc, 1e-9):.1f}x real-time")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    _transcriber(args).warmup()
    print("model ready")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from sanskrit_stt.server import create_app

    uvicorn.run(create_app(_transcriber(args)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanskrit-stt", description="Sanskrit speech-to-text (Devanagari)")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="HF repo id or local model directory")
    common.add_argument("--int8", action="store_true", help="use int8 weights (smaller, slower on most x86 CPUs)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", parents=[common], help="transcribe audio files")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", help="print JSON with segments and timings")
    p.add_argument("--segments", action="store_true", help="print timestamped segments")
    p.add_argument("--keep-accents", action="store_true", help="keep Vedic svara marks")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("eval", parents=[common], help="score against a references TSV")
    p.add_argument("references", nargs="?", default="samples/references.tsv")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("download", parents=[common], help="download and warm up the model")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print Devanagari.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
