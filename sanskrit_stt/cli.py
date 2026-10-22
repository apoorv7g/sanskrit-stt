"""Command line: run the web app, transcribe files, or score against references."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sanskrit_stt.metrics import cer
from sanskrit_stt.transcriber import SanskritTranscriber


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from sanskrit_stt.server import create_app

    print(f"Sanskrit STT on http://{args.host}:{args.port}  (first start downloads the model, ~500 MB)")
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    stt = SanskritTranscriber()
    for path in args.files:
        result = stt.transcribe(path)
        if args.segments:
            print(f"# {path}")
            for seg in result.segments:
                print(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text}")
        else:
            print(result.text if len(args.files) == 1 else f"{path}\t{result.text}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score transcripts against a TSV of `file<TAB>reference` (paths relative to the TSV)."""
    stt = SanskritTranscriber()
    stt.warmup()
    tsv = Path(args.references)
    rows = list(csv.reader(tsv.open(encoding="utf-8"), delimiter="\t"))
    total_cer = total_audio = total_proc = 0.0
    for name, reference in rows:
        result = stt.transcribe(tsv.parent / name)
        c = cer(reference, result.text)
        total_cer += c
        total_audio += result.duration_s
        total_proc += result.processing_s
        print(f"{name}\tCER {c:6.1%}\t{result.duration_s:5.1f}s in {result.processing_s:.2f}s")
        print(f"  ref: {reference}\n  hyp: {result.text}")
    print(f"\nmean CER {total_cer / max(len(rows), 1):.1%}  "
          f"speed {total_audio / max(total_proc, 1e-9):.1f}x real-time")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanskrit-stt", description="Sanskrit speech-to-text (Devanagari)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("transcribe", help="transcribe audio files")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--segments", action="store_true", help="print timestamped segments")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("eval", help="score against a references TSV (character error rate)")
    p.add_argument("references", nargs="?", default="samples/references.tsv")
    p.set_defaults(func=cmd_eval)
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
