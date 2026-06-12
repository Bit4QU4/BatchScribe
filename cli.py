"""Headless batch transcription CLI — no tkinter, no GUI imports."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from engine import LANGUAGE_CHOICES, MODEL_SIZES, create_backend
from writers import FORMAT_WRITERS


def _language_to_param(lang: str) -> str | None:
    # "auto" means let the model detect; all other values are passed through.
    return None if lang == "auto" else lang


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("batchscribe")
    except PackageNotFoundError:
        # Running from a source checkout rather than an installed package.
        return "0.3.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batchscribe",
        description="Batch transcription via faster-whisper (headless).",
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="Audio files to transcribe.")
    parser.add_argument(
        "--model",
        choices=MODEL_SIZES,
        default="small",
        help="Whisper model size (default: small).",
    )
    parser.add_argument(
        "--lang",
        choices=LANGUAGE_CHOICES,
        default="en",
        help="Transcription language or 'auto' (default: en).",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=list(FORMAT_WRITERS.keys()),
        metavar="FMT",
        help="Output format; may be repeated. Choices: %(choices)s. Default: txt.",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Output directory (default: alongside each source file).",
    )
    parser.add_argument("--prompt", default=None, help="Initial prompt / vocabulary hint.")
    parser.add_argument(
        "--strict-vad",
        action="store_true",
        help="Enable stricter VAD parameters to reduce hallucinations.",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Batched GPU inference (~2.5x faster on CUDA; ignored on CPU).",
    )
    parser.add_argument("--version", action="version", version=f"batchscribe {_version()}")

    args = parser.parse_args(argv)

    formats: list[str] = args.formats if args.formats else ["txt"]

    # Validate all input files before touching the model.
    missing = [f for f in args.files if not Path(f).exists()]
    if missing:
        for f in missing:
            print(f"error: file not found: {f}", file=sys.stderr)
        return 2

    # Create output directory once if --out is given.
    out_dir: Path | None = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    language = _language_to_param(args.lang)

    try:
        backend = create_backend(
            model_size=args.model,
            device="auto",
            language=language,
            initial_prompt=args.prompt,
            strict_vad=args.strict_vad,
            batched=args.batched,
        )
        backend.load()
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130

    ok_count = 0
    fail_count = 0
    batch_start = time.monotonic()

    try:
        for file_path_str in args.files:
            src = Path(file_path_str)
            stem = src.stem
            dest_dir = out_dir if out_dir is not None else src.parent

            file_start = time.monotonic()
            try:
                _info, seg_iter = backend.transcribe(
                    str(src),
                    language=language,
                    initial_prompt=args.prompt,
                )
                # Materialise the iterator once so all writers share the same data.
                segments = list(seg_iter)
                for fmt in formats:
                    out_path = dest_dir / f"{stem}.{fmt}"
                    FORMAT_WRITERS[fmt](segments, out_path)
                elapsed = time.monotonic() - file_start
                print(f"{src.name}: OK ({elapsed:.1f}s)")
                ok_count += 1
            except Exception as exc:
                print(f"{src.name}: FAILED ({exc})")
                fail_count += 1
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130

    total_elapsed = time.monotonic() - batch_start
    print(f"{ok_count} ok, {fail_count} failed in {total_elapsed:.1f}s")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
