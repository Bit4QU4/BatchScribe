"""
Benchmark transcription wall-clock time and realtime factor.

Usage:
    python scripts/benchmark.py <media> [--model small] [--runs 1] [--files 1] [--batched]
"""
import argparse
import subprocess
import sys
import time


def _media_duration(path: str) -> float | None:
    """Return duration in seconds via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _transcribe_openai_whisper(
    model_obj: object, media_path: str
) -> None:
    """Transcribe via openai-whisper model object."""
    model_obj.transcribe(media_path)


def _transcribe_via_engine_single(
    model_obj: object, media_path: str
) -> None:
    """Transcribe via project engine backend object."""
    _info, segments = model_obj.transcribe(media_path, language=None)
    list(segments)


def _transcribe_via_engine_batched(
    model_obj: object, media_path: str
) -> None:
    """Transcribe via BatchedInferencePipeline wrapper."""
    # Pipeline returns (segments, info), the reverse of the engine backend.
    segments, _info = model_obj.transcribe(media_path, language=None, batch_size=16)
    list(segments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription benchmark")
    parser.add_argument("media", help="Path to audio/video file")
    parser.add_argument("--model", default="small", help="Whisper model size (default: small)")
    parser.add_argument("--runs", type=int, default=1, help="Number of timed runs (default: 1)")
    parser.add_argument(
        "--files",
        type=int,
        default=1,
        help="Transcribe file N times sequentially (default: 1)",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Wrap model in BatchedInferencePipeline (faster-whisper only)",
    )
    args = parser.parse_args()

    duration = _media_duration(args.media)
    if duration is not None:
        print(f"Media duration : {duration:.1f}s")
    else:
        print("Media duration : unknown (ffprobe unavailable)")

    # Prefer project engine; fall back to openai-whisper if engine module absent.
    try:
        import engine  # noqa: F401
        use_project_engine = True
        print("Engine         : project engine (engine.create_backend)")
    except ImportError:
        use_project_engine = False
        print("Engine         : openai-whisper (engine module not yet present)")

    if args.batched and not use_project_engine:
        print("Error: --batched requires project engine (engine module)", file=sys.stderr)
        return 1

    # Load model once (timed), reuse across runs and files.
    print()
    t_load_start = time.perf_counter()
    if use_project_engine:
        from engine import create_backend
        model_obj = create_backend(args.model)
        model_obj.load()
    else:
        import whisper
        model_obj = whisper.load_model(args.model)
        transcribe_fn = _transcribe_openai_whisper
    load_s = time.perf_counter() - t_load_start
    print(f"Model load time: {load_s:.2f}s")

    if use_project_engine:
        if args.batched:
            try:
                from faster_whisper import BatchedInferencePipeline
                # The pipeline wraps the raw WhisperModel, not the backend.
                model_obj = BatchedInferencePipeline(model=model_obj._model)
                print("Pipeline       : BatchedInferencePipeline (batch_size=16)")
                transcribe_fn = _transcribe_via_engine_batched
            except (ImportError, AttributeError) as e:
                print(
                    f"Error: BatchedInferencePipeline unavailable: {e}",
                    file=sys.stderr,
                )
                return 1
        else:
            transcribe_fn = _transcribe_via_engine_single

    # Run benchmark loops: outer loop for --runs, inner loop for --files.
    all_file_times: list[float] = []
    for run in range(1, args.runs + 1):
        print(f"\nRun {run}/{args.runs} ...", flush=True)
        run_file_times: list[float] = []
        for file_idx in range(1, args.files + 1):
            t0 = time.perf_counter()
            transcribe_fn(model_obj, args.media)
            elapsed = time.perf_counter() - t0
            run_file_times.append(elapsed)
            all_file_times.append(elapsed)
            if args.files > 1:
                rtf = (duration / elapsed) if duration else None
                rtf_str = f"{rtf:.2f}x" if rtf is not None else "n/a"
                print(f"  file {file_idx}      : {elapsed:.2f}s  realtime: {rtf_str}")
        if args.files == 1:
            elapsed = run_file_times[0]
            rtf = (duration / elapsed) if duration else None
            rtf_str = f"{rtf:.2f}x" if rtf is not None else "n/a"
            print(f"  elapsed      : {elapsed:.2f}s")
            print(f"  realtime     : {rtf_str}")
        else:
            run_total = sum(run_file_times)
            run_avg = run_total / len(run_file_times)
            rtf = (duration / run_avg) if duration else None
            rtf_str = f"{rtf:.2f}x" if rtf is not None else "n/a"
            print(f"  run total    : {run_total:.2f}s  mean rtf: {rtf_str}")

    # Summary line: machine-greppable format.
    if args.runs > 1 or args.files > 1:
        avg_inference = sum(all_file_times) / len(all_file_times)
        total_wall = sum(all_file_times)
        mean_rtf = (duration / avg_inference) if duration else None
        print()
        print(f"load_s={load_s:.2f} inference_s={avg_inference:.2f}", end="")
        if mean_rtf is not None:
            print(f" rtf={mean_rtf:.2f}", end="")
        print()
        print(f"Total wall time: {total_wall:.2f}s ({args.runs * args.files} transcriptions)")


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if isinstance(rc, int) else 0)
