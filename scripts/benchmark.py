"""
Benchmark transcription wall-clock time and realtime factor.

Usage:
    python scripts/benchmark.py <media> [--model small] [--runs 1]
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


def _transcribe_openai_whisper(media_path: str, model_name: str) -> None:
    import whisper  # openai-whisper
    model = whisper.load_model(model_name)
    model.transcribe(media_path)


def _transcribe_via_engine(media_path: str, model_name: str) -> None:
    from engine import create_backend
    backend = create_backend(model_name)
    _info, segments = backend.transcribe(media_path, language=None)
    # The iterator is lazy; the transcription only happens when consumed.
    list(segments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription benchmark")
    parser.add_argument("media", help="Path to audio/video file")
    parser.add_argument("--model", default="small", help="Whisper model size (default: small)")
    parser.add_argument("--runs", type=int, default=1, help="Number of timed runs (default: 1)")
    args = parser.parse_args()

    duration = _media_duration(args.media)
    if duration is not None:
        print(f"Media duration : {duration:.1f}s")
    else:
        print("Media duration : unknown (ffprobe unavailable)")

    # Prefer project engine; fall back to openai-whisper if engine module absent.
    try:
        import engine  # noqa: F401
        transcribe = _transcribe_via_engine
        print("Engine         : project engine (engine.create_backend)")
    except ImportError:
        transcribe = _transcribe_openai_whisper
        print("Engine         : openai-whisper (engine module not yet present)")

    elapsed_times: list[float] = []
    for run in range(1, args.runs + 1):
        print(f"\nRun {run}/{args.runs} ...", flush=True)
        t0 = time.perf_counter()
        transcribe(args.media, args.model)
        elapsed = time.perf_counter() - t0
        elapsed_times.append(elapsed)
        rtf = (duration / elapsed) if duration else None
        rtf_str = f"{rtf:.2f}x" if rtf is not None else "n/a"
        print(f"  elapsed      : {elapsed:.2f}s")
        print(f"  realtime     : {rtf_str}")

    if args.runs > 1:
        avg = sum(elapsed_times) / len(elapsed_times)
        avg_rtf = (duration / avg) if duration else None
        rtf_str = f"{avg_rtf:.2f}x" if avg_rtf is not None else "n/a"
        print(f"\nAverage elapsed: {avg:.2f}s  realtime: {rtf_str}")


if __name__ == "__main__":
    sys.exit(main())
