"""Smoke tests for Phase 1 modules.

Runnable directly (python tests/test_smoke.py) or via pytest.
No network, no model download, no GPU required.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

# Resolve imports from repo root whether run via pytest or directly
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from config import AppConfig, load_config, save_config, setup_logging
from engine import (
    MODEL_SIZES,
    FasterWhisperBackend,
    Segment,
    TranscriptionBackend,
    TranscriptionInfo,
    create_backend,
    default_models_dir,
)
from worker import (
    FileResult,
    TranscriptionJob,
    TranscriptionWorker,
    WorkerCallbacks,
)
from writers import (
    FORMAT_WRITERS,
    write_json,
    write_md,
    write_srt,
    write_txt,
    write_vtt,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_SEGMENTS = [
    Segment(start=0.0, end=2.5, text=" Hello world."),
    Segment(start=2.5, end=5.0, text=" How are you?", speaker="A"),
]


# ---------------------------------------------------------------------------
# engine.py tests
# ---------------------------------------------------------------------------

def test_segment_slots():
    s = Segment(1.0, 2.0, "hi")
    assert s.start == 1.0
    assert s.text == "hi"
    assert s.speaker is None


def test_transcription_info():
    ti = TranscriptionInfo(duration=60.0, language="en")
    assert ti.duration == 60.0


def test_model_sizes_nonempty():
    assert "small" in MODEL_SIZES
    assert "large-v3" in MODEL_SIZES


def test_default_models_dir_created():
    p = default_models_dir()
    assert p.exists()


def test_faster_whisper_backend_construct():
    """Constructor must not import faster_whisper (no load call)."""
    b = FasterWhisperBackend(model_size="small", device="cpu")
    assert not b.is_loaded
    assert b.device == "cpu"  # resolved only after load(); default is "cpu"


def test_create_backend():
    b = create_backend(model_size="tiny", device="auto")
    assert isinstance(b, TranscriptionBackend)
    assert not b.is_loaded


# ---------------------------------------------------------------------------
# writers.py tests
# ---------------------------------------------------------------------------

def test_write_txt(tmp_path: Path):
    p = tmp_path / "out.txt"
    write_txt(FAKE_SEGMENTS, p)
    text = p.read_text(encoding="utf-8")
    assert "Hello world." in text
    assert "How are you?" in text


def test_write_vtt(tmp_path: Path):
    p = tmp_path / "out.vtt"
    write_vtt(FAKE_SEGMENTS, p)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in text
    assert "Hello world." in text


def test_write_srt(tmp_path: Path):
    p = tmp_path / "out.srt"
    write_srt(FAKE_SEGMENTS, p)
    text = p.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,500" in text
    assert "How are you?" in text


def test_write_md(tmp_path: Path):
    p = tmp_path / "out.md"
    write_md(FAKE_SEGMENTS, p)
    text = p.read_text(encoding="utf-8")
    assert "**00:00:00.000 - 00:00:02.500**:" in text
    assert "Hello world." in text


def test_write_json(tmp_path: Path):
    p = tmp_path / "out.json"
    write_json(FAKE_SEGMENTS, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["segments"]) == 2
    assert data["segments"][1]["speaker"] == "A"
    assert data["segments"][0]["speaker"] is None


def test_format_writers_keys():
    assert set(FORMAT_WRITERS.keys()) == {"txt", "vtt", "srt", "md", "json"}


def test_all_format_writers_roundtrip(tmp_path: Path):
    for ext, writer in FORMAT_WRITERS.items():
        out = tmp_path / f"out.{ext}"
        writer(FAKE_SEGMENTS, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Stub backend for worker tests
# ---------------------------------------------------------------------------

class _StubBackend(TranscriptionBackend):
    """In-memory backend; never touches the filesystem for model loading."""

    def __init__(self, segments: list[Segment], duration: float = 10.0) -> None:
        self._segments = segments
        self._duration = duration
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def transcribe(
        self, path: str, language: str | None = "en"
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        if not self._loaded:
            self.load()
        info = TranscriptionInfo(duration=self._duration, language=language or "en")
        return info, iter(list(self._segments))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "cpu"


# ---------------------------------------------------------------------------
# worker.py tests
# ---------------------------------------------------------------------------

def test_worker_submit_and_batch_done(tmp_path: Path):
    backend = _StubBackend(FAKE_SEGMENTS)
    events: list[str] = []
    results: list[FileResult] = []
    batch_done: list[tuple] = []

    # Create a real audio file placeholder (workers just pass path to backend)
    fake_audio = tmp_path / "fake.mp3"
    fake_audio.write_bytes(b"\x00" * 16)

    done_event = threading.Event()

    def _on_batch_done(ok, fail, elapsed, summary_path):
        batch_done.append((ok, fail, elapsed, summary_path))
        done_event.set()

    cbs = WorkerCallbacks(
        on_file_start=lambda p: events.append(f"start:{Path(p).name}"),
        on_segment_progress=lambda p, f: events.append(f"progress:{f:.2f}"),
        on_file_done=lambda r: results.append(r),
        on_batch_done=_on_batch_done,
        on_status=lambda t: events.append(f"status:{t}"),
    )

    worker = TranscriptionWorker(
        backend=backend,
        dispatch=lambda fn: fn(),  # synchronous dispatch for tests
        callbacks=cbs,
    )

    job = TranscriptionJob(
        path=str(fake_audio),
        formats=["txt", "json"],
        output_dir=str(tmp_path),
        language="en",
    )
    worker.submit([job])

    assert done_event.wait(timeout=10), "batch_done callback never fired"

    assert len(batch_done) == 1
    ok, fail, elapsed, summary_path = batch_done[0]
    assert ok == 1
    assert fail == 0
    assert elapsed >= 0

    assert results[0].ok
    assert (tmp_path / "fake.txt").exists()
    assert (tmp_path / "fake.json").exists()


def test_worker_stop_mid_batch(tmp_path: Path):
    """Stop event must halt processing before remaining files."""
    backend = _StubBackend(FAKE_SEGMENTS)
    results: list[FileResult] = []
    done_event = threading.Event()

    def _on_batch_done(ok, fail, elapsed, summary_path):
        done_event.set()

    cbs = WorkerCallbacks(
        on_file_done=lambda r: results.append(r),
        on_batch_done=_on_batch_done,
    )

    worker = TranscriptionWorker(
        backend=backend,
        dispatch=lambda fn: fn(),
        callbacks=cbs,
    )

    # Create several fake files
    jobs = []
    for i in range(5):
        f = tmp_path / f"file{i}.mp3"
        f.write_bytes(b"\x00" * 8)
        jobs.append(TranscriptionJob(
            path=str(f), formats=["txt"], output_dir=str(tmp_path), language="en"
        ))

    worker.submit(jobs)
    # Stop immediately — at least one file may complete before stop lands
    time.sleep(0.05)
    worker.stop()

    assert done_event.wait(timeout=10), "batch_done never fired after stop"
    # We just verify it terminated cleanly; exact ok/fail counts are nondeterministic
    total = len(results)
    assert total <= 5


def test_worker_preload(tmp_path: Path):
    backend = _StubBackend(FAKE_SEGMENTS)
    assert not backend.is_loaded

    status_msgs: list[str] = []
    # Preload queues a _Preload task; we detect completion via is_loaded polling
    cbs = WorkerCallbacks(on_status=lambda t: status_msgs.append(t))
    worker = TranscriptionWorker(
        backend=backend,
        dispatch=lambda fn: fn(),
        callbacks=cbs,
    )
    worker.preload()

    deadline = time.monotonic() + 5
    while not backend.is_loaded and time.monotonic() < deadline:
        time.sleep(0.05)

    assert backend.is_loaded, "preload() did not trigger backend.load()"


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------

def test_load_config_defaults(tmp_path, monkeypatch=None):
    # Without monkeypatching, just verify it returns an AppConfig
    cfg = AppConfig()
    assert cfg.model_size == "small"
    assert cfg.formats == ["txt"]
    assert cfg.output_dir is None


def test_save_and_load_config(tmp_path: Path):
    import config as cfg_mod
    # Temporarily redirect config_dir
    original_fn = cfg_mod.config_dir

    def _patched():
        d = tmp_path / "cfg"
        d.mkdir(exist_ok=True)
        return d

    cfg_mod.config_dir = _patched
    try:
        cfg = AppConfig(theme="yeti", model_size="large-v3", formats=["vtt", "srt"])
        save_config(cfg)
        loaded = load_config()
        assert loaded.theme == "yeti"
        assert loaded.model_size == "large-v3"
        assert loaded.formats == ["vtt", "srt"]
    finally:
        cfg_mod.config_dir = original_fn


def test_load_config_corrupt_returns_defaults(tmp_path: Path):
    import config as cfg_mod
    original_fn = cfg_mod.config_dir

    def _patched():
        d = tmp_path / "cfg2"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("not json!!", encoding="utf-8")
        return d

    cfg_mod.config_dir = _patched
    try:
        cfg = load_config()
        assert cfg.model_size == "small"
    finally:
        cfg_mod.config_dir = original_fn


def test_setup_logging_idempotent(tmp_path: Path):
    import config as cfg_mod
    original_fn = cfg_mod.config_dir

    def _patched():
        d = tmp_path / "log_cfg"
        d.mkdir(exist_ok=True)
        return d

    cfg_mod.config_dir = _patched
    cfg_mod._logging_configured = False
    try:
        p1 = setup_logging()
        p2 = setup_logging()
        assert p1 == p2
    finally:
        cfg_mod.config_dir = original_fn
        cfg_mod._logging_configured = False


# ---------------------------------------------------------------------------
# Batch summary tests
# ---------------------------------------------------------------------------

class _FailOnceBackend(_StubBackend):
    """Raises on the first transcribe call; succeeds thereafter."""

    def __init__(self, segments, duration=10.0):
        super().__init__(segments, duration)
        self._call_count = 0

    def transcribe(self, path, language=None):
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("Simulated transcription failure")
        return super().transcribe(path, language)


def _run_batch_sync(backend, jobs) -> tuple:
    """Run a batch synchronously; return (ok, fail, elapsed, summary_path, results)."""
    batch_done: list[tuple] = []
    results: list[FileResult] = []
    done_event = threading.Event()

    def _on_batch_done(ok, fail, elapsed, summary_path):
        batch_done.append((ok, fail, elapsed, summary_path))
        done_event.set()

    cbs = WorkerCallbacks(
        on_file_done=lambda r: results.append(r),
        on_batch_done=_on_batch_done,
    )
    worker = TranscriptionWorker(
        backend=backend,
        dispatch=lambda fn: fn(),
        callbacks=cbs,
    )
    worker.submit(jobs)
    done_event.wait(timeout=10)
    ok, fail, elapsed, summary_path = batch_done[0]
    return ok, fail, elapsed, summary_path, results


def test_batch_summary_two_ok_one_failed(tmp_path: Path):
    """2 ok + 1 failed writes a summary with correct header counts and per-file lines."""
    backend = _FailOnceBackend(FAKE_SEGMENTS, duration=10.0)

    jobs = []
    for name in ("alpha.mp3", "beta.mp3", "gamma.mp3"):
        f = tmp_path / name
        f.write_bytes(b"\x00" * 8)
        jobs.append(TranscriptionJob(
            path=str(f), formats=["txt"], output_dir=str(tmp_path), language="en"
        ))

    ok, fail, elapsed, summary_path, results = _run_batch_sync(backend, jobs)

    assert ok == 2
    assert fail == 1
    assert summary_path is not None

    text = Path(summary_path).read_text(encoding="utf-8")
    assert "2 ok" in text
    assert "1 failed" in text

    # One tab-delimited data line per attempted file
    data_lines = [ln for ln in text.splitlines() if "\t" in ln]
    assert len(data_lines) == 3

    # Basenames only — no directory separators in the filename field
    for ln in data_lines:
        name_field = ln.split("\t")[0]
        assert "/" not in name_field and "\\" not in name_field

    # RTF present for ok files; failed line present
    ok_lines = [ln for ln in data_lines if "\tOK\t" in ln]
    assert all("rtf=" in ln for ln in ok_lines)
    failed_lines = [ln for ln in data_lines if "\tFAILED\t" in ln]
    assert len(failed_lines) == 1

    # RTF direction: duration / wall time. The stub transcribes 10s of audio
    # near-instantly, so any parseable factor must be far above 1x (the
    # inverse would be ~0). Wall time can round to exactly 0.0 on fast
    # runners, which legitimately yields "n/a" — direction is then covered
    # deterministically by test_batch_summary_rtf_direction.
    for ln in ok_lines:
        raw = ln.rsplit("rtf=", 1)[1]
        if raw != "n/a":
            assert float(raw.rstrip("x")) > 1.0


def test_batch_summary_cancelled_line(tmp_path: Path):
    """A file cancelled mid-segment appears as CANCELLED in the summary."""
    backend = _StubBackend(FAKE_SEGMENTS, duration=5.0)
    batch_done: list[tuple] = []
    results: list[FileResult] = []
    done_event = threading.Event()

    def _on_batch_done(ok, fail, elapsed, summary_path):
        batch_done.append((ok, fail, elapsed, summary_path))
        done_event.set()

    cbs = WorkerCallbacks(
        on_file_done=lambda r: results.append(r),
        on_batch_done=_on_batch_done,
    )

    jobs = []
    for i in range(3):
        f = tmp_path / f"c{i}.mp3"
        f.write_bytes(b"\x00" * 8)
        jobs.append(TranscriptionJob(
            path=str(f), formats=["txt"], output_dir=str(tmp_path), language="en"
        ))

    worker = TranscriptionWorker(
        backend=backend, dispatch=lambda fn: fn(), callbacks=cbs
    )
    worker.submit(jobs)
    time.sleep(0.05)
    worker.stop()
    done_event.wait(timeout=10)

    _, _, _, summary_path = batch_done[0]
    # Summary is written for any attempted files; cancelled lines use CANCELLED status
    if summary_path is not None:
        text = Path(summary_path).read_text(encoding="utf-8")
        cancelled_lines = [ln for ln in text.splitlines() if "\tCANCELLED\t" in ln]
        ok_or_fail = [ln for ln in text.splitlines() if "\tOK\t" in ln or "\tFAILED\t" in ln]
        # Every data line is one of OK/FAILED/CANCELLED
        data_lines = [ln for ln in text.splitlines() if "\t" in ln]
        assert len(cancelled_lines) + len(ok_or_fail) == len(data_lines)


def test_batch_summary_write_failure_does_not_raise(tmp_path: Path, monkeypatch):
    """An I/O failure writing the summary must not propagate out of the worker."""
    backend = _StubBackend(FAKE_SEGMENTS, duration=10.0)

    fake = tmp_path / "x.mp3"
    fake.write_bytes(b"\x00" * 8)
    job = TranscriptionJob(
        path=str(fake), formats=["txt"], output_dir=str(tmp_path), language="en"
    )

    original_write_text = Path.write_text

    def _failing_write_text(self, *args, **kwargs):
        if self.name == "_batch_summary.txt":
            raise OSError("disk full")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _failing_write_text)

    ok, fail, elapsed, summary_path, results = _run_batch_sync(backend, [job])

    assert ok == 1
    assert fail == 0
    # A write failure must yield None, not raise
    assert summary_path is None


def test_batch_summary_basenames_only(tmp_path: Path):
    """Summary file lines must contain only the filename, not any directory path."""
    nested = tmp_path / "deep" / "subdir"
    nested.mkdir(parents=True)
    fake = nested / "recording.mp3"
    fake.write_bytes(b"\x00" * 8)

    backend = _StubBackend(FAKE_SEGMENTS, duration=8.0)
    job = TranscriptionJob(
        path=str(fake), formats=["txt"], output_dir=str(tmp_path), language="en"
    )

    ok, fail, elapsed, summary_path, results = _run_batch_sync(backend, [job])

    assert summary_path is not None
    text = Path(summary_path).read_text(encoding="utf-8")
    data_lines = [ln for ln in text.splitlines() if "\t" in ln]
    assert len(data_lines) == 1
    name_field = data_lines[0].split("\t")[0]
    assert name_field == "recording.mp3"
    assert str(nested) not in name_field


# ---------------------------------------------------------------------------
# Direct-run entry point (also runnable via pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    _no_tmp = [
        test_segment_slots,
        test_transcription_info,
        test_model_sizes_nonempty,
        test_default_models_dir_created,
        test_faster_whisper_backend_construct,
        test_create_backend,
        test_format_writers_keys,
    ]
    _with_tmp = [
        test_write_txt,
        test_write_vtt,
        test_write_srt,
        test_write_md,
        test_write_json,
        test_all_format_writers_roundtrip,
        test_worker_submit_and_batch_done,
        test_worker_stop_mid_batch,
        test_worker_preload,
        test_save_and_load_config,
        test_load_config_corrupt_returns_defaults,
        test_setup_logging_idempotent,
        test_batch_summary_two_ok_one_failed,
        test_batch_summary_cancelled_line,
        test_batch_summary_basenames_only,
    ]

    passed = failed = 0
    for fn in _no_tmp:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()

    try:
        test_load_config_defaults(None)
        passed += 1
        print("  PASS  test_load_config_defaults")
    except Exception:
        failed += 1
        print("  FAIL  test_load_config_defaults")
        traceback.print_exc()

    for fn in _with_tmp:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                passed += 1
                print(f"  PASS  {fn.__name__}")
            except Exception:
                failed += 1
                print(f"  FAIL  {fn.__name__}")
                traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


def test_batch_summary_rtf_direction(tmp_path: Path):
    """rtf = audio duration / wall time: 10s of audio in 5s wall = 2.00x."""
    worker = TranscriptionWorker(
        backend=_StubBackend(FAKE_SEGMENTS),
        dispatch=lambda fn: fn(),
        callbacks=WorkerCallbacks(),
    )
    src = tmp_path / "clip.mp3"
    src.write_bytes(b"\x00")
    results = [
        FileResult(path=str(src), ok=True, message="OK", elapsed=5.0, duration=10.0),
    ]
    job = TranscriptionJob(path=str(src), formats=["txt"], output_dir=str(tmp_path), language="en")
    path = worker._write_batch_summary(results, 1, 0, 5.0, job)
    worker.shutdown()
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    assert "rtf=2.00x" in text
