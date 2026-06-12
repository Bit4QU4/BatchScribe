"""Deterministic adversarial tests for the pure layers (no hypothesis).

Covers writers, config loading, main.py pure helpers, and worker shutdown
semantics.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import config as cfg_mod
from config import AppConfig, load_config
from engine import Segment, TranscriptionBackend, TranscriptionInfo
from main import build_jobs, format_elapsed, language_to_param
from worker import TranscriptionWorker, WorkerCallbacks
from writers import (
    FORMAT_WRITERS,
    _fmt_ts,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)

# ---------------------------------------------------------------------------
# Adversarial segment corpus
# ---------------------------------------------------------------------------

ADVERSARIAL_SEGMENTS = [
    Segment(start=0.0, end=0.0, text=""),                          # empty, zero-length
    Segment(start=1.0, end=1.0, text="zero length"),               # zero-length
    Segment(start=5.0, end=2.0, text="out of order end"),          # end < start
    Segment(start=10.0, end=3.0, text="line1\n\nline2\n\n\nline3"),  # blank lines
    Segment(start=3.0, end=2.5, text="fake 00:00:01.000 --> 00:00:02.000 arrow"),
    Segment(start=12.0, end=13.0, text="--> leading arrow"),
    Segment(start=3700.5, end=7322.125, text="over one hour"),     # >1h timestamps
    Segment(start=0.5, end=0.25, text="你好世界 こんにちは"),
    Segment(start=1.5, end=2.0, text="مرحبا שלום"),
    Segment(start=2.0, end=2.1, text="   surrounded by spaces   "),
    Segment(start=99999.0, end=100000.0, text="very late"),
]


def _blocks(body: str) -> list[str]:
    """Split a VTT/SRT body into cue blocks on blank lines."""
    return [b for b in body.strip().split("\n\n") if b.strip()]


def test_all_writers_survive_adversarial_segments(tmp_path: Path):
    for ext, writer in FORMAT_WRITERS.items():
        out = tmp_path / f"adv.{ext}"
        writer(ADVERSARIAL_SEGMENTS, out)
        assert out.exists()
        # Must be valid UTF-8
        out.read_text(encoding="utf-8")


def test_vtt_structure_with_adversarial_text(tmp_path: Path):
    out = tmp_path / "adv.vtt"
    write_vtt(ADVERSARIAL_SEGMENTS, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT\n\n")
    body = text[len("WEBVTT\n\n"):]
    blocks = _blocks(body)
    # Blank lines inside segment text must not split cues
    assert len(blocks) == len(ADVERSARIAL_SEGMENTS)
    for i, block in enumerate(blocks, 1):
        lines = block.split("\n")
        assert lines[0] == str(i)
        assert " --> " in lines[1]


def test_srt_structure_and_numbering(tmp_path: Path):
    out = tmp_path / "adv.srt"
    write_srt(ADVERSARIAL_SEGMENTS, out)
    blocks = _blocks(out.read_text(encoding="utf-8"))
    assert len(blocks) == len(ADVERSARIAL_SEGMENTS)
    for i, block in enumerate(blocks, 1):
        lines = block.split("\n")
        assert lines[0] == str(i)
        assert "," in lines[1] and " --> " in lines[1]


def test_json_roundtrip_unicode(tmp_path: Path):
    out = tmp_path / "adv.json"
    write_json(ADVERSARIAL_SEGMENTS, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data["segments"], list)
    assert len(data["segments"]) == len(ADVERSARIAL_SEGMENTS)
    # Unicode preserved exactly (ensure_ascii=False)
    assert data["segments"][7]["text"] == ADVERSARIAL_SEGMENTS[7].text
    assert data["segments"][8]["text"] == ADVERSARIAL_SEGMENTS[8].text


def test_txt_empty_segments(tmp_path: Path):
    out = tmp_path / "empty.txt"
    write_txt([], out)
    assert out.read_text(encoding="utf-8") == ""


def test_writers_empty_segment_list(tmp_path: Path):
    for ext, writer in FORMAT_WRITERS.items():
        out = tmp_path / f"none.{ext}"
        writer([], out)
        assert out.exists()
    assert (tmp_path / "none.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert json.loads((tmp_path / "none.json").read_text(encoding="utf-8")) == {"segments": []}


def test_fmt_ts_over_one_hour_and_rounding():
    assert _fmt_ts(3661.5, ".") == "01:01:01.500"
    assert _fmt_ts(3661.5, ",") == "01:01:01,500"
    assert _fmt_ts(0.0, ".") == "00:00:00.000"
    # ms rounding must carry into seconds, never produce ".1000"
    assert _fmt_ts(0.9999, ".") == "00:00:01.000"
    assert _fmt_ts(59.9999, ".") == "00:01:00.000"
    # 100h+ keeps going (no wrap)
    assert _fmt_ts(360000.0, ".") == "100:00:00.000"


def test_writers_10k_segments_perf(tmp_path: Path):
    segs = [
        Segment(start=i * 2.0, end=i * 2.0 + 1.5, text=f"segment number {i} 中文")
        for i in range(10_000)
    ]
    t0 = time.monotonic()
    for ext, writer in FORMAT_WRITERS.items():
        writer(segs, tmp_path / f"big.{ext}")
    elapsed = time.monotonic() - t0
    assert elapsed < 10.0, f"writers too slow on 10k segments: {elapsed:.1f}s"
    blocks = _blocks((tmp_path / "big.srt").read_text(encoding="utf-8"))
    assert len(blocks) == 10_000
    assert blocks[-1].split("\n")[0] == "10000"


# ---------------------------------------------------------------------------
# config.load_config against corrupt input
# ---------------------------------------------------------------------------

def _patched_config_dir(tmp_path: Path, content: str | None):
    d = tmp_path / "cfg"
    d.mkdir(exist_ok=True)
    if content is not None:
        (d / "config.json").write_text(content, encoding="utf-8")
    return lambda: d


def _load_with(tmp_path: Path, content: str | None) -> AppConfig:
    original = cfg_mod.config_dir
    cfg_mod.config_dir = _patched_config_dir(tmp_path, content)
    try:
        return load_config()
    finally:
        cfg_mod.config_dir = original


def test_config_corrupt_json(tmp_path: Path):
    assert _load_with(tmp_path, "{not json").model_size == "small"


def test_config_empty_file(tmp_path: Path):
    assert _load_with(tmp_path, "").theme == "darkly"


def test_config_json_but_not_object(tmp_path: Path):
    assert _load_with(tmp_path, "[1, 2, 3]").formats == ["txt"]
    assert _load_with(tmp_path, "42").formats == ["txt"]
    assert _load_with(tmp_path, "null").formats == ["txt"]


def test_config_wrong_types_fall_back_to_defaults(tmp_path: Path):
    raw = json.dumps({
        "theme": 42,
        "model_size": ["small"],
        "language": None,
        "formats": "txt",
        "output_dir": 7,
    })
    cfg = _load_with(tmp_path, raw)
    assert cfg.theme == "darkly"
    assert cfg.model_size == "small"
    assert cfg.language == "en"
    assert cfg.formats == ["txt"]
    assert cfg.output_dir is None


def test_config_formats_with_non_string_entries(tmp_path: Path):
    cfg = _load_with(tmp_path, json.dumps({"formats": ["txt", 5]}))
    assert cfg.formats == ["txt"]


def test_config_unknown_keys_ignored(tmp_path: Path):
    cfg = _load_with(tmp_path, json.dumps({"bogus": True, "theme": "yeti"}))
    assert cfg.theme == "yeti"
    assert not hasattr(cfg, "bogus")


def test_config_valid_values_applied(tmp_path: Path):
    raw = json.dumps({
        "theme": "yeti",
        "model_size": "large-v3",
        "language": "fr",
        "formats": ["vtt", "srt"],
        "output_dir": "C:\\out",
    })
    cfg = _load_with(tmp_path, raw)
    assert cfg.model_size == "large-v3"
    assert cfg.language == "fr"
    assert cfg.formats == ["vtt", "srt"]
    assert cfg.output_dir == "C:\\out"


def test_config_output_dir_null_allowed(tmp_path: Path):
    cfg = _load_with(tmp_path, json.dumps({"output_dir": None}))
    assert cfg.output_dir is None


# ---------------------------------------------------------------------------
# main.py pure helpers
# ---------------------------------------------------------------------------

def test_format_elapsed_edges():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(59) == "59s"
    assert format_elapsed(59.9) == "59s"
    assert format_elapsed(60) == "1m 0s"
    assert format_elapsed(3600) == "60m 0s"
    assert format_elapsed(-5) == "0s"  # clamped, never "-1m 55s"


def test_language_to_param_edges():
    assert language_to_param("auto") is None
    assert language_to_param("en") == "en"
    assert language_to_param("") == ""  # passthrough for non-"auto"


def test_build_jobs_edges():
    assert build_jobs([], ["txt"], None, "en") == []
    jobs = build_jobs(["/a.mp3"], ["txt"], "", None)
    assert jobs[0].output_dir is None
    assert jobs[0].language is None


# ---------------------------------------------------------------------------
# worker shutdown semantics (regression for stale-worker leak)
# ---------------------------------------------------------------------------

class _InstantBackend(TranscriptionBackend):
    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def transcribe(
        self, path: str, language: str | None = "en"
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        return TranscriptionInfo(duration=1.0, language="en"), iter(
            [Segment(0.0, 1.0, "hello")]
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "cpu"


def test_worker_shutdown_exits_thread_and_silences_callbacks(tmp_path: Path):
    fired: list[str] = []
    cbs = WorkerCallbacks(
        on_status=lambda t: fired.append(t),
        on_batch_done=lambda o, f, e, sp: fired.append("batch"),
    )
    worker = TranscriptionWorker(
        backend=_InstantBackend(), dispatch=lambda fn: fn(), callbacks=cbs
    )
    worker.shutdown()
    worker._thread.join(timeout=5)
    assert not worker._thread.is_alive(), "worker thread did not exit after shutdown"
    assert fired == [], "callbacks fired after shutdown"


def test_worker_shutdown_cancels_in_flight_batch(tmp_path: Path):
    import threading

    from worker import TranscriptionJob

    started = threading.Event()
    gate = threading.Event()

    class _GatedBackend(_InstantBackend):
        def transcribe(self, path, language="en"):
            started.set()
            gate.wait(timeout=5)
            return super().transcribe(path, language)

    fired: list[str] = []
    cbs = WorkerCallbacks(
        on_batch_done=lambda o, f, e, sp: fired.append("batch"),
        on_file_done=lambda r: fired.append("file"),
    )
    worker = TranscriptionWorker(
        backend=_GatedBackend(), dispatch=lambda fn: fn(), callbacks=cbs
    )
    job = TranscriptionJob(
        path=str(tmp_path / "a.mp3"), formats=["txt"], output_dir=str(tmp_path), language="en"
    )
    worker.submit([job])
    assert started.wait(timeout=5), "worker never started the batch"
    worker.shutdown()  # while transcribe is blocked on the gate
    gate.set()
    worker._thread.join(timeout=5)
    assert not worker._thread.is_alive()
    assert fired == [], "callbacks escaped after shutdown"
    assert not (tmp_path / "a.txt").exists(), "output written after shutdown"
