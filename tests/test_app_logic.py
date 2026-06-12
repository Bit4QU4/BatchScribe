"""Tests for pure helper functions in main.py — no GUI required."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from main import build_jobs, format_elapsed, language_to_param


def test_format_elapsed_seconds_only():
    assert format_elapsed(45.7) == "45s"


def test_format_elapsed_minutes_and_seconds():
    assert format_elapsed(192.0) == "3m 12s"


def test_format_elapsed_zero():
    assert format_elapsed(0) == "0s"


def test_format_elapsed_exact_minute():
    assert format_elapsed(60) == "1m 0s"


def test_language_to_param_auto():
    assert language_to_param("auto") is None


def test_language_to_param_en():
    assert language_to_param("en") == "en"


def test_language_to_param_other():
    assert language_to_param("fr") == "fr"


def test_build_jobs_basic():
    jobs = build_jobs(["/a/b.mp3", "/c/d.wav"], ["txt", "srt"], None, "en")
    assert len(jobs) == 2
    assert jobs[0].path == "/a/b.mp3"
    assert jobs[0].formats == ["txt", "srt"]
    assert jobs[0].output_dir is None
    assert jobs[0].language == "en"
    assert jobs[1].path == "/c/d.wav"


def test_build_jobs_with_output_dir():
    jobs = build_jobs(["/a/b.mp3"], ["vtt"], "/out", None)
    assert jobs[0].output_dir == "/out"
    assert jobs[0].language is None


def test_build_jobs_empty_outdir_becomes_none():
    jobs = build_jobs(["/a/b.mp3"], ["txt"], "", "en")
    assert jobs[0].output_dir is None


def test_filter_media_paths(tmp_path):
    keep = tmp_path / "a.mp3"
    keep.write_bytes(b"x")
    wrong_ext = tmp_path / "notes.txt"
    wrong_ext.write_bytes(b"x")
    folder = tmp_path / "folder.mp3"  # extension-named directory must be dropped
    folder.mkdir()
    missing = tmp_path / "ghost.wav"
    upper = tmp_path / "B.MP4"
    upper.write_bytes(b"x")

    from main import filter_media_paths
    got = filter_media_paths([str(keep), str(wrong_ext), str(folder), str(missing), str(upper)])
    assert got == [str(keep), str(upper)]
