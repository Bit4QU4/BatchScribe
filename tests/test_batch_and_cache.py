"""Tests for batch-progress helper and model_is_cached.

No network, no model download, no GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from engine import model_is_cached
from main import batch_progress_text  # noqa: E402

# ---------------------------------------------------------------------------
# batch_progress_text
# ---------------------------------------------------------------------------

def test_batch_progress_no_completions_yet():
    """Before any file finishes, no ETA is shown."""
    txt = batch_progress_text(started=1, total=5, elapsed=10.0, completed_wall_times=[])
    assert "File 1 of 5" in txt
    assert "10s elapsed" in txt
    assert "remaining" not in txt


def test_batch_progress_mid_batch_with_eta():
    """After one completion, ETA is estimated from mean wall time x files remaining.

    remaining = total - len(completed_wall_times) = 5 - 1 = 4.
    ETA = mean_wall (30s) * 4 = 120s = 2m 0s.
    """
    txt = batch_progress_text(
        started=2,
        total=5,
        elapsed=35.0,
        completed_wall_times=[30.0],
    )
    assert "File 2 of 5" in txt
    assert "remaining" in txt
    # 30s * 4 remaining = 120s = 2m 0s
    assert "2m 0s" in txt


def test_batch_progress_last_file_no_remaining():
    """When all started files are done, remaining count is 0 — no ETA phrase."""
    txt = batch_progress_text(
        started=3,
        total=3,
        elapsed=90.0,
        completed_wall_times=[30.0, 30.0, 30.0],
    )
    assert "File 3 of 3" in txt
    assert "remaining" not in txt


def test_batch_progress_single_file():
    """Single-file batch shows no ETA even after completion."""
    txt = batch_progress_text(
        started=1,
        total=1,
        elapsed=15.0,
        completed_wall_times=[],
    )
    assert "File 1 of 1" in txt
    assert "remaining" not in txt


def test_batch_progress_elapsed_formatting():
    """format_elapsed is used: values >= 60s show Xm Ys."""
    txt = batch_progress_text(started=1, total=2, elapsed=130.0, completed_wall_times=[])
    assert "2m 10s elapsed" in txt


# ---------------------------------------------------------------------------
# model_is_cached
# ---------------------------------------------------------------------------

def _make_snapshot(base: Path, model_name: str, hash_dir: str = "abc123") -> Path:
    """Create a fake HF snapshot directory tree and return the snapshot leaf."""
    snap = base / f"models--Systran--faster-whisper-{model_name}" / "snapshots" / hash_dir
    snap.mkdir(parents=True)
    # Add a dummy file so the directory is non-empty
    (snap / "model.bin").write_bytes(b"\x00")
    return snap


def test_model_is_cached_true(tmp_path: Path):
    _make_snapshot(tmp_path, "small")
    assert model_is_cached("small", "fr", models_dir=tmp_path) is True


def test_model_is_cached_false_missing_dir(tmp_path: Path):
    # Nothing created under tmp_path
    assert model_is_cached("small", "fr", models_dir=tmp_path) is False


def test_model_is_cached_false_empty_snapshot(tmp_path: Path):
    # Snapshot directory exists but is empty
    snap = tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert model_is_cached("small", "fr", models_dir=tmp_path) is False


def test_model_is_cached_en_variant_cached(tmp_path: Path):
    """When language='en' and model supports .en variant, checks for the .en model name."""
    _make_snapshot(tmp_path, "small.en")
    assert model_is_cached("small", "en", models_dir=tmp_path) is True


def test_model_is_cached_en_variant_not_present(tmp_path: Path):
    """If the .en model is absent but the base is cached, returns False for en language."""
    _make_snapshot(tmp_path, "small")
    # 'en' language -> looks for small.en, which is not there
    assert model_is_cached("small", "en", models_dir=tmp_path) is False


def test_model_is_cached_large_no_en_variant(tmp_path: Path):
    """large-v3 has no .en variant; language='en' still resolves to 'large-v3'."""
    _make_snapshot(tmp_path, "large-v3")
    assert model_is_cached("large-v3", "en", models_dir=tmp_path) is True


def test_model_is_cached_none_language(tmp_path: Path):
    """language=None (auto-detect) uses the base model name, not .en."""
    _make_snapshot(tmp_path, "base")
    assert model_is_cached("base", None, models_dir=tmp_path) is True


def test_model_is_cached_no_snapshots_subdir(tmp_path: Path):
    """Model dir exists but lacks a snapshots/ subdirectory -> not cached."""
    model_dir = tmp_path / "models--Systran--faster-whisper-tiny"
    model_dir.mkdir(parents=True)
    assert model_is_cached("tiny", None, models_dir=tmp_path) is False


def test_model_is_cached_distil_repo_name(tmp_path):
    """distil sizes resolve to faster-distil-whisper-* HF repo directories."""
    snap = (
        tmp_path
        / "models--Systran--faster-distil-whisper-large-v3"
        / "snapshots"
        / "abc123"
    )
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    assert model_is_cached("distil-large-v3", "en", models_dir=tmp_path) is True
    # The wrong (non-distil) layout must not be treated as a hit
    assert model_is_cached("distil-large-v3", "en", models_dir=tmp_path / "nope") is False
