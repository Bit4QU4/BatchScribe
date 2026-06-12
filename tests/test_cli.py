"""Tests for cli.py — no model download, no tkinter, no GUI."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import cli
from engine import Segment, TranscriptionBackend, TranscriptionInfo

# ---------------------------------------------------------------------------
# Shared stub backend (mirrors the pattern from test_smoke.py)
# ---------------------------------------------------------------------------

FAKE_SEGMENTS = [
    Segment(start=0.0, end=2.0, text=" Hello CLI."),
    Segment(start=2.0, end=4.0, text=" Goodbye."),
]


class _StubBackend(TranscriptionBackend):
    """In-memory backend; never touches the filesystem for model loading."""

    def __init__(
        self,
        segments: list[Segment] | None = None,
        fail: bool = False,
        duration: float = 5.0,
    ) -> None:
        self._segments = segments if segments is not None else FAKE_SEGMENTS
        self._fail = fail
        self._duration = duration
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def transcribe(
        self,
        path: str,
        language: str | None = "en",
        initial_prompt: str | None = None,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        if not self._loaded:
            self.load()
        if self._fail:
            raise RuntimeError("stub transcription failure")
        info = TranscriptionInfo(duration=self._duration, language=language or "en")
        return info, iter(list(self._segments))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "cpu"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_audio(tmp_path: Path, name: str = "sample.mp3") -> Path:
    """Create a zero-byte placeholder audio file."""
    p = tmp_path / name
    p.write_bytes(b"\x00" * 16)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_writes_txt_and_exits_0(tmp_path: Path, monkeypatch):
    audio = _make_audio(tmp_path)

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend())

    rc = cli.main([str(audio), "--format", "txt", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "sample.txt").exists()
    content = (tmp_path / "sample.txt").read_text(encoding="utf-8")
    assert "Hello CLI." in content


def test_happy_path_multiple_formats(tmp_path: Path, monkeypatch):
    audio = _make_audio(tmp_path)

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend())

    rc = cli.main(
        [str(audio), "--format", "txt", "--format", "srt", "--out", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "sample.txt").exists()
    assert (tmp_path / "sample.srt").exists()


def test_missing_input_file_exits_2_without_backend(tmp_path: Path, monkeypatch):
    backend_created = []

    def _no_backend(**_kw):
        backend_created.append(True)
        return _StubBackend()

    monkeypatch.setattr(cli, "create_backend", _no_backend)

    rc = cli.main([str(tmp_path / "nonexistent.mp3")])
    assert rc == 2
    assert not backend_created, "create_backend must not be called when inputs are missing"


def test_one_failed_file_exits_1(tmp_path: Path, monkeypatch):
    audio = _make_audio(tmp_path)

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend(fail=True))

    rc = cli.main([str(audio), "--format", "txt", "--out", str(tmp_path)])
    assert rc == 1
    # Failure should not produce an output file.
    assert not (tmp_path / "sample.txt").exists()


def test_format_repeat_all_written(tmp_path: Path, monkeypatch):
    audio = _make_audio(tmp_path)

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend())

    rc = cli.main(
        [
            str(audio),
            "--format", "txt",
            "--format", "vtt",
            "--format", "json",
            "--out", str(tmp_path),
        ]
    )
    assert rc == 0
    for ext in ("txt", "vtt", "json"):
        assert (tmp_path / f"sample.{ext}").exists(), f"missing sample.{ext}"


def test_auto_language_maps_to_none(tmp_path: Path, monkeypatch):
    """'auto' must result in language=None passed to create_backend."""
    captured: dict = {}

    def _capture_backend(**kw):
        captured.update(kw)
        return _StubBackend()

    monkeypatch.setattr(cli, "create_backend", _capture_backend)

    audio = _make_audio(tmp_path)
    rc = cli.main([str(audio), "--lang", "auto", "--out", str(tmp_path)])
    assert rc == 0
    assert captured.get("language") is None


def test_no_tkinter_in_import_chain():
    """cli.py and its transitive imports must not pull in tkinter.

    Checked in a subprocess so other tests in the suite cannot pre-load tkinter
    and produce a false pass or false fail.
    """
    import subprocess

    script = (
        "import sys, importlib; "
        f"sys.path.insert(0, {str(_repo)!r}); "
        "import cli; "
        "assert 'tkinter' not in sys.modules, "
        "'tkinter imported via cli import chain'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"tkinter leak detected.\nstderr: {result.stderr}\nstdout: {result.stdout}"
    )


def test_default_format_is_txt(tmp_path: Path, monkeypatch):
    """Omitting --format should write .txt."""
    audio = _make_audio(tmp_path)

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend())

    rc = cli.main([str(audio), "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "sample.txt").exists()


def test_out_dir_created_if_missing(tmp_path: Path, monkeypatch):
    audio = _make_audio(tmp_path)
    new_dir = tmp_path / "nested" / "out"

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend())

    rc = cli.main([str(audio), "--out", str(new_dir)])
    assert rc == 0
    assert new_dir.is_dir()
    assert (new_dir / "sample.txt").exists()


def test_multiple_files_all_ok(tmp_path: Path, monkeypatch):
    a = _make_audio(tmp_path, "a.mp3")
    b = _make_audio(tmp_path, "b.mp3")

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _StubBackend())

    rc = cli.main([str(a), str(b), "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").exists()


def test_mixed_ok_and_fail_exits_1(tmp_path: Path, monkeypatch):
    """Two files: first succeeds, second fails -> exit 1."""
    a = _make_audio(tmp_path, "good.mp3")
    b = _make_audio(tmp_path, "bad.mp3")

    call_count = [0]

    class _ConditionalBackend(_StubBackend):
        def transcribe(self, path, language=None, initial_prompt=None):
            call_count[0] += 1
            if "bad" in path:
                raise RuntimeError("deliberate failure")
            return super().transcribe(path, language, initial_prompt)

    monkeypatch.setattr(cli, "create_backend", lambda **_kw: _ConditionalBackend())

    rc = cli.main([str(a), str(b), "--out", str(tmp_path)])
    assert rc == 1
    assert (tmp_path / "good.txt").exists()
    assert not (tmp_path / "bad.txt").exists()
