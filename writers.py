"""Pure-stdlib output writers for transcription segments."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from engine import Segment


def _fmt_ts(seconds: float, separator: str) -> str:
    """HH:MM:SS{sep}mmm — separator is '.' for VTT, ',' for SRT."""
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def _cue_text(text: str) -> str:
    """Cue payload for VTT/SRT: blank lines terminate a cue block, so drop them.

    Multi-line text stays multi-line; whisper output never contains newlines,
    this only guards against pathological segment text corrupting the file.
    """
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(lines)


def write_txt(segments: Sequence[Segment], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip())
            f.write("\n")


def write_vtt(segments: Sequence[Segment], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ts(seg.start, '.')} --> {_fmt_ts(seg.end, '.')}\n")
            text = _cue_text(seg.text)
            # Empty payload: one blank line ends the cue; two would leave a stray blank block.
            f.write(f"{text}\n\n" if text else "\n")


def write_srt(segments: Sequence[Segment], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_ts(seg.start, ',')} --> {_fmt_ts(seg.end, ',')}\n")
            text = _cue_text(seg.text)
            f.write(f"{text}\n\n" if text else "\n")


def write_md(segments: Sequence[Segment], path: Path) -> None:
    """Bold 'start - end' timestamp header then text."""
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"**{_fmt_ts(seg.start, '.')} - {_fmt_ts(seg.end, '.')}**:")
            f.write(seg.text + "\n")


def write_json(segments: Sequence[Segment], path: Path) -> None:
    data = {
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
            }
            for seg in segments
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


FORMAT_WRITERS: dict[str, Callable[[Sequence[Segment], Path], None]] = {
    "txt": write_txt,
    "vtt": write_vtt,
    "srt": write_srt,
    "md": write_md,
    "json": write_json,
}
