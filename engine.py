"""Transcription engine abstractions — no tkinter, no torch."""

from __future__ import annotations

import abc
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# faster-whisper 1.x exposes these sizes; "turbo" is an alias for "large-v3-turbo"
# in some builds but is not a canonical HF model name — omit to avoid silent 404s.
MODEL_SIZES: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "distil-large-v3",
)

# Sizes that have a dedicated .en variant on HuggingFace
_EN_VARIANT_SIZES: frozenset[str] = frozenset({"tiny", "base", "small", "medium"})


def default_models_dir() -> Path:
    """Return (and create) the platform-appropriate model cache directory."""
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        base = Path(localappdata) / "TranscriptionHackery" / "models"
    else:
        base = Path.home() / ".local" / "share" / "TranscriptionHackery" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(slots=True)
class TranscriptionInfo:
    duration: float
    language: str


class TranscriptionBackend(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None:
        """Idempotent: download/load model into memory."""

    @abc.abstractmethod
    def transcribe(
        self,
        path: str,
        language: str | None = "en",
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        """Return metadata and a lazy segment iterator."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abc.abstractmethod
    def device(self) -> str:
        """'cuda' or 'cpu'."""


class FasterWhisperBackend(TranscriptionBackend):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        models_dir: Path | None = None,
        language: str | None = "en",
    ) -> None:
        self._model_size = model_size
        self._device_arg = device
        self._models_dir = models_dir or default_models_dir()
        self._language = language
        self._model = None
        self._resolved_device: str = "cpu"  # filled in by load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self._device_arg == "auto":
            try:
                import ctranslate2  # lazy; raises ImportError on bare CPU envs
                if ctranslate2.get_cuda_device_count() > 0:
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return self._device_arg

    def _effective_model_name(self, language: str | None) -> str:
        if language == "en" and self._model_size in _EN_VARIANT_SIZES:
            return f"{self._model_size}.en"
        return self._model_size

    # ------------------------------------------------------------------
    # TranscriptionBackend interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # lazy import

        self._resolved_device = self._resolve_device()
        compute_type = "float16" if self._resolved_device == "cuda" else "int8"
        self._model = WhisperModel(
            self._effective_model_name(self._language),
            device=self._resolved_device,
            compute_type=compute_type,
            download_root=str(self._models_dir),
        )

    def transcribe(
        self,
        path: str,
        language: str | None = "en",
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        if self._model is None:
            self.load()

        fw_segments, info = self._model.transcribe(
            path,
            language=language,
            vad_filter=True,
        )

        ti = TranscriptionInfo(
            duration=info.duration,
            language=info.language,
        )

        def _iter() -> Iterator[Segment]:
            for seg in fw_segments:
                yield Segment(start=seg.start, end=seg.end, text=seg.text)

        return ti, _iter()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._resolved_device


def create_backend(
    model_size: str = "small",
    device: str = "auto",
    language: str | None = "en",
) -> TranscriptionBackend:
    return FasterWhisperBackend(model_size=model_size, device=device, language=language)
