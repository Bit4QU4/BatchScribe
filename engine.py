"""Transcription engine abstractions — no tkinter, no torch."""

from __future__ import annotations

import abc
import itertools
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("transcriber")

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

# Languages offered in the UI; "auto" maps to None (model auto-detect).
LANGUAGE_CHOICES: tuple[str, ...] = ("en", "auto", "es", "fr", "de", "zh", "ja", "pt", "ru")


def default_models_dir() -> Path:
    """Return (and create) the platform-appropriate model cache directory."""
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        base = Path(localappdata) / "BatchScribe" / "models"
    else:
        base = Path.home() / ".local" / "share" / "BatchScribe" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _add_nvidia_dll_dirs() -> None:
    """CTranslate2 loads cuBLAS/cuDNN by DLL name; the pip nvidia-* wheels
    install them under site-packages/nvidia/*/bin, outside the Windows DLL
    search path, so expose those directories explicitly."""
    if sys.platform != "win32":
        return
    import site
    import sysconfig

    candidates: set[Path] = set()
    try:
        candidates.update(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.add(Path(site.getusersitepackages()))
    except Exception:
        pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.add(Path(purelib))

    for root in candidates:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            bin_str = str(bin_dir)
            # CTranslate2 resolves CUDA DLLs with plain LoadLibrary, which
            # searches PATH but ignores add_dll_directory registrations, so
            # both are needed.
            if bin_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(bin_str)
            except OSError:
                pass


def _is_cuda_library_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("cublas", "cudnn", "cudart", "cuda"))


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
        initial_prompt: str | None = None,
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
        initial_prompt: str | None = None,
        strict_vad: bool = False,
        batched: bool = False,
    ) -> None:
        self._model_size = model_size
        self._device_arg = device
        self._models_dir = models_dir or default_models_dir()
        self._language = language
        # Empty string is indistinguishable from no hint; normalize to None so
        # faster-whisper does not bias from an empty context window.
        self._initial_prompt: str | None = initial_prompt or None
        self._strict_vad = strict_vad
        self._batched = batched
        self._model = None
        self._pipeline = None
        self._resolved_device: str = "cpu"

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
        from faster_whisper import WhisperModel

        self._resolved_device = self._resolve_device()
        if self._resolved_device == "cuda":
            _add_nvidia_dll_dirs()
            # Benchmarked 2026-06-11 on the reference GPU: float16 beats
            # int8_float16 by ~10% on the small model, so it stays the
            # default; int8 is the fallback for GPUs that reject fp16.
            compute_types = ("float16", "int8")
        else:
            compute_types = ("int8",)

        last_exc: Exception | None = None
        for compute_type in compute_types:
            try:
                self._model = WhisperModel(
                    self._effective_model_name(self._language),
                    device=self._resolved_device,
                    compute_type=compute_type,
                    download_root=str(self._models_dir),
                )
                break
            except ValueError as exc:
                logger.warning("compute_type %s unavailable (%s)", compute_type, exc)
                last_exc = exc
        if self._model is None:
            raise last_exc if last_exc else RuntimeError("model load failed")

        # Benchmarked 2026-06-11: ~2.5x throughput over the sequential path on
        # the reference GPU. CPU stays sequential; batching gains are GPU-bound.
        if self._batched and self._resolved_device == "cuda":
            from faster_whisper import BatchedInferencePipeline

            self._pipeline = BatchedInferencePipeline(model=self._model)
        else:
            self._pipeline = None

    def _start_transcription(
        self, path: str, language: str | None, initial_prompt: str | None
    ):
        """Prefetch the first segment so CUDA runtime failures (which surface
        on the first encode, not at model load) are raised here."""
        kwargs: dict = {
            "language": language,
            "vad_filter": True,
            "initial_prompt": initial_prompt,
        }
        if self._strict_vad:
            kwargs["vad_parameters"] = dict(min_silence_duration_ms=500, speech_pad_ms=200)
            kwargs["no_speech_threshold"] = 0.5
            kwargs["condition_on_previous_text"] = False
        if self._pipeline is not None:
            kwargs["batch_size"] = 16
            fw_segments, info = self._pipeline.transcribe(path, **kwargs)
        else:
            fw_segments, info = self._model.transcribe(path, **kwargs)
        seg_iter = iter(fw_segments)
        first = next(seg_iter, None)
        return info, first, seg_iter

    def transcribe(
        self,
        path: str,
        language: str | None = "en",
        initial_prompt: str | None = None,
    ) -> tuple[TranscriptionInfo, Iterator[Segment]]:
        if self._model is None:
            self.load()

        # Call-site prompt takes precedence; fall back to the constructor-level
        # hint. worker.py cannot pass per-job prompts (TranscriptionJob has no
        # prompt field and worker.py is not modified), so the constructor path
        # is the only route for GUI-configured hints.
        effective_prompt = initial_prompt or self._initial_prompt

        try:
            info, first, rest = self._start_transcription(path, language, effective_prompt)
        except RuntimeError as exc:
            if self._resolved_device != "cuda" or not _is_cuda_library_error(exc):
                raise
            logger.warning("CUDA runtime unavailable (%s); falling back to CPU", exc)
            self._model = None
            self._pipeline = None
            self._device_arg = "cpu"
            self.load()
            info, first, rest = self._start_transcription(path, language, effective_prompt)

        ti = TranscriptionInfo(
            duration=info.duration,
            language=info.language,
        )

        fw_segments = itertools.chain([first], rest) if first is not None else iter(())

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
    initial_prompt: str | None = None,
    strict_vad: bool = False,
    batched: bool = False,
) -> TranscriptionBackend:
    return FasterWhisperBackend(
        model_size=model_size,
        device=device,
        language=language,
        initial_prompt=initial_prompt,
        strict_vad=strict_vad,
        batched=batched,
    )
