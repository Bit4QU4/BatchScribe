"""CUDA-failure fallback behavior of FasterWhisperBackend."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faster_whisper

import engine


class _FakeModel:
    """Stands in for WhisperModel: raises the cuBLAS DLL error on the first
    encode (segment fetch) when built for cuda, works when built for cpu."""

    def __init__(self, model_name, device, compute_type, download_root):
        self.device = device

    def transcribe(self, path, language=None, **kwargs):
        info = SimpleNamespace(duration=2.0, language=language or "en")

        def gen():
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            yield SimpleNamespace(start=0.0, end=1.0, text="hello")
            yield SimpleNamespace(start=1.0, end=2.0, text="world")

        return gen(), info


def test_cuda_library_error_falls_back_to_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeModel)
    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cuda", models_dir=tmp_path, language="en"
    )

    info, seg_iter = backend.transcribe("dummy.m4a", language="en")
    segments = list(seg_iter)

    assert backend.device == "cpu"
    assert info.duration == 2.0
    assert [s.text for s in segments] == ["hello", "world"]


def test_non_cuda_runtime_error_propagates(monkeypatch, tmp_path):
    class _BrokenModel(_FakeModel):
        def transcribe(self, path, language=None, **kwargs):
            raise RuntimeError("Invalid audio stream")

    monkeypatch.setattr(faster_whisper, "WhisperModel", _BrokenModel)
    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cuda", models_dir=tmp_path, language="en"
    )

    try:
        backend.transcribe("dummy.m4a", language="en")
    except RuntimeError as exc:
        assert "Invalid audio stream" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")


def test_is_cuda_library_error_markers():
    assert engine._is_cuda_library_error(
        RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    )
    assert engine._is_cuda_library_error(RuntimeError("cudnn_ops64_9.dll missing"))
    assert not engine._is_cuda_library_error(RuntimeError("Invalid audio stream"))
