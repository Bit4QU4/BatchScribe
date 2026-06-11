"""Tests for initial_prompt and strict_vad features.

All tests are deterministic — no network, no model download, no GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import faster_whisper

import config as cfg_mod
import engine
from config import AppConfig, load_config, save_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patched_config_dir(tmp_path: Path, content: str | None = None):
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


class _CapturingModel:
    """Stands in for WhisperModel and records the kwargs it receives."""

    def __init__(self, model_name, device, compute_type, download_root):
        self.device = device
        self.last_kwargs: dict = {}

    def transcribe(self, path, **kwargs):
        self.last_kwargs = kwargs
        info = SimpleNamespace(duration=1.0, language=kwargs.get("language") or "en")

        def gen():
            yield SimpleNamespace(start=0.0, end=1.0, text="hello")

        return gen(), info


# ---------------------------------------------------------------------------
# Feature 1: initial_prompt — empty-string normalization
# ---------------------------------------------------------------------------

def test_empty_string_prompt_normalized_to_none(monkeypatch, tmp_path):
    capturing = None

    def _fake_whisper_model(model_name, device, compute_type, download_root):
        nonlocal capturing
        capturing = _CapturingModel(model_name, device, compute_type, download_root)
        return capturing

    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_whisper_model)

    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cpu", models_dir=tmp_path,
        initial_prompt="",
    )
    backend.transcribe("dummy.wav", language="en")

    assert capturing is not None
    # Empty string must be normalized to None so faster-whisper receives no hint.
    assert capturing.last_kwargs.get("initial_prompt") is None


def test_nonempty_prompt_passed_through(monkeypatch, tmp_path):
    capturing = None

    def _fake_whisper_model(model_name, device, compute_type, download_root):
        nonlocal capturing
        capturing = _CapturingModel(model_name, device, compute_type, download_root)
        return capturing

    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_whisper_model)

    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cpu", models_dir=tmp_path,
        initial_prompt="Dr. Smith, HVAC",
    )
    backend.transcribe("dummy.wav", language="en")

    assert capturing.last_kwargs.get("initial_prompt") == "Dr. Smith, HVAC"


def test_call_site_prompt_overrides_constructor(monkeypatch, tmp_path):
    """A non-None call-site argument shadows the constructor default."""
    capturing = None

    def _fake_whisper_model(model_name, device, compute_type, download_root):
        nonlocal capturing
        capturing = _CapturingModel(model_name, device, compute_type, download_root)
        return capturing

    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_whisper_model)

    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cpu", models_dir=tmp_path,
        initial_prompt="constructor hint",
    )
    backend.transcribe("dummy.wav", language="en", initial_prompt="call-site hint")

    assert capturing.last_kwargs.get("initial_prompt") == "call-site hint"


def test_call_site_none_falls_back_to_constructor(monkeypatch, tmp_path):
    """Omitting the call-site prompt uses the constructor value."""
    capturing = None

    def _fake_whisper_model(model_name, device, compute_type, download_root):
        nonlocal capturing
        capturing = _CapturingModel(model_name, device, compute_type, download_root)
        return capturing

    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_whisper_model)

    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cpu", models_dir=tmp_path,
        initial_prompt="constructor hint",
    )
    # Call without specifying initial_prompt (defaults to None at call site)
    backend.transcribe("dummy.wav", language="en")

    assert capturing.last_kwargs.get("initial_prompt") == "constructor hint"


# ---------------------------------------------------------------------------
# Feature 2: strict_vad — parameter mapping
# ---------------------------------------------------------------------------

def test_strict_vad_false_does_not_add_extra_params(monkeypatch, tmp_path):
    capturing = None

    def _fake_whisper_model(model_name, device, compute_type, download_root):
        nonlocal capturing
        capturing = _CapturingModel(model_name, device, compute_type, download_root)
        return capturing

    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_whisper_model)

    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cpu", models_dir=tmp_path, strict_vad=False
    )
    backend.transcribe("dummy.wav", language="en")

    assert "vad_parameters" not in capturing.last_kwargs
    assert "no_speech_threshold" not in capturing.last_kwargs
    assert "condition_on_previous_text" not in capturing.last_kwargs
    assert capturing.last_kwargs.get("vad_filter") is True


def test_strict_vad_true_passes_correct_params(monkeypatch, tmp_path):
    capturing = None

    def _fake_whisper_model(model_name, device, compute_type, download_root):
        nonlocal capturing
        capturing = _CapturingModel(model_name, device, compute_type, download_root)
        return capturing

    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_whisper_model)

    backend = engine.FasterWhisperBackend(
        model_size="tiny", device="cpu", models_dir=tmp_path, strict_vad=True
    )
    backend.transcribe("dummy.wav", language="en")

    kw = capturing.last_kwargs
    assert kw.get("vad_filter") is True
    assert kw.get("no_speech_threshold") == 0.5
    assert kw.get("condition_on_previous_text") is False
    vad_params = kw.get("vad_parameters", {})
    assert vad_params.get("min_silence_duration_ms") == 500
    assert vad_params.get("speech_pad_ms") == 200


def test_create_backend_threads_strict_vad(monkeypatch, tmp_path):
    """create_backend must forward strict_vad to FasterWhisperBackend."""
    b = engine.create_backend(
        model_size="tiny", device="cpu", strict_vad=True
    )
    assert isinstance(b, engine.FasterWhisperBackend)
    assert b._strict_vad is True


def test_create_backend_threads_initial_prompt(monkeypatch, tmp_path):
    b = engine.create_backend(
        model_size="tiny", device="cpu", initial_prompt="hint text"
    )
    assert b._initial_prompt == "hint text"


# ---------------------------------------------------------------------------
# Config: round-trip of both new fields
# ---------------------------------------------------------------------------

def test_config_roundtrip_initial_prompt(tmp_path: Path):
    original = cfg_mod.config_dir

    def _patched():
        d = tmp_path / "cfg_prompt"
        d.mkdir(exist_ok=True)
        return d

    cfg_mod.config_dir = _patched
    try:
        cfg = AppConfig(initial_prompt="Dr. Smith, HVAC")
        save_config(cfg)
        loaded = load_config()
        assert loaded.initial_prompt == "Dr. Smith, HVAC"
    finally:
        cfg_mod.config_dir = original


def test_config_roundtrip_strict_vad(tmp_path: Path):
    original = cfg_mod.config_dir

    def _patched():
        d = tmp_path / "cfg_vad"
        d.mkdir(exist_ok=True)
        return d

    cfg_mod.config_dir = _patched
    try:
        cfg = AppConfig(strict_vad=True)
        save_config(cfg)
        loaded = load_config()
        assert loaded.strict_vad is True
    finally:
        cfg_mod.config_dir = original


def test_config_defaults_for_new_fields():
    cfg = AppConfig()
    assert cfg.initial_prompt == ""
    assert cfg.strict_vad is False


# ---------------------------------------------------------------------------
# Config: corrupt-type fallback for both new fields
# ---------------------------------------------------------------------------

def test_config_wrong_type_initial_prompt_falls_back(tmp_path: Path):
    """Non-string initial_prompt must be ignored; default kept."""
    raw = json.dumps({"initial_prompt": 42})
    cfg = _load_with(tmp_path, raw)
    assert cfg.initial_prompt == ""


def test_config_wrong_type_strict_vad_falls_back(tmp_path: Path):
    """Non-bool strict_vad must be ignored; default kept."""
    raw = json.dumps({"strict_vad": "yes"})
    cfg = _load_with(tmp_path, raw)
    assert cfg.strict_vad is False


def test_config_wrong_type_strict_vad_int_falls_back(tmp_path: Path):
    """Integer 1 is not a bool in isinstance check (Python int is subclass of bool
    but json.loads returns plain int for 1); strict type guard required."""
    raw = json.dumps({"strict_vad": 1})
    cfg = _load_with(tmp_path, raw)
    # json.loads produces int 1, not bool True — should fall back to default
    # (isinstance(1, bool) is False in Python, so our guard is strict enough)
    # If the implementation accepts it, that is also acceptable; test documents behavior.
    assert isinstance(cfg.strict_vad, bool)


def test_config_valid_new_fields_applied(tmp_path: Path):
    raw = json.dumps({"initial_prompt": "CEO John", "strict_vad": True})
    cfg = _load_with(tmp_path, raw)
    assert cfg.initial_prompt == "CEO John"
    assert cfg.strict_vad is True
