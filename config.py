"""App configuration persistence and logging setup — stdlib only."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


def config_dir() -> Path:
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        base = Path(localappdata) / "BatchScribe"
    else:
        base = Path.home() / ".local" / "share" / "BatchScribe"
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class AppConfig:
    theme: str = "darkly"
    model_size: str = "small"
    language: str = "en"
    formats: list[str] = field(default_factory=lambda: ["txt"])
    output_dir: str | None = None
    initial_prompt: str = ""
    strict_vad: bool = False


_CONFIG_FILE = "config.json"


def load_config() -> AppConfig:
    path = config_dir() / _CONFIG_FILE
    if not path.exists():
        return AppConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppConfig()
    if not isinstance(raw, dict):
        return AppConfig()

    cfg = AppConfig()
    # Only apply known keys with expected types; unknown/wrong-typed keys keep defaults.
    for key, val in raw.items():
        if key == "output_dir":
            if val is None or isinstance(val, str):
                cfg.output_dir = val
        elif key == "formats":
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                cfg.formats = val
        elif key in ("theme", "model_size", "language", "initial_prompt"):
            if isinstance(val, str):
                setattr(cfg, key, val)
        elif key == "strict_vad":
            if isinstance(val, bool):
                cfg.strict_vad = val
    return cfg


def save_config(cfg: AppConfig) -> None:
    target = config_dir() / _CONFIG_FILE
    data = json.dumps(asdict(cfg), ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_logging_configured = False


def setup_logging() -> Path:
    """Configure rotating file + stderr handlers. Idempotent."""
    global _logging_configured
    log_path = config_dir() / "transcription.log"

    root_logger = logging.getLogger("transcriber")
    if _logging_configured:
        return log_path

    root_logger.setLevel(logging.DEBUG)

    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=1 * 1024 * 1024,  # 1 MB
        backupCount=2,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(sh)

    _logging_configured = True
    return log_path
