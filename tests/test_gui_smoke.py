"""Optional GUI smoke test: instantiate the app, pump events, tear down.

Skipped automatically when no display or ttkbootstrap is unavailable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

try:
    import ttkbootstrap  # noqa: F401
    _HAVE_TTKB = True
except ImportError:
    _HAVE_TTKB = False

_HAVE_DISPLAY = bool(os.environ.get("DISPLAY")) or sys.platform == "win32"

pytestmark = pytest.mark.skipif(
    not (_HAVE_DISPLAY and _HAVE_TTKB),
    reason="requires a display and ttkbootstrap",
)


def test_app_constructs_and_pumps_events(tmp_path, monkeypatch):
    import json

    import config as cfg_mod
    from main import TranscriberApp

    # Hermetic: keep config/log writes out of the user profile, and use the
    # tiny model (cached by the e2e run) so preload does not start a large
    # download in the 3s pump window.
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        json.dumps({"model_size": "tiny", "language": "en"}), encoding="utf-8"
    )
    monkeypatch.setattr(cfg_mod, "config_dir", lambda: cfg_dir)

    app = TranscriberApp()
    errors: list[BaseException] = []

    # Surface exceptions raised inside Tk callbacks (after-jobs, bindings)
    def _report(exc_type, exc_value, tb):
        errors.append(exc_value)

    app._app.report_callback_exception = _report

    deadline = time.monotonic() + 3.0
    statuses: list[str] = []
    try:
        while time.monotonic() < deadline:
            app._app.update()
            statuses.append(app._status_var.get())
            time.sleep(0.02)
    finally:
        try:
            app._app.destroy()
        except Exception:
            pass

    assert not errors, f"exceptions in Tk callbacks: {errors}"
    # The deferred _init_backend preload should have updated the status bar
    # ("Loading model..." / "Model ready." / a load-failure message) or, at
    # minimum, the app stayed alive showing its initial status.
    assert statuses, "event loop never pumped"
    assert any(s for s in statuses), "status bar never had text"
