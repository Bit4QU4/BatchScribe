# Working Notes for AI Assistants

## Quick Start

Run tests: `python -m pytest tests/ -q` (or `python tests/test_smoke.py` standalone)

## Layout

- Flat module structure; no packages. All imports from repo root.
- **main.py** = GUI only (no engine logic).
- **engine.py** = TranscriptionBackend ABC + FasterWhisperBackend impl.
- **worker.py** = single daemon thread; dispatch pattern for UI callbacks.
- **writers.py** = stateless output format functions.
- **config.py** = persistence + logging (stdlib only).

## Key Rules

1. **No widget access from worker**: all callbacks routed via dispatch (in main: `root.after()`). Worker is ttkbootstrap-agnostic.

2. **Flat = fast**: no package imports, no circular deps. Modules can be tested/iterated independently.

3. **Comments are DRY**: explain "why", not "what". Code is readable; avoid emoji.

4. **Deps pinned in two places**:
   - `requirements.txt` (pip install)
   - `pyproject.toml` (build, CI, lock files)
   - Keep in sync.

5. **Target Windows but keep modules CPU-Linux-testable**: no `%LOCALAPPDATA%` in engine/worker/writers, only in config. Workers stay pure.

6. **Encoding = UTF-8 everywhere**: all file writes specify `encoding="utf-8"`.

7. **PyInstaller one-dir via main.spec**: UPX disabled (no benefit after faster-whisper swap).

## Extension Points

- **New engine**: subclass TranscriptionBackend, implement load/transcribe/is_loaded/device, register in create_backend().
- **New writer**: add function to writers.py, register in FORMAT_WRITERS dict.
- **New config field**: add to AppConfig dataclass, handled by load_config() and save_config() automatically.

## Testing Notes

- Pure functions (format_elapsed, language_to_param) tested in test_app_logic.py.
- Integration tests (engine, worker, writers) in test_smoke.py with stub backend (no model download).
- Workers use stub backend + synchronous dispatch (lambda fn: fn()) for determinism.
- Config tests patch config_dir to avoid touching user's real config.

## Build & Package

PyInstaller: `pyinstaller main.spec` (see scripts/build.md for full steps).

One-dir mode recommended (not one-file; faster startup).
