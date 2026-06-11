# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## 0.2.0 - 2026-06-11

### Changed

- **Engine swap**: Replaced openai-whisper + torch with faster-whisper (CTranslate2). 2-4x faster inference, removes 2GB torch dependency, cleaner CUDA handling.
- **Single worker thread**: Replaced ThreadPoolExecutor with one persistent daemon thread. Sequential processing avoids GPU contention and simplifies cancellation.
- **GUI overhaul**: Ttkbootstrap Treeview for per-file status (Queued/Running/Done/Failed) and progress. Dark/light theme toggle persisted. Inline status bar replaces messagebox spam.
- **Output formats**: Added md (Markdown with timestamps), json, vtt, srt writers. Txt and formats configurable in GUI.
- **Config & logging**: App settings (theme, model size, language, formats, output dir) persisted to disk. Rotating file logger to %LOCALAPPDATA%\TranscriptionHackery\transcription.log.
- **Model management**: Model size dropdown (tiny-large-v3). Models cache in %LOCALAPPDATA% (or ~/.local/share on Linux), download on demand.
- **Packaging**: PyInstaller one-dir (not one-file); faster startup and easier distribution.

### Added

- Device auto-detection (CUDA if available, fallback to CPU).
- VAD (voice activity detection) filter enabled by default; skips silence for 20-40% faster real-world media.
- Cancel button (stop_event checked between files and segments).
- Tests in tests/ (pure functions in test_app_logic.py; integration in test_smoke.py with stub backend).
- CI skeleton in .github/workflows.
- ARCHITECTURE.md, CLAUDE.md, this CHANGELOG.md.
- MIT license.

### Removed

- Dead code: install_cuda_drivers(), openai-whisper dependency, old torch + CUDA 11.7 instructions.
- ThreadPoolExecutor and Max Workers slider (single worker is better).
- Splash screen messagebox (blocks startup; moved to background preload).
- Bundled ffmpeg.exe and the find_ffmpeg() helper: PyAV (shipped with faster-whisper deps) handles all decoding, and we do not redistribute a standalone FFmpeg binary.

### Fixed

- File write encoding: all output now explicitly UTF-8 (no mojibake on Windows cp1252).
- UI thread safety: all worker callbacks routed via root.after() dispatch pattern.

## 0.1.0 (Earlier)

Initial version with openai-whisper + torch, ThreadPoolExecutor, basic Tkinter UI.
