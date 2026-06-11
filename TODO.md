# TODO

## High Priority

- **Installer wrapping**: Inno Setup or equivalent for first-run setup (ffmpeg bundling, shortcut creation). Currently: distribute as zip with exe.
- **First-run setup dialog**: Guide user through model download and device detection on first launch. Currently: "Loading model..." appears inline.

## Medium Priority

- **Drag-and-drop file support**: Via tkinterdnd2. Nice-to-have but low friction if added.
- **Per-file output directory override**: UI to set output dir per file instead of batch-wide only.
- **Benchmark results recording**: Collect wall-clock and RTF metrics over time, trend analysis. scripts/benchmark.py exists but doesn't record results.

## Backlog (Post-Stabilization)

- **Diarization**: Speaker identification. Data model already includes `speaker` field in Segment; await upstream faster-whisper support.
- **Pipeline I/O with inference**: Parallel decode/prep of N+1 while GPU transcribes N. Moderate complexity, needs a second thread.
- **Language auto-detect UX improvement**: Show detected language in per-file status after inference.

## Completed (Verified in Code)

- Engine swap to faster-whisper (CTranslate2)
- Single persistent worker thread
- Ttkbootstrap GUI with Treeview file table
- Dark/light theme toggle (persisted)
- Model size + language + output dir + formats (txt/vtt/srt/md/json)
- Config + logs in %LOCALAPPDATA%/BatchScribe
- PyInstaller one-dir packaging
- Tests in tests/
- Output format writers (all 5)
- Proper thread-safe UI dispatch pattern
- UTF-8 encoding on all file writes
