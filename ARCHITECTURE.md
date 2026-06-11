# Architecture

## Module Layout

The codebase is flat (no packages). All modules import directly from the repo root.

### main.py - GUI (ttkbootstrap)

- **TranscriberApp**: main window class
- **Toolbar**: Add/Remove/Clear files, Transcribe/Stop buttons, dark/light toggle
- **File table**: ttk.Treeview with columns (file name, status, progress %)
- **Settings panel**: model size, language, output directory, output format checkboxes
- **Status bar**: real-time status and batch summary
- **Callbacks**: dispatches all worker callbacks to the Tk main thread via `root.after()`

Pure helper functions (tested in `tests/test_app_logic.py`):
- `format_elapsed()` - seconds to "Xm Ys" string
- `language_to_param()` - UI choice to engine param (None = auto-detect)
- `build_jobs()` - construct TranscriptionJob list from UI selections

### engine.py - Transcription abstraction (no tkinter, no torch)

**TranscriptionBackend** (abstract base class):
- `load()` - idempotent model load
- `transcribe(path, language)` -> (TranscriptionInfo, Iterator[Segment]) - yields segments lazily
- `is_loaded` (property) - whether model is in memory
- `device` (property) - "cuda" or "cpu"

**FasterWhisperBackend** (implementation):
- Uses faster-whisper 1.x and CTranslate2
- Resolves device at load time (auto-detect CUDA, fallback to CPU)
- Compute type: float16 on CUDA, int8 on CPU
- VAD filter enabled by default (skips silence)
- Downloads models to `%LOCALAPPDATA%/TranscriptionHackery/models` (Windows) or `~/.local/share/TranscriptionHackery/models` (Linux)

**Segment** (dataclass): `start`, `end`, `text`, optional `speaker`

**TranscriptionInfo** (dataclass): `duration`, `language`

**create_backend()** factory: returns FasterWhisperBackend (extendable to other engines)

### worker.py - Single persistent worker thread

**TranscriptionWorker**:
- One daemon thread per app lifetime; never touches widgets
- **Public API**:
  - `submit(jobs: list[TranscriptionJob])` - queue batch
  - `stop()` - set stop_event; halts between files and mid-segment
  - `preload()` - queue a background model load
  - `is_busy` (property) - whether processing
- **Dispatch rule**: all callbacks routed through `dispatch()` callable (in main, this is `root.after()`)
- **Callbacks** (WorkerCallbacks):
  - `on_file_start(path)` - file processing begins
  - `on_segment_progress(path, fraction)` - per-segment progress (0..1)
  - `on_file_done(FileResult)` - file complete (ok, message, elapsed)
  - `on_batch_done(ok_count, fail_count, elapsed)` - all files done
  - `on_status(text)` - informational status updates

### writers.py - Output format writers

Pure functions, all write UTF-8 encoded files:
- `write_txt()` - one line per segment
- `write_vtt()` - WebVTT with HH:MM:SS.mmm timestamps
- `write_srt()` - SRT with HH:MM:SS,mmm timestamps
- `write_md()` - Markdown with bold **HH:MM:SS.mmm - HH:MM:SS.mmm**: timestamp headers
- `write_json()` - segments array with start, end, text, speaker

**FORMAT_WRITERS** dict maps format names to writer functions.

### config.py - Persistence and logging (stdlib only)

**AppConfig** (dataclass): theme, model_size, language, formats (list), output_dir

Functions:
- `config_dir()` - returns (and creates) platform-appropriate directory
- `load_config()` -> AppConfig
- `save_config(cfg)` - atomic write (temp file + rename)
- `setup_logging()` -> log_path - rotating file handler (1 MB, 2 backups) + stderr handler (warnings+)

## Design Patterns

### Thread-UI Dispatch Rule

**No worker thread ever touches a tkinter widget directly.** All callbacks use the dispatch pattern:

```python
def _cb(self, fn: Callable[[], None]) -> None:
    self._dispatch(fn)  # In main: lambda fn: root.after(0, fn)
```

This ensures UI updates happen on the Tk main thread.

### Backend Abstraction

The `TranscriptionBackend` interface decouples the engine from the GUI. To add a new engine (e.g., Whisper.cpp, Distil):

1. Subclass `TranscriptionBackend`
2. Implement `load()`, `transcribe()`, `is_loaded`, `device`
3. Register in `create_backend()` (e.g., via a device or model-name check)

The worker and UI remain unchanged.

### Single Worker Thread

Sequential processing on a single GPU is as fast as parallel and avoids contention. The worker:
- Processes one file at a time (no prefetch yet)
- Checks `stop_event` between files and between segments
- Collects output writers' results per-file
- Fires callbacks via dispatch to stay thread-safe

## Testing

- `tests/test_app_logic.py` - pure functions from main.py
- `tests/test_smoke.py` - integration tests for engine, workers, writers, and config (no real model download or GPU required; uses stub backend)

Run:
```bash
python -m pytest tests/ -q
```

Or directly:
```bash
python tests/test_smoke.py
```

## Dependencies

- **faster-whisper** - fast inference wrapper around CTranslate2
- **ttkbootstrap** - themable Tkinter (dark/light, Treeview support)
- **pytest** (dev) - testing

Python 3.10+ (f-strings, type hints, dataclass).
