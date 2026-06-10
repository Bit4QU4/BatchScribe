# TranscriptionHackery

A fast, single-threaded batch transcription GUI using faster-whisper (CTranslate2).

Transcribe audio and video files to text in multiple formats (txt, vtt, srt, md, json) with a simple Tkinter UI. Supports GPU acceleration (NVIDIA CUDA 12 + cuDNN 9) or CPU-only operation.

## Quick Start

### Prerequisites

- Python 3.10+
- Windows or Linux

### Installation & Run

```bash
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Linux
pip install -r requirements.txt
python main.py
```

The app opens with default settings (model: small, language: English, output format: txt).

## Features

- **Fast inference**: faster-whisper engine (2-4x faster than openai-whisper)
- **Single GPU pass**: sequential file processing, no contention
- **Multiple output formats**: txt, vtt, srt, md (with timestamps), json
- **GUI controls**: model size, language, output directory, output formats
- **Theme toggle**: dark/light mode with ttkbootstrap
- **File table**: per-file status (Queued/Running/Done/Failed) with progress
- **Cancellable**: Stop button halts processing between files and mid-segment
- **Config persistence**: theme, model size, language, formats, output dir saved to disk
- **Logging**: all operations (transcription, timing, device) logged to file

## GPU Notes

### NVIDIA (CUDA)

Requires CUDA 12 and cuDNN 9. CTranslate2 auto-detects CUDA availability; if present, uses float16 compute for speed and memory efficiency.

```bash
pip install -r requirements.txt
python main.py
```

If CUDA is unavailable, the app falls back to CPU with int8 quantization.

### CPU

Works out of the box. Int8 quantization is automatically applied for reasonable memory usage. No CUDA, CUDA Toolkit, or torch installation needed.

## Models & Cache

Models download on first use to:
- Windows: `%LOCALAPPDATA%\TranscriptionHackery\models`
- Linux: `~/.local/share/TranscriptionHackery/models`

Models are cached indefinitely; re-running with the same model is instant (except first load per session).

## Settings

Configured via the GUI; settings persist in:
- Windows: `%LOCALAPPDATA%\TranscriptionHackery\config.json`
- Linux: `~/.local/share/TranscriptionHackery/config.json`

Logs are at:
- Windows: `%LOCALAPPDATA%\TranscriptionHackery\transcription.log`
- Linux: `~/.local/share/TranscriptionHackery/transcription.log`

## Helper Scripts

### vttmd.py

Convert VTT transcript to Markdown with timestamps:

```bash
python vttmd.py  # prompts for directory
```

For each `.vtt` file in the directory, writes a `.md` file with bold timestamp headers.

### mp3titler.py

Rename MP3 files by their ID3 tag title:

```bash
python mp3titler.py <directory>
```

Reads ID3 tags and renames files sanitized titles. Handles duplicates by appending `_2`, `_3`, etc.

## Building (PyInstaller)

See `scripts/build.md` for detailed instructions.

One-dir mode (not one-file) is recommended:

```bash
pip install -r requirements.txt pyinstaller
pyinstaller main.spec
```

Output: `dist/TranscriptionHackery/` (ready to zip and distribute).

Optional: include `ffmpeg.exe` in the build directory; it will be bundled. Otherwise, faster-whisper uses PyAV for fallback decoding.

## Benchmarking

```bash
python scripts/benchmark.py <media-file> [--model small] [--runs 1]
```

Records wall-clock time and realtime factor (audio_duration / wall_time). Example: 60s of audio transcribed in 30s = 2.0x realtime factor.

## Architecture

See `ARCHITECTURE.md` for the module layout, backend abstraction, and how to extend the engine.

## License

See LICENSE (if present).
