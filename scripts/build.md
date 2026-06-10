# Building TranscriptionHackery

## Prerequisites

- Python 3.8 or later
- Windows (the output is a Windows .exe)

## Build Steps

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt pyinstaller
   ```

3. Build the executable:
   ```
   pyinstaller main.spec
   ```

4. The built application will be in `dist/TranscriptionHackery/`.

5. To distribute, zip the entire `dist/TranscriptionHackery/` folder.

## Notes

- **Models directory**: Whisper models download on first run to `%LOCALAPPDATA%\TranscriptionHackery\models` and are cached there for subsequent runs.
- **FFmpeg**: Optional. If `ffmpeg.exe` is present in the build directory at build time, it will be bundled in the output directory. Otherwise, faster-whisper uses PyAV for decoding (included in dependencies).
