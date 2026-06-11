# Building BatchScribe

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

4. The built application will be in `dist/BatchScribe/`.

5. To distribute, zip the entire `dist/BatchScribe/` folder.

## Notes

- **Models directory**: Whisper models download on first run to `%LOCALAPPDATA%\BatchScribe\models` and are cached there for subsequent runs.
- **FFmpeg**: Not required. Decoding is handled by PyAV (included in dependencies); no external `ffmpeg.exe` is bundled or used.
