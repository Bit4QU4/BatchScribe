# Quarterly Dependency Update Ritual

Follow this checklist to update and validate dependencies:

1. Create a fresh virtual environment in a temporary location
   ```
   python3 -m venv /tmp/th-venv-update
   source /tmp/th-venv-update/bin/activate
   ```

2. Upgrade to latest versions of the primary dependencies
   ```
   pip install -U faster-whisper ttkbootstrap
   ```

3. Run the full test suite to ensure compatibility
   ```
   pip install -r requirements.txt pytest
   pytest tests/ -q
   ```

4. Run the benchmark script on a sample audio file to verify performance
   ```
   python scripts/benchmark.py <sample_audio_file>
   ```

5. Update pinned versions in both requirements.txt and pyproject.toml
   - Run `pip freeze | grep -E '(faster-whisper|ttkbootstrap)'` to get exact versions
   - Update requirements.txt with new pinned versions
   - Update pyproject.toml [project] dependencies section with matching pins

6. Note the new versions and any relevant changes in the release notes

7. Create a git tag for the release
   ```
   git tag -a v0.2.X -m "Release v0.2.X with updated dependencies"
   ```
