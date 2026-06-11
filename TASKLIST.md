# TranscriptionHackery — Execution Tasklist
Priorities: **performance first**, then looks, then
long-term updatability.

Effort: S = <2h, M = half/full day, L = multi-day.

---

## Phase 0 — Repo hygiene & safety net (do first; everything else builds on this)

- [x] **Fix `.gitignore` and untrack junk** (S, High) — stop tracking `.venv/`, `TranscriptionLibs/`,
      `build/`, `dist/`, `__pycache__/`, `output/`, test media (`*.mp4`, `tst1.md`),
      `models/*.pt`, `ffmpeg.exe`. `git rm --cached` them.
- [x] **Purge big binaries from history** (M, High) — RESOLVED AS MOOT: `git ls-files` proved the binaries were never tracked on this branch; no history rewrite needed. — repo has GB-class blobs (model, exes, venvs)
      in only ~9 commits. Use `git filter-repo` (or fresh re-init) **now**, before more history
      accrues. Clone size drops from GBs to ~MBs.
- [x] **Rename `# Left To Do's.md` → `TODO.md`** (S, Low) — the `#` filename breaks tooling/grep.
- [x] **Fix `requirements.txt`** (S, High) — remove `whisper` (it's an unrelated PyPI package;
      `openai-whisper` is the real one). Pin versions. (Superseded by Phase 1 engine swap, but
      fix immediately so a fresh install isn't broken.)
- [x] **Baseline benchmark** (S, High) — script that transcribes `tst1.mp4` and records wall-clock
      time + peak VRAM. Every perf change in Phase 1 gets measured against this.

## Phase 1 — Performance core (the big wins)

- [x] **Swap engine to `faster-whisper` (CTranslate2)** (M, High) — ~2–4x faster than
      openai-whisper, `compute_type="float16"` on CUDA / `int8` on CPU, built-in VAD
      (`vad_filter=True`) skips silence for another 20–40% on real-world media.
      **Bonus: removes the torch dependency entirely** → install/bundle shrinks by ~2GB and
      the CUDA-version-matching problem mostly disappears.
- [x] **Define a `TranscriptionBackend` interface first** (M, High) — extract engine code from
      `WhisperTranscriberApp` into `engine.py` behind a small interface
      (`transcribe(path, opts) -> segments`). The faster-whisper swap is then an implementation
      detail, and the next engine (distil/turbo/whatever 2027 brings) slots in without touching UI.
- [x] **Replace ThreadPoolExecutor with ONE persistent worker thread** (M, High) — parallel
      files on a single GPU just contend; sequential is as fast and makes cancellation/progress
      sane. Remove the executor-recreation on slider change and the Max Workers slider
      (or repurpose as CPU-threads setting for the CPU path).
- [ ] **Pipeline I/O with inference** (M, Med) — while the GPU transcribes file N, decode/prepare
      file N+1 (ffmpeg/audio load) in a second thread. True overlap, no GPU contention.
- [x] **Fast startup** (M, High) — show the window in <500ms: remove the blocking startup
      messagebox; move heavy import + GPU detection to a background thread after the window
      renders, with an inline "Initializing…" status; pre-load the model in the background so the
      first job doesn't stall.
- [x] **Model management** (M, Med) — model-size dropdown (tiny→large-v3, default
      `small`/`distil-small.en` on GPU), use `.en` variants when language is English; store models
      in `%LOCALAPPDATA%\TranscriptionHackery\models` (not the exe bundle, not git) with a
      download-progress bar.
- [ ] **Re-run benchmark, record results in README** (S, High).

## Phase 2 — Make it pretty (panels conflicted: polish ttkbootstrap vs. migrate to CustomTkinter — verdict: stay on ttkbootstrap; it's already in place, has 25+ themes incl. dark, and styles Treeview, which CustomTkinter lacks. Revisit only if the polish pass disappoints.)

- [x] **File list → `ttk.Treeview` with per-file status** (M, High) — columns: name, duration,
      status (pending/running/done/failed), per-file progress. Replaces the ScrolledText dump. Right-click to remove.
- [x] **Kill all messagebox spam → inline status bar** (S, High) — no startup popup, no per-error
      modals; bottom status frame + batch summary when done ("12 ok, 2 failed — see log").
- [x] **Consistent ttkbootstrap styling** (S, Med) — replace raw `tk.Checkbutton`s with
      `ttk.Checkbutton` in a "Output Formats" LabelFrame; group settings into labeled frames;
      use bootstyle accents (primary/success) on buttons.
- [x] **Dark/light theme toggle, persisted** (S, Med) — `style.theme_use("darkly"/"yeti")` via a
      menu or toggle; save choice to config.
- [x] **Resizable window with proper grid weights** (S, Med) — Treeview grows, buttons anchor.
- [x] **Per-segment progress within a file** (M, Med) — faster-whisper yields segments as a
      generator, so a real progress bar per file (segment_end / duration) is nearly free.
- [ ] **Drag-and-drop files onto the list** (M, Low) — via `tkinterdnd2`; nice-to-have, do last.

## Phase 3 — Robustness & correctness

- [x] **Delete `install_cuda_drivers()`** (S, High) — dead code running `sudo apt-get` in a
      Windows app.
- [x] **All UI updates via `root.after()` only** (S, High) — no messagebox/widget calls from
      worker threads (current code does both; it's a latent crash).
- [x] **Real cancellation** (M, Med) — single-worker design checks `stop_event` between files;
      with faster-whisper's segment generator, also check between segments → Stop responds
      in seconds, not "after the file finishes".
- [x] **`encoding="utf-8"` on every file write** (S, High) — current txt writes mojibake on cp1252.
- [x] **Logging to file** (M, Med) — `logging` → `%LOCALAPPDATA%\...\transcription.log`
      (per-file result, timing, device); replaces `print`. Final summary dialog links to it.
- [x] **Input validation** (S, Med) — disable Transcribe when no files selected; `shutil.which`
      fallback for ffmpeg; don't `sys.exit(1)` after the window is half-built.

## Phase 4 — Packaging & distribution

- [x] **One canonical PyInstaller spec, one-dir mode** (M, High) — delete the duplicate spec;
      one-file mode re-extracts ~GBs to temp on every launch. One-dir = near-instant start.
      Disable UPX (slows torch-class binaries and breaks things). With faster-whisper the
      bundle is dramatically smaller anyway.
- [x] **Bundle decisions** (S, Med) — keep ffmpeg.exe shipped next to the exe (not in git —
      fetch in build script / GitHub Release asset); models download on demand (Phase 1).
- [x] **Installer: zip + shortcut now, Inno Setup only if distributing publicly** (S, Low).

## Phase 5 — Stay-up-to-date machinery (the "use it for years" part)

- [x] **`pyproject.toml` + `uv` with lockfile** (M, High) — pinned, reproducible env;
      `uv sync` recreates it anywhere; dev-deps separated.
- [x] **Tests that make dep bumps safe** (M, High) — pytest on the pure parts:
      vtt→md conversion, filename sanitizing, output writers, engine wrapper with a mocked
      model, plus one 5-second real-audio smoke test.
- [x] **CI (GitHub Actions)** (M, High) — ruff + pyright + pytest on every push; the smoke test
      on main. This is what lets you bump faster-whisper/ctranslate2 quarterly without fear.
- [x] **Quarterly update ritual** (S, High) — scripted: `uv lock --upgrade` → run tests →
      benchmark → tag release. Calendar reminder; takes <30 min when green.
- [x] **CLAUDE.md + ARCHITECTURE.md** (S, High) — where the backend interface lives, how to
      add an engine, build commands — so future-you (or an AI assistant) onboards in minutes.
- [x] **Semver tags + CHANGELOG.md, real commit messages** (S, Med).

## Phase 6 — Feature backlog (post-stabilization, from the README's original goal)

- [x] **Output options in GUI**: Markdown output (integrate `vttmd.py` as a checkbox),
      output-directory picker, JSON segments output. (S–M, Med)
- [x] **Language dropdown + auto-detect** instead of hardcoded `en`. (S, Med)


---

### Conflicts the panels resolved (for the record)
- **Threads vs processes vs single worker:** single GPU ⇒ single sequential worker +
  I/O prefetch wins on both speed and simplicity.
- **CustomTkinter vs ttkbootstrap:** stay on ttkbootstrap (already integrated, themable,
  has Treeview); migration is a rewrite with no perf payoff.
- **CPU-only vs CUDA bundle:** moot after faster-whisper — ctranslate2 ships CUDA support
  without the 2GB torch wheel.
- **Diarization now vs later:** later, behind a toggle — but bake `speaker` into the data
  model now.
- **History rewrite vs leave it:** rewrite now while the repo is 9 commits old.
