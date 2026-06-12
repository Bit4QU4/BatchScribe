"""GUI entry point for BatchScribe."""

from __future__ import annotations

import logging
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import ttkbootstrap as ttk
from ttkbootstrap.constants import DISABLED, END, HORIZONTAL, LEFT, NORMAL, RIGHT, VERTICAL
from ttkbootstrap.widgets import ToolTip

from config import AppConfig, load_config, save_config, setup_logging
from engine import (
    LANGUAGE_CHOICES,
    MODEL_SIZES,
    TranscriptionBackend,
    create_backend,
    model_is_cached,
)
from worker import (
    FileResult,
    TranscriptionJob,
    TranscriptionWorker,
    WorkerCallbacks,
)
from writers import FORMAT_WRITERS

logger = logging.getLogger("transcriber")

MEDIA_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".avi", ".mov", ".flv", ".wmv",
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a",
)

_FORMAT_TOOLTIPS: dict[str, str] = {
    "txt": "Plain text, no timestamps.",
    "md": "Markdown with bold timestamp headers.",
    "json": "Structured segments with start/end times, for scripts and tools.",
    "vtt": "WebVTT subtitles, for web video players.",
    "srt": "SubRip subtitles, the most widely supported caption format.",
}


# ---------------------------------------------------------------------------
# Pure helpers (no GUI) — tested in tests/test_app_logic.py
# ---------------------------------------------------------------------------

def format_elapsed(seconds: float) -> str:
    """Return human-readable elapsed string: Xm Ys or Xs."""
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def language_to_param(choice: str) -> str | None:
    """Convert UI language string to engine language parameter (None = auto-detect)."""
    return None if choice == "auto" else choice


def build_jobs(
    paths: list[str],
    formats: list[str],
    output_dir: str | None,
    language: str | None,
) -> list[TranscriptionJob]:
    return [
        TranscriptionJob(path=p, formats=formats, output_dir=output_dir or None, language=language)
        for p in paths
    ]


def filter_media_paths(paths) -> list[str]:
    """Keep only existing files with a known media extension (for drag-drop,
    which can deliver folders and arbitrary file types)."""
    return [
        p for p in paths
        if Path(p).suffix.lower() in MEDIA_EXTENSIONS and Path(p).is_file()
    ]


def batch_progress_text(
    started: int,
    total: int,
    elapsed: float,
    completed_wall_times: list[float],
) -> str:
    """Return the status bar string for a running batch.

    ETA is omitted until at least one file completes, because we have no
    per-file duration for unstarted files — we use mean wall time instead.
    """
    elapsed_str = format_elapsed(elapsed)
    file_part = f"File {started} of {total}"
    if not completed_wall_times:
        return f"{file_part} - {elapsed_str} elapsed"
    remaining = total - len(completed_wall_times)
    if remaining <= 0:
        return f"{file_part} - {elapsed_str} elapsed"
    mean_wall = sum(completed_wall_times) / len(completed_wall_times)
    eta = format_elapsed(mean_wall * remaining)
    return f"{file_part} - {elapsed_str} elapsed, ~{eta} remaining"


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class TranscriberApp:
    def __init__(self) -> None:
        self._cfg: AppConfig = load_config()
        self._log_path: Path = setup_logging()

        try:
            self._app = ttk.Window(themename=self._cfg.theme)
        except Exception:
            # A hand-edited or stale config theme must not brick startup.
            logger.warning("Unknown theme %r; falling back to darkly", self._cfg.theme)
            self._cfg.theme = "darkly"
            self._app = ttk.Window(themename="darkly")
        self._app.title("BatchScribe")
        self._app.minsize(700, 480)
        self._app.resizable(True, True)

        # Tracks full paths; iid == path
        self._file_paths: list[str] = []

        # Current backend params so we can detect when recreation is needed
        self._backend_model: str = self._cfg.model_size
        self._backend_lang: str = self._cfg.language
        self._backend_strict_vad: bool = self._cfg.strict_vad
        self._backend_batched: bool = self._cfg.batched_gpu
        self._backend_initial_prompt: str = self._cfg.initial_prompt
        self._backend: TranscriptionBackend | None = None
        self._worker: TranscriptionWorker | None = None

        self._running = False

        # Batch-level progress counters; reset in _start_transcribe.
        self._batch_total: int = 0
        self._batch_started: int = 0
        self._batch_start_time: float = 0.0
        self._batch_file_wall_times: list[float] = []
        # Wall-clock when the current file's on_file_start fired (for ETA accumulation)
        self._current_file_start_time: float = 0.0

        self._build_ui()
        self._app.protocol("WM_DELETE_WINDOW", self._on_close)

        # Deferred backend init after window renders
        self._app.after(100, self._init_backend)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self._app
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self._build_toolbar(root)
        self._build_file_table(root)
        self._build_settings(root)
        self._build_progress(root)
        self._build_statusbar(root)
        self._enable_dnd()
        self._bind_shortcuts()

    def _bind_shortcuts(self) -> None:
        """Keyboard operation of the full workflow.

        Handlers re-check running state because key bindings bypass the
        disabled state of the toolbar buttons.
        """
        root = self._app
        root.bind("<Control-o>", lambda _e: self._running or self._add_files())
        root.bind("<Control-Return>", lambda _e: self._running or self._start_transcribe())
        root.bind("<Escape>", lambda _e: self._stop() if self._running else None)
        # Plain Delete/BackSpace stay scoped to the file table so text entries
        # keep their editing keys.
        self._tree.bind("<Delete>", lambda _e: self._running or self._remove_selected())
        self._tree.bind("<BackSpace>", lambda _e: self._running or self._remove_selected())

    def _enable_dnd(self) -> None:
        """Register the whole window as a file drop target.

        tkinterdnd2 ships a platform-specific tkdnd binary; treat any failure
        (missing wheel, unsupported platform, broken bundle) as 'no drag-drop'
        rather than a startup error.
        """
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD

            # ttk.Window is a plain Tk subclass; graft the DnD mixin onto the
            # live instance so the mixin's internal self._dnd_* helpers resolve.
            base = self._app.__class__
            self._app.__class__ = type("_DnDWindow", (base, TkinterDnD.DnDWrapper), {})
            self._app.TkdndVersion = TkinterDnD._require(self._app)
            self._app.drop_target_register(DND_FILES)
            self._app.dnd_bind("<<Drop>>", self._on_drop)
        except Exception as exc:
            logger.info("Drag-and-drop unavailable: %s", exc)

    def _build_toolbar(self, parent: tk.Widget) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        self._btn_add = ttk.Button(bar, text="Add Files", command=self._add_files)
        self._btn_add.pack(side=LEFT, padx=2)
        ToolTip(self._btn_add, text="Pick audio/video files to add to the queue (Ctrl+O).")

        self._btn_remove = ttk.Button(bar, text="Remove Selected", command=self._remove_selected)
        self._btn_remove.pack(side=LEFT, padx=2)
        ToolTip(self._btn_remove, text="Remove the highlighted files from the queue (Delete).")

        self._btn_clear = ttk.Button(bar, text="Clear", command=self._clear_files)
        self._btn_clear.pack(side=LEFT, padx=2)
        ToolTip(self._btn_clear, text="Empty the queue. Files on disk are not touched.")

        self._btn_transcribe = ttk.Button(
            bar, text="Transcribe", bootstyle="success", command=self._start_transcribe
        )
        self._btn_transcribe.pack(side=LEFT, padx=(12, 2))
        ToolTip(self._btn_transcribe,
                text="Transcribe every queued file with the current settings (Ctrl+Enter).")

        self._btn_stop = ttk.Button(
            bar, text="Stop", bootstyle="danger", command=self._stop, state=DISABLED
        )
        self._btn_stop.pack(side=LEFT, padx=2)
        ToolTip(self._btn_stop,
                text="Stop after the current segment; finished files are kept (Esc).")

        self._dark_var = tk.BooleanVar(value=self._cfg.theme == "darkly")
        cb_theme = ttk.Checkbutton(
            bar, text="Dark mode", variable=self._dark_var, bootstyle="round-toggle",
            command=self._toggle_theme,
        )
        cb_theme.pack(side=RIGHT, padx=4)
        ToolTip(cb_theme, text="Switch between dark and light themes; remembered between runs.")

    def _build_file_table(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=2)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        cols = ("file", "status", "progress")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="extended")
        self._tree.heading("file", text="File")
        self._tree.heading("status", text="Status")
        self._tree.heading("progress", text="Progress")
        self._tree.column("file", width=380, stretch=True)
        self._tree.column("status", width=90, stretch=False)
        self._tree.column("progress", width=80, stretch=False)
        self._tree.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(frame, orient=VERTICAL, command=self._tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)

    def _build_settings(self, parent: tk.Widget) -> None:
        outer = ttk.Frame(parent)
        outer.grid(row=2, column=0, sticky="ew", padx=6, pady=2)
        outer.columnconfigure(1, weight=1)

        sf = ttk.LabelFrame(outer, text="Settings")
        sf.grid(row=0, column=0, sticky="nsew", padx=(0, 4), ipadx=4, ipady=4)

        ttk.Label(sf, text="Model:").grid(row=0, column=0, sticky="w")
        self._model_var = tk.StringVar(value=self._cfg.model_size)
        model_cb = ttk.Combobox(sf, textvariable=self._model_var, values=list(MODEL_SIZES),
                                state="readonly", width=16)
        model_cb.grid(row=0, column=1, sticky="w", padx=4)
        model_cb.bind("<<ComboboxSelected>>", self._on_model_change)
        ToolTip(
            model_cb,
            text=(
                "Larger models are more accurate but slower and use more VRAM.\n"
                "tiny ~75MB, base ~145MB, small ~480MB, medium ~1.5GB, large ~3GB.\n"
                "Larger = more accurate, slower."
            ),
        )

        ttk.Label(sf, text="Language:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._lang_var = tk.StringVar(value=self._cfg.language)
        lang_cb = ttk.Combobox(sf, textvariable=self._lang_var, values=LANGUAGE_CHOICES,
                               state="readonly", width=16)
        lang_cb.grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        lang_cb.bind("<<ComboboxSelected>>", self._on_lang_change)
        ToolTip(lang_cb, text="'auto' lets the model detect the spoken language.")

        ttk.Label(sf, text="Output dir:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._outdir_var = tk.StringVar(value=self._cfg.output_dir or "")
        outdir_entry = ttk.Entry(sf, textvariable=self._outdir_var, width=24)
        outdir_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(sf, text="Browse", command=self._browse_outdir, width=7).grid(
            row=2, column=2, padx=2, pady=(4, 0)
        )
        ToolTip(outdir_entry, text="Leave empty to write output alongside each source file.")
        # FocusOut rather than a write-trace: a trace would rewrite the config
        # file on every keystroke; _on_close still does the final sync.
        outdir_entry.bind("<FocusOut>", self._on_outdir_change)

        prompt_label = ttk.Label(sf, text="Vocabulary hint:")
        prompt_label.grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._prompt_var = tk.StringVar(value=self._cfg.initial_prompt)
        prompt_entry = ttk.Entry(sf, textvariable=self._prompt_var, width=24)
        prompt_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=4, pady=(4, 0))
        prompt_help = (
            "Words or names the audio is likely to contain, so recognition is "
            "biased toward them. Example: 'Dr. Okafor, RNA-seq, Kubernetes'. "
            "Leave empty for general audio."
        )
        # Same tip on label and entry: hovering the label is how people explore
        # an unfamiliar field.
        ToolTip(prompt_label, text=prompt_help)
        ToolTip(prompt_entry, text=prompt_help)
        prompt_entry.bind("<FocusOut>", self._on_prompt_change)

        self._strict_vad_var = tk.BooleanVar(value=self._cfg.strict_vad)
        strict_cb = ttk.Checkbutton(
            sf,
            text="Strict silence filtering",
            variable=self._strict_vad_var,
            command=self._on_strict_vad_change,
        )
        strict_cb.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ToolTip(
            strict_cb,
            text=(
                "Applies tighter VAD thresholds to cut more silence. "
                "Reduces hallucinations on quiet recordings but may clip soft speech."
            ),
        )

        self._batched_var = tk.BooleanVar(value=self._cfg.batched_gpu)
        batched_cb = ttk.Checkbutton(
            sf,
            text="Batched GPU mode",
            variable=self._batched_var,
            command=self._on_batched_change,
        )
        batched_cb.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ToolTip(
            batched_cb,
            text=(
                "Transcribes audio chunks in parallel on the GPU - measured "
                "~2.5x faster. Ignored on CPU. Uses more VRAM."
            ),
        )

        ff = ttk.LabelFrame(outer, text="Output Formats")
        ff.grid(row=0, column=1, sticky="nsew", ipadx=4, ipady=4)
        ToolTip(ff, text="Each checked format writes one file per transcribed input.")

        self._fmt_vars: dict[str, tk.BooleanVar] = {}
        for i, fmt in enumerate(FORMAT_WRITERS.keys()):
            var = tk.BooleanVar(value=fmt in self._cfg.formats)
            self._fmt_vars[fmt] = var
            cb = ttk.Checkbutton(ff, text=fmt.upper(), variable=var,
                                 command=self._on_format_change)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=6)
            tip = _FORMAT_TOOLTIPS.get(fmt)
            if tip:
                ToolTip(cb, text=tip)

    def _build_progress(self, parent: tk.Widget) -> None:
        self._progress = ttk.Progressbar(parent, orient=HORIZONTAL, mode="determinate",
                                         bootstyle="success-striped", maximum=100)
        self._progress.grid(row=3, column=0, sticky="ew", padx=6, pady=(2, 0))

    def _build_statusbar(self, parent: tk.Widget) -> None:
        self._status_var = tk.StringVar(value="Ready.")
        bar = ttk.Label(parent, textvariable=self._status_var, anchor="w",
                        bootstyle="secondary", padding=(6, 2))
        bar.grid(row=4, column=0, sticky="ew")
        self._status_label = bar

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        self._ensure_backend()
        if self._worker:
            self._set_status("Loading model...")
            self._worker.preload()

    def _ensure_backend(self) -> None:
        """(Re)create backend and worker if any construction-time params changed."""
        lang = self._lang_var.get()
        model = self._model_var.get()
        strict_vad = self._strict_vad_var.get()
        batched = self._batched_var.get()
        initial_prompt = self._prompt_var.get().strip()

        if (self._backend is not None
                and model == self._backend_model
                and lang == self._backend_lang
                and strict_vad == self._backend_strict_vad
                and batched == self._backend_batched
                and initial_prompt == self._backend_initial_prompt
                and self._worker is not None):
            return

        # Retire the previous worker: its thread would otherwise live forever,
        # pin the old model in memory, and could fire stale callbacks at the UI.
        if self._worker is not None:
            self._worker.shutdown()

        self._backend_model = model
        self._backend_lang = lang
        self._backend_strict_vad = strict_vad
        self._backend_batched = batched
        self._backend_initial_prompt = initial_prompt
        self._backend = create_backend(
            model_size=model,
            device="auto",
            language=language_to_param(lang),
            initial_prompt=initial_prompt or None,
            strict_vad=strict_vad,
            batched=batched,
        )
        cbs = WorkerCallbacks(
            on_file_start=self._on_file_start,
            on_segment_progress=self._on_segment_progress,
            on_file_done=self._on_file_done,
            on_batch_done=self._on_batch_done,
            on_status=self._set_status,
        )
        self._worker = TranscriptionWorker(
            backend=self._backend,
            dispatch=self._dispatch_to_ui,
            callbacks=cbs,
        )

    def _dispatch_to_ui(self, fn) -> None:
        """Schedule fn on the Tk thread; runs on the worker thread.

        after() raises once the window is destroyed (e.g. worker finishes a
        model load mid-close); swallow that instead of killing the worker
        thread with an unhandled exception.
        """
        try:
            self._app.after(0, fn)
        except (RuntimeError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                ("Media files", " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        self._add_paths(paths)

    def _add_paths(self, paths) -> None:
        existing = set(self._file_paths)
        added = 0
        for p in paths:
            if p not in existing:
                self._file_paths.append(p)
                existing.add(p)
                self._tree.insert("", END, iid=p, values=(Path(p).name, "Queued", ""))
                added += 1
        if added:
            self._set_status(f"Added {added} file(s). {len(self._file_paths)} total.")

    def _on_drop(self, event: object) -> None:
        # tkdnd wraps paths containing spaces in braces; splitlist undoes that.
        raw = self._app.tk.splitlist(event.data)  # type: ignore[attr-defined]
        self._add_paths(filter_media_paths(raw))

    def _remove_selected(self) -> None:
        for iid in self._tree.selection():
            self._tree.delete(iid)
            if iid in self._file_paths:
                self._file_paths.remove(iid)

    def _clear_files(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._file_paths.clear()
        self._progress["value"] = 0
        self._set_status("Ready.")

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self._outdir_var.set(d)
            self._on_outdir_change()

    # ------------------------------------------------------------------
    # Transcription control
    # ------------------------------------------------------------------

    def _start_transcribe(self) -> None:
        if not self._file_paths:
            self._set_status("No files added. Use 'Add Files' first.", danger=True)
            return
        formats = [fmt for fmt, var in self._fmt_vars.items() if var.get()]
        if not formats:
            self._set_status("No output format selected. Check at least one format.", danger=True)
            return

        self._ensure_backend()
        assert self._worker is not None

        out_dir = self._outdir_var.get().strip() or None
        lang = language_to_param(self._lang_var.get())
        jobs = build_jobs(self._file_paths, formats, out_dir, lang)

        for p in self._file_paths:
            self._tree_set(p, status="Queued", progress="")

        # Reset batch counters so a new run never inherits state from the previous one.
        self._batch_total = len(jobs)
        self._batch_started = 0
        self._batch_start_time = time.monotonic()
        self._batch_file_wall_times = []
        self._current_file_start_time = 0.0

        self._progress["value"] = 0
        self._running = True
        self._set_running_state(running=True)
        self._worker.submit(jobs)

        self._set_status("Transcribing...")

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self._set_status("Stop requested...")

    # ------------------------------------------------------------------
    # Worker callbacks (always called on Tk thread via dispatch)
    # ------------------------------------------------------------------

    def _on_file_start(self, path: str) -> None:
        self._tree_set(path, status="Running", progress="0%")
        self._progress["value"] = 0
        self._batch_started += 1
        self._current_file_start_time = time.monotonic()
        elapsed = time.monotonic() - self._batch_start_time
        txt = batch_progress_text(
            self._batch_started,
            self._batch_total,
            elapsed,
            self._batch_file_wall_times,
        )
        self._set_status(txt)

    def _on_segment_progress(self, path: str, fraction: float) -> None:
        pct = int(fraction * 100)
        self._tree_set(path, progress=f"{pct}%")
        self._progress["value"] = pct

    def _on_file_done(self, result: FileResult) -> None:
        if result.ok:
            self._tree_set(result.path, status="Done", progress="100%")
            # Track wall time of completed files to improve ETA for remaining ones.
            if self._current_file_start_time > 0:
                self._batch_file_wall_times.append(
                    time.monotonic() - self._current_file_start_time
                )
        elif "Cancelled" in result.message:
            self._tree_set(result.path, status="Cancelled", progress="")
        else:
            self._tree_set(result.path, status="Failed", progress="")
            logger.error("Failed %s: %s", Path(result.path).name, result.message)

    def _on_batch_done(self, ok: int, fail: int, elapsed: float, summary_path: str | None) -> None:
        self._running = False
        self._set_running_state(running=False)
        for iid in self._tree.get_children():
            if self._tree.set(iid, "status") == "Queued":
                self._tree_set(iid, status="Cancelled", progress="")
        summary = (
            f"Done: {ok} ok, {fail} failed in {format_elapsed(elapsed)} (log: {self._log_path})"
        )
        if summary_path:
            summary = f"{summary} - summary: {Path(summary_path).name}"
        self._set_status(summary)
        self._progress["value"] = 100 if fail == 0 and ok > 0 else self._progress["value"]

    # ------------------------------------------------------------------
    # Settings change handlers
    # ------------------------------------------------------------------

    def _save_setting(self, attr: str, value: object) -> None:
        setattr(self._cfg, attr, value)
        save_config(self._cfg)

    def _on_model_change(self, _event: object = None) -> None:
        # Backend will be recreated lazily on next transcribe / explicit preload
        self._save_setting("model_size", self._model_var.get())

    def _on_lang_change(self, _event: object = None) -> None:
        self._save_setting("language", self._lang_var.get())

    def _on_format_change(self) -> None:
        self._save_setting("formats", [f for f, v in self._fmt_vars.items() if v.get()])

    def _on_outdir_change(self, _event: object = None) -> None:
        self._save_setting("output_dir", self._outdir_var.get().strip() or None)

    def _on_prompt_change(self, _event: object = None) -> None:
        self._save_setting("initial_prompt", self._prompt_var.get().strip())

    def _on_strict_vad_change(self) -> None:
        self._save_setting("strict_vad", self._strict_vad_var.get())

    def _on_batched_change(self) -> None:
        self._save_setting("batched_gpu", self._batched_var.get())

    def _toggle_theme(self) -> None:
        theme = "darkly" if self._dark_var.get() else "yeti"
        self._app.style.theme_use(theme)
        self._save_setting("theme", theme)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tree_set(self, iid: str, **kwargs: str) -> None:
        """Update treeview columns by name; silently skip unknown iids."""
        if self._tree.exists(iid):
            for col, val in kwargs.items():
                self._tree.set(iid, col, val)

    def _set_status(self, text: str, danger: bool = False) -> None:
        # The worker fires "Loading model..." without knowing whether the model
        # needs downloading; only the GUI knows about cache semantics, so we
        # substitute the download message here rather than coupling engine.py
        # imports into worker.py.
        if text == "Loading model...":
            model = self._model_var.get()
            lang = language_to_param(self._lang_var.get())
            if not model_is_cached(model, lang):
                size_hint = {
                    "tiny": "~75 MB",
                    "base": "~145 MB",
                    "small": "~480 MB",
                    "medium": "~1.5 GB",
                    "large-v2": "~3 GB",
                    "large-v3": "~3 GB",
                    "distil-large-v3": "~1.5 GB",
                }.get(model, "hundreds of MB - GB")
                text = (
                    f"Downloading {model} model (first use; {size_hint})..."
                )
        self._status_var.set(text)
        style = "danger" if danger else "secondary"
        self._status_label.configure(bootstyle=style)
        logger.info("Status: %s", text)

    def _set_running_state(self, running: bool) -> None:
        state = DISABLED if running else NORMAL
        idle_state = NORMAL if running else DISABLED
        for btn in (self._btn_add, self._btn_remove, self._btn_clear, self._btn_transcribe):
            btn.configure(state=state)
        self._btn_stop.configure(state=idle_state)

    def _on_close(self) -> None:
        if self._worker:
            self._worker.stop()
        # Entries may hold un-FocusOut'd edits; flush both here as a last sync.
        self._save_setting("output_dir", self._outdir_var.get().strip() or None)
        self._save_setting("initial_prompt", self._prompt_var.get().strip())
        self._app.destroy()

    def run(self) -> None:
        self._app.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = TranscriberApp()
    app.run()
