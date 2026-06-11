"""GUI entry point for TranscriptionHackery."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import ttkbootstrap as ttk
from ttkbootstrap.constants import DISABLED, END, HORIZONTAL, LEFT, NORMAL, RIGHT, VERTICAL
from ttkbootstrap.widgets import ToolTip

from config import AppConfig, load_config, save_config, setup_logging
from engine import MODEL_SIZES, TranscriptionBackend, create_backend
from worker import (
    FileResult,
    TranscriptionJob,
    TranscriptionWorker,
    WorkerCallbacks,
)
from writers import FORMAT_WRITERS

logger = logging.getLogger("transcriber")

LANGUAGE_CHOICES: list[str] = ["en", "auto", "es", "fr", "de", "zh", "ja", "pt", "ru"]


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
        self._app.title("AV Voice2Text")
        self._app.minsize(700, 480)
        self._app.resizable(True, True)

        # Tracks full paths; iid == path
        self._file_paths: list[str] = []

        # Current backend params so we can detect when recreation is needed
        self._backend_model: str = self._cfg.model_size
        self._backend_lang: str = self._cfg.language
        self._backend: TranscriptionBackend | None = None
        self._worker: TranscriptionWorker | None = None

        self._running = False

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

    def _build_toolbar(self, parent: tk.Widget) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        self._btn_add = ttk.Button(bar, text="Add Files", command=self._add_files)
        self._btn_add.pack(side=LEFT, padx=2)

        self._btn_remove = ttk.Button(bar, text="Remove Selected", command=self._remove_selected)
        self._btn_remove.pack(side=LEFT, padx=2)

        self._btn_clear = ttk.Button(bar, text="Clear", command=self._clear_files)
        self._btn_clear.pack(side=LEFT, padx=2)

        self._btn_transcribe = ttk.Button(
            bar, text="Transcribe", bootstyle="success", command=self._start_transcribe
        )
        self._btn_transcribe.pack(side=LEFT, padx=(12, 2))

        self._btn_stop = ttk.Button(
            bar, text="Stop", bootstyle="danger", command=self._stop, state=DISABLED
        )
        self._btn_stop.pack(side=LEFT, padx=2)

        self._dark_var = tk.BooleanVar(value=self._cfg.theme == "darkly")
        cb_theme = ttk.Checkbutton(
            bar, text="Dark mode", variable=self._dark_var, bootstyle="round-toggle",
            command=self._toggle_theme,
        )
        cb_theme.pack(side=RIGHT, padx=4)

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
        ToolTip(model_cb, text="Larger models are more accurate but slower and use more VRAM.")

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
        self._outdir_var.trace_add("write", lambda *_: (
            setattr(self._cfg, "output_dir", self._outdir_var.get().strip() or None),
            save_config(self._cfg),
        ))

        ff = ttk.LabelFrame(outer, text="Output Formats")
        ff.grid(row=0, column=1, sticky="nsew", ipadx=4, ipady=4)

        self._fmt_vars: dict[str, tk.BooleanVar] = {}
        for i, fmt in enumerate(FORMAT_WRITERS.keys()):
            var = tk.BooleanVar(value=fmt in self._cfg.formats)
            self._fmt_vars[fmt] = var
            cb = ttk.Checkbutton(ff, text=fmt.upper(), variable=var,
                                 command=self._on_format_change)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=6)

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
        """(Re)create backend and worker if model/lang params changed."""
        lang = self._lang_var.get() if hasattr(self, "_lang_var") else self._cfg.language
        model = self._model_var.get() if hasattr(self, "_model_var") else self._cfg.model_size

        if (self._backend is not None
                and model == self._backend_model
                and lang == self._backend_lang
                and self._worker is not None):
            return

        # Retire the previous worker: its thread would otherwise live forever,
        # pin the old model in memory, and could fire stale callbacks at the UI.
        if self._worker is not None:
            self._worker.shutdown()

        self._backend_model = model
        self._backend_lang = lang
        self._backend = create_backend(
            model_size=model,
            device="auto",
            language=language_to_param(lang),
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
                (
                    "Media files",
                    "*.mp4 *.avi *.mov *.flv *.wmv *.mp3 *.wav *.aac *.flac *.ogg *.m4a",
                ),
                ("All files", "*.*"),
            ],
        )
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

    def _on_segment_progress(self, path: str, fraction: float) -> None:
        pct = int(fraction * 100)
        self._tree_set(path, progress=f"{pct}%")
        self._progress["value"] = pct

    def _on_file_done(self, result: FileResult) -> None:
        if result.ok:
            self._tree_set(result.path, status="Done", progress="100%")
        elif "Cancelled" in result.message:
            self._tree_set(result.path, status="Cancelled", progress="")
        else:
            self._tree_set(result.path, status="Failed", progress="")
            logger.error("Failed %s: %s", result.path, result.message)

    def _on_batch_done(self, ok: int, fail: int, elapsed: float) -> None:
        self._running = False
        self._set_running_state(running=False)
        for iid in self._tree.get_children():
            if self._tree.set(iid, "status") == "Queued":
                self._tree_set(iid, status="Cancelled", progress="")
        summary = (
            f"Done: {ok} ok, {fail} failed in {format_elapsed(elapsed)} (log: {self._log_path})"
        )
        self._set_status(summary)
        self._progress["value"] = 100 if fail == 0 and ok > 0 else self._progress["value"]

    # ------------------------------------------------------------------
    # Settings change handlers
    # ------------------------------------------------------------------

    def _on_model_change(self, _event: object = None) -> None:
        self._cfg.model_size = self._model_var.get()
        save_config(self._cfg)
        # Backend will be recreated lazily on next transcribe / explicit preload

    def _on_lang_change(self, _event: object = None) -> None:
        self._cfg.language = self._lang_var.get()
        save_config(self._cfg)

    def _on_format_change(self) -> None:
        self._cfg.formats = [f for f, v in self._fmt_vars.items() if v.get()]
        save_config(self._cfg)

    def _toggle_theme(self) -> None:
        theme = "darkly" if self._dark_var.get() else "yeti"
        self._app.style.theme_use(theme)
        self._cfg.theme = theme
        save_config(self._cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tree_set(self, iid: str, **kwargs: str) -> None:
        """Update treeview columns by name; silently skip unknown iids."""
        if self._tree.exists(iid):
            for col, val in kwargs.items():
                self._tree.set(iid, col, val)

    def _set_status(self, text: str, danger: bool = False) -> None:
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
        self._cfg.output_dir = self._outdir_var.get().strip() or None
        self._cfg.model_size = self._model_var.get()
        self._cfg.language = self._lang_var.get()
        self._cfg.formats = [f for f, v in self._fmt_vars.items() if v.get()]
        save_config(self._cfg)
        self._app.destroy()

    def run(self) -> None:
        self._app.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = TranscriberApp()
    app.run()
