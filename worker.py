"""Single-threaded transcription worker — no tkinter imports."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from engine import TranscriptionBackend
from writers import FORMAT_WRITERS

logger = logging.getLogger("transcriber")


@dataclass
class TranscriptionJob:
    path: str
    formats: list[str]
    # None means write alongside the source file
    output_dir: str | None
    language: str | None


@dataclass
class FileResult:
    path: str
    ok: bool
    message: str
    elapsed: float


@dataclass
class WorkerCallbacks:
    on_file_start: Callable[[str], None] | None = None
    on_segment_progress: Callable[[str, float], None] | None = None
    on_file_done: Callable[[FileResult], None] | None = None
    on_batch_done: Callable[[int, int, float], None] | None = None
    on_status: Callable[[str], None] | None = None


@dataclass
class _Batch:
    jobs: list[TranscriptionJob]


_PRELOAD = object()
_SHUTDOWN = object()


class TranscriptionWorker:
    def __init__(
        self,
        backend: TranscriptionBackend,
        dispatch: Callable[[Callable[[], None]], None],
        callbacks: WorkerCallbacks,
    ) -> None:
        self._backend = backend
        self._dispatch = dispatch
        self._callbacks = callbacks
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._shutdown_requested = False
        self._busy = False
        self._busy_lock = threading.Lock()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, jobs: Sequence[TranscriptionJob]) -> None:
        self._queue.put(_Batch(list(jobs)))

    def stop(self) -> None:
        self._stop_event.set()

    def preload(self) -> None:
        self._queue.put(_PRELOAD)

    def shutdown(self) -> None:
        """Permanently retire this worker: cancel in-flight work, suppress any
        further callbacks, and make the thread exit so the backend can be freed.
        """
        # Lambdas in _loop read self._callbacks at execution time, so swapping
        # the dataclass silences callbacks that are already dispatched too.
        self._callbacks = WorkerCallbacks()
        self._shutdown_requested = True
        self._stop_event.set()
        self._queue.put(_SHUTDOWN)

    @property
    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cb(self, fn: Callable[[], None]) -> None:
        """Route a zero-arg callable through dispatch."""
        self._dispatch(fn)

    def _loop(self) -> None:
        while True:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Flag is set before _SHUTDOWN is queued, so pending _Batch tasks are
            # also dropped rather than processed after a stop.
            if self._shutdown_requested:
                return

            if task is _PRELOAD:
                if not self._backend.is_loaded:
                    self._cb(lambda: self._fire_status("Loading model..."))
                    try:
                        self._backend.load()
                        self._cb(lambda: self._fire_status("Model ready."))
                    except Exception as exc:
                        logger.exception("Preload failed")
                        # Bind now: exc is unbound once the except block exits,
                        # but dispatch may run the lambda later on the UI thread.
                        msg = f"Model load failed: {exc}"
                        self._cb(lambda m=msg: self._fire_status(m))
                continue

            if not isinstance(task, _Batch):
                continue

            with self._busy_lock:
                self._busy = True
            self._stop_event.clear()
            batch_start = time.monotonic()
            ok_count = 0
            fail_count = 0

            try:
                if not self._backend.is_loaded:
                    self._cb(lambda: self._fire_status("Loading model..."))
                    self._backend.load()
                    self._cb(lambda: self._fire_status("Model ready."))

                for job in task.jobs:
                    if self._stop_event.is_set():
                        self._cb(lambda: self._fire_status("Stopped."))
                        break
                    result = self._process_file(job)
                    if result.ok:
                        ok_count += 1
                    else:
                        fail_count += 1

            except Exception:
                logger.exception("Unexpected batch error")
            finally:
                with self._busy_lock:
                    self._busy = False

            elapsed = time.monotonic() - batch_start
            self._cb(
                lambda o=ok_count, f=fail_count, e=elapsed: self._callbacks.on_batch_done
                and self._callbacks.on_batch_done(o, f, e)
            )

    def _process_file(self, job: TranscriptionJob) -> FileResult:
        path = job.path
        self._cb(lambda: self._callbacks.on_file_start and self._callbacks.on_file_start(path))
        self._cb(lambda: self._fire_status(f"Transcribing: {Path(path).name}"))

        t0 = time.monotonic()
        try:
            info, seg_iter = self._backend.transcribe(path, language=job.language)

            segments: list = []
            for seg in seg_iter:
                if self._stop_event.is_set():
                    break
                segments.append(seg)
                fraction = min(seg.end / info.duration, 1.0) if info.duration > 0 else 0.0
                self._cb(
                    lambda p=path, f=fraction: (
                        self._callbacks.on_segment_progress
                        and self._callbacks.on_segment_progress(p, f)
                    )
                )

            if self._stop_event.is_set():
                result = FileResult(
                    path=path, ok=False, message="Cancelled.", elapsed=time.monotonic() - t0
                )
                self._cb(
                    lambda r=result: self._callbacks.on_file_done
                    and self._callbacks.on_file_done(r)
                )
                return result

            out_dir = Path(job.output_dir) if job.output_dir else Path(path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(path).stem

            for fmt in job.formats:
                writer = FORMAT_WRITERS.get(fmt)
                if writer is None:
                    logger.warning("Unknown format %r — skipping", fmt)
                    continue
                writer(segments, out_dir / f"{stem}.{fmt}")

            elapsed = time.monotonic() - t0
            result = FileResult(path=path, ok=True, message="OK", elapsed=elapsed)
            logger.info("Done %s in %.1fs", path, elapsed)

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception("Failed to transcribe %s", path)
            result = FileResult(path=path, ok=False, message=str(exc), elapsed=elapsed)

        self._cb(lambda r=result: self._callbacks.on_file_done and self._callbacks.on_file_done(r))
        return result

    def _fire_status(self, text: str) -> None:
        if self._callbacks.on_status:
            self._callbacks.on_status(text)
