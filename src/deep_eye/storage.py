"""Reading and writing traces as JSON Lines files, one file per run.

A file starts with a run header line (format version, process), then each span
produces two lines: a "start" event when it opens and an "end" event when it
closes. Writing the start immediately means a crashed or still-running agent
leaves a readable (partial) trace behind. schema.py describes the format.

Files are plain JSON on purpose, so they are easy to read, grep and load into other
tools. They are kept private to your user instead: the folder is created with mode 700
and each file with mode 600 (POSIX only; on Windows they inherit the folder's ACL).
"""

import json
import logging
import os
import queue
import stat
import sys
import threading
import time
from pathlib import Path

from .config import get_trace_dir, sync_writes
from .models import HOST, Run, Span
from .schema import SCHEMA_VERSION, normalize_kind
from .serialize import to_jsonable

log = logging.getLogger("deep_eye")

_lock = threading.Lock()
_cache: dict[Path, tuple[tuple[int, int], Run]] = {}  # path -> ((mtime_ns, size), parsed run)
_checked_dirs: set[Path] = set()  # folders already created or permission-checked
_last_cleanup: dict[Path, float] = {}  # folder -> when old traces were last deleted
CLEANUP_EVERY = 60  # seconds: at most one max_age_days sweep per folder in this window

_FILE_FLAGS = (os.O_WRONLY | os.O_APPEND | os.O_CREAT
               | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))


def run_file(run_id: str, trace_dir: Path | None = None) -> Path:
    return (trace_dir or get_trace_dir()) / f"{run_id}.jsonl"


def _ensure_dir(folder: Path) -> None:
    """Create the folder as owner-only, or warn once if an existing one is open to others."""
    if folder in _checked_dirs:
        return
    if not folder.is_dir():
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    elif sys.platform != "win32":
        mode = stat.S_IMODE(folder.stat().st_mode)
        if mode & 0o077:
            log.warning("deep-eye: trace folder %s is readable by other users (mode %o); "
                        "run `chmod 700` on it if traces may contain private data", folder, mode)
    _checked_dirs.add(folder)


def _open_for_append(path: Path) -> int:
    _ensure_dir(path.parent)
    # os.open with a mode, so the file is never readable by others, not even for a moment
    try:
        return os.open(path, _FILE_FLAGS, 0o600)
    except FileNotFoundError:  # the folder was deleted (e.g. `deep-eye clear`) while we ran
        _checked_dirs.discard(path.parent)
        _ensure_dir(path.parent)
        return os.open(path, _FILE_FLAGS, 0o600)


def _write_batch(batch: list[tuple[Path, dict]]) -> None:
    """Append events, opening each file once. Raises the first error after trying every file."""
    lines: dict[Path, list[str]] = {}
    for path, event in batch:
        lines.setdefault(path, []).append(json.dumps(event, ensure_ascii=False) + "\n")
    first_error: Exception | None = None
    with _lock:
        for path, chunk in lines.items():
            try:
                with os.fdopen(_open_for_append(path), "ab") as f:
                    f.write("".join(chunk).encode("utf-8"))
            except Exception as exc:
                first_error = first_error or exc
    if first_error is not None:
        raise first_error


# --- the background writer ------------------------------------------------------------
#
# Traced code only snapshots its values (to_jsonable) and queues the event; a daemon
# thread turns events into JSON and appends them to the files. That keeps file I/O off
# the caller's thread and off the asyncio event loop. Events still queued when the
# process exits are flushed by an atexit hook (see tracer.py); only a hard kill or a
# crash in native code can lose the last few. `sync_writes` trades speed for that.

QUEUE_SIZE = 10_000  # events waiting to be written; beyond that new events are dropped
_queue: queue.Queue = queue.Queue(QUEUE_SIZE)
_writer: threading.Thread | None = None
_writer_pid: int | None = None
_writer_lock = threading.Lock()
_warned = False


def _warn_once(message: str, exc_info: bool = False) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning("deep-eye: %s; later problems are not reported", message, exc_info=exc_info)


def _writer_loop(q: queue.Queue) -> None:
    while True:
        batch = [q.get()]
        while len(batch) < 1000:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        try:
            _write_batch(batch)
        except Exception:
            _warn_once("failed to write trace events", exc_info=True)
        finally:
            for _ in batch:
                q.task_done()


def _writer_running() -> bool:
    return _writer is not None and _writer_pid == os.getpid() and _writer.is_alive()


def _start_writer() -> bool:
    """Start the writer thread if needed (again after a fork). False if threads can't start."""
    global _queue, _writer, _writer_pid
    if _writer_running():
        return True
    with _writer_lock:
        if _writer_running():
            return True
        if _writer_pid is not None and _writer_pid != os.getpid():
            _queue = queue.Queue(QUEUE_SIZE)  # a forked child: the parent's queue isn't ours
        thread = threading.Thread(target=_writer_loop, args=(_queue,), name="deep-eye-writer", daemon=True)
        try:
            thread.start()
        except RuntimeError:  # the interpreter is shutting down
            return False
        _writer, _writer_pid = thread, os.getpid()
    return True


def _append(span: Span, event: dict) -> None:
    item = (span.file or run_file(span.run_id), event)
    if sync_writes() or not _start_writer():
        _write_batch([item])
        return
    try:
        _queue.put_nowait(item)
    except queue.Full:  # the disk can't keep up: drop rather than slow the program down
        _warn_once(f"more than {QUEUE_SIZE} trace events are waiting to be written; dropping new ones")


def flush(timeout: float | None = 5.0) -> bool:
    """Wait until every queued event is on disk. Returns False if `timeout` (seconds) ran out."""
    global _queue
    if _writer_pid is not None and _writer_pid != os.getpid():
        _queue = queue.Queue(QUEUE_SIZE)  # a forked child: the parent writes its own events
        return True
    if not _writer_running():
        batch = []
        while True:
            try:
                batch.append(_queue.get_nowait())
            except queue.Empty:
                break
            _queue.task_done()
        if batch:
            _write_batch(batch)
        return True
    deadline = None if timeout is None else time.monotonic() + timeout
    q = _queue
    with q.all_tasks_done:
        while q.unfinished_tasks:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            q.all_tasks_done.wait(remaining)
    return True


# --- events ---------------------------------------------------------------------------

def write_header(root: Span) -> None:
    """The run's first line. pid and host let the viewer tell a crashed run from a live one."""
    _append(root, {
        "event": "run",
        "schema_version": SCHEMA_VERSION,
        "run_id": root.run_id,
        "start": root.start,
        "pid": os.getpid(),
        "host": HOST,
    })


def write_start(span: Span) -> None:
    event = {
        "event": "start",
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "kind": span.kind,
        "start": span.start,
        "input": to_jsonable(span.input),
    }
    if span.attrs:
        event["attrs"] = to_jsonable(span.attrs)
    _append(span, event)


def write_end(span: Span, late: dict | None = None) -> None:
    """`late`: name / kind / input / attrs learned only when the span ended. They replace
    the start event's values (attrs are merged into them)."""
    event = {
        "event": "end",
        "span_id": span.span_id,
        "end": span.end,
        "output": to_jsonable(span.output),
        "error": to_jsonable(span.error),
        "usage": to_jsonable(span.usage),
    }
    if span.interrupted:
        event["interrupted"] = True
    for key, value in (late or {}).items():
        event[key] = to_jsonable(value)
    _append(span, event)


# --- deleting traces ------------------------------------------------------------------

def trace_files(trace_dir: Path) -> list[Path]:
    """The *.jsonl files directly inside `trace_dir`. Symlinks are skipped, never followed."""
    if not trace_dir.is_dir():
        return []
    return [p for p in trace_dir.glob("*.jsonl") if not p.is_symlink() and p.is_file()]


def delete_run(path: Path, trace_dir: Path | None = None) -> None:
    """Delete one trace file, refusing anything that isn't a plain .jsonl file in `trace_dir`."""
    trace_dir = (trace_dir or get_trace_dir()).resolve()
    if path.is_symlink() or path.suffix != ".jsonl" or path.resolve().parent != trace_dir:
        raise ValueError(f"{path} is not a trace file in {trace_dir}")
    path.unlink()
    _cache.pop(path, None)


def delete_older_than(trace_dir: Path, days: float) -> list[Path]:
    """Delete trace files last written more than `days` days ago; returns what was deleted."""
    cutoff = time.time() - days * 86400
    deleted = []
    for f in trace_files(trace_dir):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted.append(f)
        except OSError:
            pass  # in use (Windows) or already gone
    return deleted


def cleanup_if_due(trace_dir: Path, days: float | None) -> None:
    """The `max_age_days` sweep, run when a new run starts (at most once a minute per folder)."""
    if days is None:
        return
    now = time.time()
    if now - _last_cleanup.get(trace_dir, 0) < CLEANUP_EVERY:
        return
    _last_cleanup[trace_dir] = now
    delete_older_than(trace_dir, days)


# --- reading traces -------------------------------------------------------------------

def _parse_line(line: str) -> dict | None:
    """One event, or None for anything that isn't one (half-written line, foreign data, ...)."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    if event.get("event") != "run" and not isinstance(event.get("span_id"), str):
        return None
    return event


def _schema_version(events: list[dict]) -> object:
    """The file's format version: from the run header (version 2+), else from the events'
    "v" field (version 1 had no header, and the oldest files have neither)."""
    for event in events:
        if event.get("event") == "run":
            return event.get("schema_version")
    versions = {event.get("v", 1) for event in events}
    return 1 if versions <= {1} else next(v for v in versions if v != 1)


def _dict(value: object) -> dict | None:
    return value if isinstance(value, dict) else None


def _read_run(path: Path, mtime: float) -> Run:
    run = Run(run_id=path.stem, path=path, spans=[], mtime=mtime)
    spans: dict[str, Span] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").rstrip().split("\n")  # not splitlines(): it also splits on U+2028/U+2029/U+0085
    if lines and lines != [""]:
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            run.incomplete = True  # the writer stopped in the middle of a line
    events = [e for e in map(_parse_line, lines) if e is not None]
    run.schema_version = _schema_version(events)
    if not run.supported:
        return run  # a newer format: guessing at it could show wrong data
    for event in events:
        try:
            if event.get("event") == "run":
                if isinstance(event.get("pid"), int):
                    run.pid, run.host = event["pid"], str(event.get("host"))
            elif event.get("event") == "start":
                if event.get("parent_id") is None and isinstance(event.get("pid"), int):  # version 1
                    run.pid, run.host = event["pid"], str(event.get("host"))
                kind, attrs = normalize_kind(str(event.get("kind", "function")), _dict(event.get("attrs")))
                spans[event["span_id"]] = Span(
                    span_id=event["span_id"],
                    run_id=path.stem,
                    parent_id=event.get("parent_id"),
                    name=str(event.get("name", "?")),
                    kind=kind,
                    start=float(event["start"]),
                    input=event.get("input"),
                    attrs=attrs,
                    file=path,
                )
            elif event.get("event") == "end" and event["span_id"] in spans:
                span = spans[event["span_id"]]
                span.end = float(event["end"])
                span.output = event.get("output")
                span.error = event.get("error")
                span.usage = _dict(event.get("usage"))
                span.interrupted = event.get("interrupted") is True
                # values the writer only learned when the span ended
                if "name" in event:
                    span.name = str(event["name"])
                if "input" in event:
                    span.input = event["input"]
                if _dict(event.get("attrs")):
                    span.attrs = {**(span.attrs or {}), **event["attrs"]}
                if "kind" in event:
                    span.kind, span.attrs = normalize_kind(str(event["kind"]), span.attrs)
        except (KeyError, TypeError, ValueError):
            continue  # a line with the right shape but bad values: skip just that line
    run.spans = list(spans.values())
    return run


def load_run(path: Path) -> Run:
    """Rebuild a Run by merging the start/end events of every span.

    Results are cached until the file changes, so refreshing the viewer only
    re-parses runs that are still being written.
    """
    flush()  # see this process's own queued events too
    try:
        st = path.stat()
    except OSError:
        return Run(run_id=path.stem, path=path, spans=[])
    key = (st.st_mtime_ns, st.st_size)
    cached = _cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    run = _read_run(path, st.st_mtime)
    _cache[path] = (key, run)
    return run


def list_runs(trace_dir: Path | None = None) -> list[Run]:
    """All runs in the trace directory, newest first."""
    flush()
    trace_dir = trace_dir or get_trace_dir()
    files = sorted(trace_files(trace_dir), reverse=True)  # names start with a timestamp
    present = set(files)
    for stale in [p for p in _cache if p.parent == trace_dir and p not in present]:
        del _cache[stale]  # deleted files
    return [load_run(f) for f in files]
