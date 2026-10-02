"""Reading and writing traces as JSON Lines files, one file per run.

Each span produces two lines: a "start" event when it opens and an "end" event
when it closes. Writing the start immediately means a crashed or still-running
agent leaves a readable (partial) trace behind.
"""

import json
import threading
from pathlib import Path

from .config import get_trace_dir
from .models import Run, Span
from .serialize import to_jsonable

_lock = threading.Lock()
_cache: dict[Path, tuple[tuple[int, int], Run]] = {}  # path -> ((mtime_ns, size), parsed run)


def _append(run_id: str, event: dict) -> None:
    path = get_trace_dir() / f"{run_id}.jsonl"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_start(span: Span) -> None:
    _append(span.run_id, {
        "event": "start",
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "kind": span.kind,
        "start": span.start,
        "input": to_jsonable(span.input),
    })


def write_end(span: Span) -> None:
    _append(span.run_id, {
        "event": "end",
        "span_id": span.span_id,
        "end": span.end,
        "output": to_jsonable(span.output),
        "error": to_jsonable(span.error),
        "usage": to_jsonable(span.usage),
    })


def _parse_line(line: str) -> dict | None:
    """One event, or None for anything that isn't one (half-written line, foreign data, ...)."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or not isinstance(event.get("span_id"), str):
        return None
    return event


def _read_run(path: Path) -> Run:
    spans: dict[str, Span] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        event = _parse_line(line)
        if event is None:
            continue
        try:
            if event.get("event") == "start":
                spans[event["span_id"]] = Span(
                    span_id=event["span_id"],
                    run_id=path.stem,
                    parent_id=event.get("parent_id"),
                    name=str(event.get("name", "?")),
                    kind=str(event.get("kind", "function")),
                    start=float(event["start"]),
                    input=event.get("input"),
                )
            elif event.get("event") == "end" and event["span_id"] in spans:
                span = spans[event["span_id"]]
                span.end = float(event["end"])
                span.output = event.get("output")
                span.error = event.get("error")
                span.usage = event.get("usage") if isinstance(event.get("usage"), dict) else None
        except (KeyError, TypeError, ValueError):
            continue  # a line with the right shape but bad values: skip just that line
    return Run(run_id=path.stem, path=path, spans=list(spans.values()))


def load_run(path: Path) -> Run:
    """Rebuild a Run by merging the start/end events of every span.

    Results are cached until the file changes, so refreshing the viewer only
    re-parses runs that are still being written.
    """
    try:
        stat = path.stat()
    except OSError:
        return Run(run_id=path.stem, path=path, spans=[])
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    run = _read_run(path)
    _cache[path] = (key, run)
    return run


def list_runs(trace_dir: Path | None = None) -> list[Run]:
    """All runs in the trace directory, newest first."""
    trace_dir = trace_dir or get_trace_dir()
    if not trace_dir.is_dir():
        return []
    files = sorted(trace_dir.glob("*.jsonl"), reverse=True)  # names start with a timestamp
    present = set(files)
    for stale in [p for p in _cache if p.parent == trace_dir and p not in present]:
        del _cache[stale]  # deleted files
    return [load_run(f) for f in files]
