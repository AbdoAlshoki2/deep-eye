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
        "error": span.error,
        "usage": to_jsonable(span.usage),
    })


def load_run(path: Path) -> Run:
    """Rebuild a Run by merging the start/end events of every span."""
    spans: dict[str, Span] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a half-written last line of a live run
        if event["event"] == "start":
            spans[event["span_id"]] = Span(
                span_id=event["span_id"],
                run_id=path.stem,
                parent_id=event["parent_id"],
                name=event["name"],
                kind=event["kind"],
                start=event["start"],
                input=event["input"],
            )
        elif event["span_id"] in spans:
            span = spans[event["span_id"]]
            span.end = event["end"]
            span.output = event["output"]
            span.error = event["error"]
            span.usage = event.get("usage")
    return Run(run_id=path.stem, path=path, spans=list(spans.values()))


def list_runs(trace_dir: Path | None = None) -> list[Run]:
    """All runs in the trace directory, newest first."""
    trace_dir = trace_dir or get_trace_dir()
    if not trace_dir.is_dir():
        return []
    files = sorted(trace_dir.glob("*.jsonl"), reverse=True)  # names start with a timestamp
    return [load_run(f) for f in files]
