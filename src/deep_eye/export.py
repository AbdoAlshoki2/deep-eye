"""Flat, one-record-per-span exports of trace files, for datasets and other tools.

A trace file stores each span as two events (start and end). An export merges them
and adds the run's details, so every record stands on its own:

    from deep_eye.export import span_records
    from deep_eye.storage import list_runs

    llm_calls = [r for r in span_records(list_runs()) if r["kind"] == "llm"]
"""

import json
from collections.abc import Iterable, Iterator
from typing import IO

from .models import Run, Span
from .schema import SCHEMA_VERSION


def _depths(run: Run) -> dict[str, int]:
    """Nesting level of each span (the root is 0)."""
    by_id = {s.span_id: s for s in run.spans}
    depths = {}
    for span in run.spans:
        depth, parent = 0, by_id.get(span.parent_id)
        while parent is not None and depth < len(run.spans):  # the bound guards against cycles
            depth, parent = depth + 1, by_id.get(parent.parent_id)
        depths[span.span_id] = depth
    return depths


def span_record(run: Run, span: Span, depth: int | None = None) -> dict:
    return {
        "v": SCHEMA_VERSION,
        "run_id": run.run_id,
        "run_name": run.name,
        "run_status": run.status,
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "depth": _depths(run)[span.span_id] if depth is None else depth,
        "name": span.name,
        "kind": span.kind,
        "status": run.span_status(span),
        "start": span.start,
        "end": span.end,
        "duration": span.duration,
        "input": span.input,
        "output": span.output,
        "error": span.error,
        "usage": span.usage,
        "attrs": span.attrs,
    }


def span_records(
    runs: Iterable[Run],
    kinds: Iterable[str] | None = None,
    run_statuses: Iterable[str] | None = None,
) -> Iterator[dict]:
    """Every span of every run, in start order, optionally limited to some span kinds
    (e.g. {"llm"}) and to runs with some statuses (e.g. {"ok"})."""
    kinds = set(kinds) if kinds else None
    run_statuses = set(run_statuses) if run_statuses else None
    for run in runs:
        if run_statuses and run.status not in run_statuses:
            continue
        depths = _depths(run)
        for span in sorted(run.spans, key=lambda s: s.start):
            if kinds is None or span.kind in kinds:
                yield span_record(run, span, depths[span.span_id])


def write_jsonl(records: Iterable[dict], out: IO[bytes]) -> int:
    """Write records as JSON Lines (UTF-8); returns how many were written."""
    count = 0
    for record in records:
        out.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        count += 1
    return count
