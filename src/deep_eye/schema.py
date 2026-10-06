"""The trace file format: its version, the span kinds, and a checker for adapter tests.

A trace file is JSON Lines. The first line is a run header, then each span writes a
"start" line when it opens and an "end" line when it closes:

    {"event": "run", "schema_version": 2, "run_id": "...", "start": 1760000000.0, "pid": 1234, "host": "..."}
    {"event": "start", "span_id": "...", "parent_id": null, "name": "...", "kind": "agent", "start": ..., "input": ...}
    {"event": "end", "span_id": "...", "end": ..., "output": ..., "error": null, "usage": null}

The README ("Trace files") documents every field. Adapter tests can check their output with

    from deep_eye.schema import validate_file
    assert validate_file(path) == []
"""

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2  # bumped whenever a change would break existing readers

# Every span has one of these kinds. Anything else is stored as "other", with the
# original name kept in attrs["original_kind"].
KINDS = ("agent", "chain", "llm", "tool", "retriever", "embedding", "function", "other")
ORIGINAL_KIND = "original_kind"

USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def normalize_kind(kind: Any, attrs: dict | None) -> tuple[str, dict | None]:
    """Map a kind outside KINDS to "other", keeping its name in attrs."""
    if kind in KINDS:
        return kind, attrs
    return "other", {**(attrs or {}), ORIGINAL_KIND: str(kind)}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_kind(event: dict, where: str, problems: list[str]) -> None:
    kind = event.get("kind")
    if kind not in KINDS:
        problems.append(f"{where}: kind {kind!r} is not one of {', '.join(KINDS)}")
    elif kind == "other" and not isinstance((event.get("attrs") or {}).get(ORIGINAL_KIND), str):
        problems.append(f"{where}: kind 'other' without attrs.{ORIGINAL_KIND}")


def _check_optional(event: dict, key: str, types: type | tuple, where: str, problems: list[str]) -> None:
    if key in event and event[key] is not None and not isinstance(event[key], types):
        problems.append(f"{where}: {key} must be {getattr(types, '__name__', types)} or null")


def validate_lines(lines: list[str], require_ended: bool = True) -> list[str]:
    """Problems with one run's lines (empty list: the run follows the schema).

    With require_ended=False, spans without an end line (a run still being written,
    or one that crashed) are accepted.
    """
    problems: list[str] = []
    events = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number}: not valid JSON")
            continue
        if not isinstance(event, dict):
            problems.append(f"line {number}: not a JSON object")
            continue
        events.append((number, event))
    if not events:
        return problems or ["the file is empty"]

    number, header = events[0]
    if header.get("event") != "run":
        problems.append(f"line {number}: the first line must be the run header (event 'run')")
    else:
        if header.get("schema_version") != SCHEMA_VERSION:
            problems.append(f"line {number}: schema_version is {header.get('schema_version')!r}, "
                            f"expected {SCHEMA_VERSION}")
        if not isinstance(header.get("run_id"), str):
            problems.append(f"line {number}: run_id must be a string")
        if not _is_number(header.get("start")):
            problems.append(f"line {number}: start must be a number")
        if not isinstance(header.get("pid"), int) or not isinstance(header.get("host"), str):
            problems.append(f"line {number}: pid (int) and host (string) are required")
        events = events[1:]

    started: dict[str, dict] = {}
    ended: set[str] = set()
    for number, event in events:
        where = f"line {number}"
        span_id = event.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            problems.append(f"{where}: span_id must be a non-empty string")
            continue
        kind_of_event = event.get("event")
        if kind_of_event == "start":
            if span_id in started:
                problems.append(f"{where}: span {span_id} started twice")
            started[span_id] = event
            if not isinstance(event.get("name"), str):
                problems.append(f"{where}: name must be a string")
            _check_kind(event, where, problems)
            if not _is_number(event.get("start")):
                problems.append(f"{where}: start must be a number")
            _check_optional(event, "parent_id", str, where, problems)
            _check_optional(event, "attrs", dict, where, problems)
        elif kind_of_event == "end":
            if span_id not in started:
                problems.append(f"{where}: end of span {span_id}, which never started")
            elif span_id in ended:
                problems.append(f"{where}: span {span_id} ended twice")
            ended.add(span_id)
            if not _is_number(event.get("end")):
                problems.append(f"{where}: end must be a number")
            elif span_id in started and _is_number(started[span_id].get("start")) \
                    and event["end"] < started[span_id]["start"]:
                problems.append(f"{where}: span {span_id} ends before it starts")
            _check_optional(event, "error", str, where, problems)
            _check_optional(event, "usage", dict, where, problems)
            _check_optional(event, "attrs", dict, where, problems)
            _check_optional(event, "name", str, where, problems)
            if "kind" in event:
                _check_kind(event, where, problems)
            if "interrupted" in event and not isinstance(event["interrupted"], bool):
                problems.append(f"{where}: interrupted must be true or false")
            for key in USAGE_KEYS:
                value = (event.get("usage") or {}).get(key)
                if value is not None and not isinstance(value, int):
                    problems.append(f"{where}: usage.{key} must be an integer")
        else:
            problems.append(f"{where}: unknown event {kind_of_event!r}")

    roots = [s for s, e in started.items() if e.get("parent_id") is None]
    if len(roots) != 1:
        problems.append(f"a run has exactly one root span (parent_id null), found {len(roots)}")
    for span_id, event in started.items():
        parent = event.get("parent_id")
        if parent is not None and parent not in started:
            problems.append(f"span {span_id}: parent {parent} is not in this run")
        if require_ended and span_id not in ended:
            problems.append(f"span {span_id} never ended")
    return problems


def validate_file(path: str | Path, require_ended: bool = True) -> list[str]:
    """Problems with one trace file (empty list: it follows the schema)."""
    return validate_lines(Path(path).read_text(encoding="utf-8").splitlines(), require_ended)
