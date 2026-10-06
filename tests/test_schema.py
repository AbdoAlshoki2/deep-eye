"""The trace format: run header, fixed kinds, old files, and newer versions the viewer can't read."""

import asyncio
import json

import pytest
from textual.widgets import Static

import deep_eye
from deep_eye import span, trace
from deep_eye.cli import main
from deep_eye.export import span_records
from deep_eye.schema import KINDS, SCHEMA_VERSION, validate_file, validate_lines
from deep_eye.storage import list_runs
from deep_eye.tui import DeepEyeApp


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)
    return tmp_path


def _write(trace_dir, name, events):
    path = trace_dir / f"{name}.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


def test_traced_code_follows_the_schema():
    @trace(kind="agent")
    def agent():
        with span("call", kind="llm", attrs={"model": "m"}) as s:
            s.usage = {"input_tokens": 1, "output_tokens": 2}
        try:
            with span("bad", kind="tool"):
                raise ValueError("x")
        except ValueError:
            pass

    agent()
    (run,) = list_runs()
    assert validate_file(run.path) == []
    header = json.loads(run.path.read_text(encoding="utf-8").splitlines()[0])
    assert (header["event"], header["schema_version"], header["run_id"]) == ("run", SCHEMA_VERSION, run.run_id)


@pytest.mark.parametrize("lines, problem", [
    ([], "empty"),
    (['{"event": "start", "span_id": "a", "parent_id": null, "name": "x", "kind": "llm", "start": 1}'],
     "first line must be the run header"),
    (['{"event": "run", "schema_version": 2, "run_id": "r", "start": 1, "pid": 1, "host": "h"}',
      '{"event": "start", "span_id": "a", "parent_id": null, "name": "x", "kind": "magic", "start": 1}',
      '{"event": "end", "span_id": "a", "end": 2}'], "kind 'magic'"),
    (['{"event": "run", "schema_version": 2, "run_id": "r", "start": 1, "pid": 1, "host": "h"}',
      '{"event": "start", "span_id": "a", "parent_id": null, "name": "x", "kind": "llm", "start": 1}'],
     "never ended"),
    (['{"event": "run", "schema_version": 2, "run_id": "r", "start": 1, "pid": 1, "host": "h"}',
      '{"event": "start", "span_id": "a", "parent_id": null, "name": "x", "kind": "llm", "start": 1}',
      '{"event": "start", "span_id": "b", "parent_id": "zzz", "name": "y", "kind": "llm", "start": 1}',
      '{"event": "end", "span_id": "b", "end": 2}', '{"event": "end", "span_id": "a", "end": 2}'],
     "parent zzz is not in this run"),
    (['{"event": "run", "schema_version": 2, "run_id": "r", "start": 1, "pid": 1, "host": "h"}',
      '{"event": "start", "span_id": "a", "parent_id": null, "name": "x", "kind": "llm", "start": 1}',
      '{"event": "end", "span_id": "a", "end": 2, "usage": {"input_tokens": "3"}}'],
     "usage.input_tokens must be an integer"),
])
def test_validator_reports_problems(lines, problem):
    assert any(problem in p for p in validate_lines(lines)), validate_lines(lines)


def test_version_1_files_still_load(trace_dir):
    _write(trace_dir, "20250101-000000_aaaaaa", [
        {"v": 1, "event": "start", "span_id": "a", "parent_id": None, "name": "old", "kind": "planner",
         "start": 1.0, "input": "q", "pid": 1, "host": "elsewhere"},
        {"v": 1, "event": "end", "span_id": "a", "end": 2.0, "output": "ok", "error": None, "usage": None},
    ])
    (run,) = list_runs()
    assert run.supported and run.status == "ok" and run.host == "elsewhere"
    assert (run.root.kind, run.root.display_kind) == ("other", "planner")


def _future_run(trace_dir):
    return _write(trace_dir, "29990101-000000_ffffff", [
        {"event": "run", "schema_version": 99, "run_id": "29990101-000000_ffffff", "start": 1.0},
        {"event": "start", "span_id": "a", "parent_id": None, "name": "future", "kind": "llm", "start": 1.0},
    ])


def test_unknown_versions_are_not_guessed_at(trace_dir):
    _future_run(trace_dir)
    (run,) = list_runs()
    assert not run.supported and run.status == "unsupported" and run.spans == []
    assert "version 99" in run.unsupported_reason and "Upgrade deep-eye" in run.unsupported_reason
    assert list(span_records([run])) == []


def test_cli_explains_unknown_versions(trace_dir, capsys):
    _future_run(trace_dir)
    main(["list"])
    assert "unsupported" in capsys.readouterr().out
    main(["show", "latest", "--no-color"])
    assert "only reads versions 1 to" in " ".join(capsys.readouterr().out.split())
    main(["export"])
    assert "Skipped 29990101-000000_ffffff" in capsys.readouterr().err


def test_viewer_explains_unknown_versions(trace_dir):
    _future_run(trace_dir)

    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            detail = str(app.screen.query_one("#detail", Static).render())
            assert "Upgrade deep-eye" in detail

    asyncio.run(scenario())


def test_every_kind_has_a_style():
    from deep_eye.themes import THEMES

    for theme in THEMES.values():
        assert all(theme.palette.kind_style(kind) for kind in KINDS)


def test_export_includes_attrs():
    with span("call", kind="llm", attrs={"model": "m"}):
        pass
    (record,) = span_records(list_runs())
    assert record["attrs"] == {"model": "m"} and record["v"] == SCHEMA_VERSION
