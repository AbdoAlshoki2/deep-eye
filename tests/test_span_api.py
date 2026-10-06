"""start_span() / end_span(): explicit handles, closed from any thread or task, never raising."""

import asyncio
import json
import logging
import threading

import pytest

import deep_eye
from deep_eye import current_span, end_span, span, start_span, storage, tracer
from deep_eye.models import Span
from deep_eye.schema import validate_file
from deep_eye.storage import list_runs


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)
    return tmp_path


def _events(trace_dir):
    deep_eye.flush()
    (path,) = trace_dir.glob("*.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_spans_started_in_threads_and_ended_out_of_order_keep_their_parents(trace_dir):
    handles = {}

    def start(name, parent_name):
        handles[name] = start_span(name, parent=handles[parent_name] if parent_name else None)

    for name, parent_name in (("root", None), ("child", "root"), ("grandchild", "child")):
        t = threading.Thread(target=start, args=(name, parent_name))
        t.start()
        t.join()

    enders = [threading.Thread(target=end_span, args=(handles[name],))
              for name in ("root", "grandchild", "child")]  # the parent first: out of order
    for t in enders:
        t.start()
    for t in enders:
        t.join()

    (run,) = list_runs()
    spans = {s.name: s for s in run.spans}
    assert spans["root"].parent_id is None
    assert spans["child"].parent_id == spans["root"].span_id
    assert spans["grandchild"].parent_id == spans["child"].span_id
    assert all(s.end is not None for s in run.spans) and run.status == "ok"
    assert validate_file(run.path) == []


def test_span_started_in_one_task_can_end_in_another():
    async def main():
        handle = start_span("request", kind="agent")
        await asyncio.gather(asyncio.create_task(asyncio.sleep(0)))
        await asyncio.create_task(asyncio.to_thread(end_span, handle, "done"))

    asyncio.run(main())
    (run,) = list_runs()
    assert (run.root.name, run.root.output, run.status) == ("request", "done", "ok")


def test_start_span_does_not_become_the_current_span():
    with span("outer") as outer:
        handle = start_span("explicit")
        assert current_span() is outer
        assert handle.parent_id == outer.span_id  # but it nests under the current one
        end_span(handle)


def test_parent_none_starts_a_new_run():
    with span("outer"):
        end_span(start_span("separate", parent=None))
    assert sorted(r.name for r in list_runs()) == ["outer", "separate"]


def test_ending_twice_is_ignored_with_a_debug_log(trace_dir, caplog):
    handle = start_span("once")
    end_span(handle, "first")
    with caplog.at_level(logging.DEBUG, logger="deep_eye"):
        end_span(handle, "second")
    assert "already ended" in caplog.text
    assert [e["event"] for e in _events(trace_dir)] == ["run", "start", "end"]
    assert list_runs()[0].root.output == "first"


@pytest.mark.parametrize("handle", [None, "abc", object(), Span(span_id="x", run_id="r", name="n")])
def test_unknown_handles_are_ignored_with_a_debug_log(handle, caplog):
    with caplog.at_level(logging.DEBUG, logger="deep_eye"):
        end_span(handle, "out", error="boom")
    assert "not a handle from start_span()" in caplog.text
    assert list_runs() == []


def test_a_span_loaded_from_a_file_is_not_a_live_handle():
    end_span(start_span("done"))
    loaded = list_runs()[0].root
    end_span(loaded, "again")  # no error, no new event
    assert list_runs()[0].root.output is None


def test_output_set_on_the_handle_is_kept():
    handle = start_span("llm-call", kind="llm")
    handle.output = "text"
    handle.usage = {"input_tokens": 3, "output_tokens": 4}
    end_span(handle)
    root = list_runs()[0].root
    assert (root.output, root.tokens) == ("text", 7)


def test_details_known_only_at_the_end_replace_the_start_ones(trace_dir):
    handle = start_span("pending", kind="function", attrs={"a": 1})
    end_span(handle, "out", input={"q": 1}, name="final", kind="retriever", attrs={"b": 2})
    root = list_runs()[0].root
    assert (root.name, root.kind, root.input) == ("final", "retriever", {"q": 1})
    assert root.attrs == {"a": 1, "b": 2}
    assert validate_file(root.file) == []


def test_kinds_outside_the_list_are_stored_as_other(trace_dir):
    end_span(start_span("plan", kind="planner"))
    start = _events(trace_dir)[1]
    assert start["kind"] == "other" and start["attrs"] == {"original_kind": "planner"}
    root = list_runs()[0].root
    assert (root.kind, root.display_kind) == ("other", "planner")


def test_explicit_times_are_used():
    end_span(start_span("replayed", start_time=100.0), end_time=102.5)
    root = list_runs()[0].root
    assert (root.start, root.end, root.duration) == (100.0, 102.5, 2.5)


def test_start_and_end_never_raise_even_if_deep_eye_breaks(monkeypatch):
    def broken(*args):
        raise RuntimeError("deep-eye bug")

    monkeypatch.setattr(tracer, "_new_run_id", broken)
    handle = start_span("x")
    assert handle.started and not handle.recording  # a stand-in that is never written
    end_span(handle)

    monkeypatch.undo()
    monkeypatch.setattr(storage, "write_end", broken)
    end_span(start_span("y"), "out")  # the write fails inside; the caller never sees it


def test_bad_parent_falls_back_to_the_current_span():
    with span("outer") as outer:
        handle = start_span("child", parent="not a span")
        assert handle.parent_id == outer.span_id
        end_span(handle)


def test_tracing_off_gives_stand_in_handles(trace_dir):
    deep_eye.configure(enabled=False)
    handle = start_span("off")
    end_span(handle, "x")
    end_span(handle, "x")  # still ignored quietly
    assert not handle.recording and list(trace_dir.glob("*.jsonl")) == []
