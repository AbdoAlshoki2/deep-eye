"""Turning tracing off, sampling, the background writer, the format version and exports."""

import asyncio
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading

import pytest

import deep_eye
from deep_eye import span, storage, trace
from deep_eye.cli import main
from deep_eye.storage import list_runs


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path / "traces")
    return tmp_path / "traces"


def _events(folder) -> list[dict]:
    deep_eye.flush()
    return [json.loads(line) for p in folder.glob("*.jsonl")
            for line in p.read_text(encoding="utf-8").splitlines()]


# --- off switch and sampling ------------------------------------------------------------

def test_disabled_tracing_runs_code_untouched(trace_dir):
    deep_eye.configure(enabled=False)

    @trace
    def f(x):
        return x + 1

    @trace
    async def af(x):
        return x * 2

    @trace
    def gen():
        received = yield 1
        yield received

    @trace
    async def agen():
        yield "a"

    async def collect():
        return [x async for x in agen()]

    assert f(1) == 2 and asyncio.run(af(2)) == 4 and asyncio.run(collect()) == ["a"]
    g = gen()
    assert next(g) == 1 and g.send("back") == "back"
    with span("block") as s:
        s.output = "ignored"
    assert not trace_dir.exists()


def test_env_var_turns_tracing_off(trace_dir, monkeypatch):
    monkeypatch.setenv("DEEP_EYE_ENABLED", "false")
    trace(lambda: None)()
    assert not trace_dir.exists()
    monkeypatch.setenv("DEEP_EYE_ENABLED", "1")
    trace(lambda: None)()
    assert len(list_runs()) == 1


def test_sampling_keeps_or_drops_whole_runs(trace_dir, monkeypatch):
    import random

    random.seed(1)
    deep_eye.configure(sample_rate=0.5)

    @trace
    def child():
        pass

    @trace
    def root():
        child()
        with span("block"):
            child()

    for _ in range(100):
        root()
    runs = list_runs()
    assert 20 < len(runs) < 80
    assert all(len(r.spans) == 4 for r in runs)  # never half a run


def test_sample_rate_zero_records_nothing_and_bad_values_are_safe(trace_dir, monkeypatch):
    deep_eye.configure(sample_rate=0)
    trace(lambda: None)()
    assert not trace_dir.exists()
    with pytest.raises(ValueError):
        deep_eye.configure(sample_rate=2)

    from deep_eye import config

    monkeypatch.setattr(config, "_sample_rate", None)
    monkeypatch.setenv("DEEP_EYE_SAMPLE_RATE", "oops")
    trace(lambda: None)()  # a typo in the variable means "record everything"
    assert len(list_runs()) == 1


# --- background writer ------------------------------------------------------------------

def test_files_are_written_off_the_calling_thread(monkeypatch):
    threads = []
    real = storage._write_batch
    monkeypatch.setattr(storage, "_write_batch", lambda batch: (threads.append(threading.current_thread().name), real(batch)))

    trace(lambda: None)()
    deep_eye.flush()
    assert threads and set(threads) == {"deep-eye-writer"}

    threads.clear()
    deep_eye.configure(sync_writes=True)
    trace(lambda: None)()
    assert threads == [threading.current_thread().name] * 3  # header, start and end, written immediately


def test_queued_events_are_written_at_exit(tmp_path):
    script = """
        from deep_eye import trace

        @trace
        def step(i):
            return i

        for i in range(200):
            step(i)
    """
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60, env=os.environ)
    assert result.returncode == 0, result.stderr
    runs = list_runs(tmp_path / ".deep-eye")
    assert len(runs) == 200 and all(r.status == "ok" for r in runs)


def test_full_queue_drops_events_with_one_warning(monkeypatch, caplog):
    trace(lambda: None)()  # the writer thread is running, on the real queue
    monkeypatch.setattr(storage, "_queue", queue.Queue(2))  # nobody drains this one
    monkeypatch.setattr(storage, "_warned", False)
    monkeypatch.setattr(storage, "flush", lambda timeout=None: True)

    @trace
    def f():
        return "still works"

    assert all(f() == "still works" for _ in range(5))
    assert sum("dropping new ones" in r.message for r in caplog.records) == 1


# --- format version and export ----------------------------------------------------------

def test_run_header_carries_the_format_version(trace_dir):
    with span("a"):
        with span("b"):
            pass
    header, *events = _events(trace_dir)
    assert header["event"] == "run" and header["schema_version"] == storage.SCHEMA_VERSION
    assert [e["event"] for e in events] == ["start", "start", "end", "end"]


def _sample_runs():
    @trace(kind="llm")
    def llm(prompt):
        return f"answer to {prompt}"

    @trace(kind="tool")
    def tool():
        raise ValueError("bad")

    @trace(name="agent", kind="agent")
    def agent(q):
        llm(q)
        try:
            tool()
        except ValueError:
            pass

    agent("q1")
    agent("q2")


def test_export_writes_one_flat_record_per_span(tmp_path):
    _sample_runs()
    out = tmp_path / "out.jsonl"
    main(["export", "-o", str(out)])
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert records[0]["depth"] == 0  # each run starts with its root
    # the two runs can start in the same clock tick (~16ms on Windows), so don't rely on their order
    first = next(r for r in records if r["depth"] == 0 and r["input"] == {"q": "q1"})
    records = [r for r in records if r["run_id"] == first["run_id"]]
    assert first["v"] == storage.SCHEMA_VERSION and first["run_name"] == "agent"
    assert (first["depth"], first["kind"], first["input"]) == (0, "agent", {"q": "q1"})
    tool_rec = next(r for r in records if r["kind"] == "tool")
    assert tool_rec["status"] == "error" and tool_rec["error"] == "ValueError: bad"
    assert tool_rec["parent_id"] == first["span_id"] and tool_rec["depth"] == 1


def test_export_filters_by_kind_and_run_to_stdout(capsysbinary):
    _sample_runs()
    run_id = next(r.run_id for r in list_runs() if r.root.input == {"q": "q1"})
    main(["export", run_id, "--kind", "llm"])
    lines = capsysbinary.readouterr().out.decode("utf-8").splitlines()
    (record,) = [json.loads(line) for line in lines]
    assert (record["kind"], record["input"], record["output"]) == ("llm", {"prompt": "q1"}, "answer to q1")


def test_export_filters_by_run_status(tmp_path):
    _sample_runs()
    out = tmp_path / "out.jsonl"
    main(["export", "--status", "ok", "-o", str(out)])
    assert out.read_text(encoding="utf-8") == ""  # both runs contain a failed tool call
