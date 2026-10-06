"""Regression tests for the edge cases listed in the README's Limitations section."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import deep_eye
from deep_eye import propagate, span, trace
from deep_eye.cli import main
from deep_eye.storage import list_runs


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)
    return tmp_path


# --- threads ----------------------------------------------------------------------

def test_propagate_nests_thread_pool_work_in_one_run():
    @trace
    def work(i):
        return i * 2

    @trace
    def fan_out():
        with ThreadPoolExecutor(max_workers=3) as ex:
            return list(ex.map(propagate(work), range(5)))

    assert fan_out() == [0, 2, 4, 6, 8]
    (run,) = list_runs()
    assert [c.name for c in run.children_of(run.root.span_id)] == ["work"] * 5


def test_propagate_works_with_plain_threads():
    @trace
    def work():
        return "done"

    with span("parent"):
        t = threading.Thread(target=propagate(work))
        t.start()
        t.join()

    (run,) = list_runs()
    assert len(run.spans) == 2


# --- generators -------------------------------------------------------------------

def test_generator_span_covers_iteration_and_records_items():
    @trace
    def child():
        return "c"

    @trace
    def gen(n):
        for i in range(n):
            child()
            yield i
        return "finished"

    assert list(gen(3)) == [0, 1, 2]
    (run,) = list_runs()
    assert run.root.output == [0, 1, 2] and run.root.end is not None
    assert [c.name for c in run.children_of(run.root.span_id)] == ["child"] * 3


def test_generator_closed_early_is_not_an_error():
    @trace
    def gen():
        yield from range(100)

    g = gen()
    next(g), next(g)
    g.close()
    (run,) = list_runs()
    assert run.status == "ok" and run.root.output == [0, 1]


def test_generator_error_is_recorded():
    @trace
    def gen():
        yield 1
        raise ValueError("bad")

    with pytest.raises(ValueError):
        list(gen())
    (run,) = list_runs()
    assert run.root.error == "ValueError: bad" and run.root.output == [1]


def test_generator_send_and_throw_are_forwarded():
    @trace
    def echo():
        received = []
        try:
            while True:
                received.append((yield len(received)))
        except KeyError:
            yield "caught"

    g = echo()
    assert next(g) == 0
    assert g.send("a") == 1
    assert g.throw(KeyError("x")) == "caught"
    g.close()


def test_async_generator_is_traced():
    @trace
    async def agen():
        for i in range(3):
            await asyncio.sleep(0)
            yield i

    async def consume():
        return [x async for x in agen()]

    assert asyncio.run(consume()) == [0, 1, 2]
    (run,) = list_runs()
    assert run.root.output == [0, 1, 2] and run.status == "ok"


# --- what gets stored -------------------------------------------------------------

def test_secrets_are_redacted_by_key_and_by_pattern(trace_dir):
    @trace
    def call(api_key, headers, prompt):
        return "ok"

    call("hunter2", {"Authorization": "abc"}, "my key is sk-ant-0123456789abcdefghij")
    deep_eye.flush()
    text = next(trace_dir.glob("*.jsonl")).read_text()
    assert "hunter2" not in text and "abc" not in text and "sk-ant-0123456789" not in text
    root = list_runs()[0].root
    assert root.input["api_key"] == "[redacted]" and root.input["prompt"] == "my key is [redacted]"


def test_redaction_can_be_extended_or_disabled():
    @trace
    def call(ssn):
        return "ok"

    deep_eye.configure(redact_keys=["ssn"])
    call("123-45-6789")
    assert list_runs()[0].root.input["ssn"] == "[redacted]"

    deep_eye.configure(redact=False)
    call("123-45-6789")
    assert {r.root.input["ssn"] for r in list_runs()} == {"[redacted]", "123-45-6789"}


def test_long_lists_are_capped():
    deep_eye.configure(max_items=3)

    @trace
    def big():
        return ["x"] * 10

    big()
    assert list_runs()[0].root.output == ["x", "x", "x", "... [+7 items]"]


def test_methods_do_not_record_self():
    class Agent:
        @trace
        def act(self, x):
            return x

    Agent().act(1)
    assert list_runs()[0].root.input == {"x": 1}


# --- reading traces ----------------------------------------------------------------

def test_malformed_lines_are_skipped_not_fatal(trace_dir):
    @trace
    def ok():
        return 1

    ok()
    (trace_dir / "zz_bad.jsonl").write_text(
        '{"event":"start"}\n[1,2]\n{"event":"start","span_id":"x"}\nnot json\n'
    )
    runs = list_runs()
    assert len(runs) == 2 and any(r.root and r.root.name == "ok" for r in runs)


def test_list_does_not_create_trace_folder(tmp_path):
    missing = tmp_path / "nothing-here"
    main(["--dir", str(missing), "list"])
    assert not missing.exists()


def test_show_refuses_ambiguous_prefix(capsys):
    @trace
    def a():
        pass

    a(), a()
    main(["show", list_runs()[0].run_id[:4]])  # both runs share the year prefix
    assert "matches 2 runs" in capsys.readouterr().out


def test_clear_older_than_keeps_recent_files(trace_dir):
    @trace
    def a():
        pass

    a()
    main(["clear", str(trace_dir), "-y", "--older-than", "1"])
    assert len(list(trace_dir.glob("*.jsonl"))) == 1
