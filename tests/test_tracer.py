import asyncio

import pytest

import deep_eye
from deep_eye import span, trace
from deep_eye.storage import list_runs


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)
    return tmp_path


def test_nested_functions_form_a_tree():
    @trace
    def inner(x):
        return x * 2

    @trace(name="outer", kind="agent")
    def outer(x):
        return inner(x) + inner(x + 1)

    assert outer(1) == 6

    (run,) = list_runs()
    root = run.root
    assert (root.name, root.kind, root.input, root.output) == ("outer", "agent", {"x": 1}, 6)
    children = run.children_of(root.span_id)
    assert [(c.name, c.input, c.output) for c in children] == [
        ("inner", {"x": 1}, 2),
        ("inner", {"x": 2}, 4),
    ]
    assert run.status == "ok"


def test_exceptions_are_recorded_and_reraised():
    @trace
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()

    (run,) = list_runs()
    assert run.root.error == "ValueError: nope"
    assert run.status == "error"


def test_span_context_manager_records_output():
    with span("step", kind="tool", input={"q": 1}) as s:
        s.output = "done"

    (run,) = list_runs()
    assert run.root.output == "done" and run.root.kind == "tool"


def test_async_functions_and_separate_runs():
    @trace
    async def child():
        return "c"

    @trace
    async def parent():
        return await child()

    asyncio.run(parent())
    asyncio.run(parent())

    runs = list_runs()
    assert len(runs) == 2 and all(len(r.spans) == 2 for r in runs)


def test_unserializable_values_do_not_break_tracing():
    @trace
    def f(obj):
        return obj

    class Weird:
        pass

    f(Weird())
    (run,) = list_runs()
    assert "Weird" in run.root.output


def test_unfinished_span_shows_as_running(trace_dir):
    from deep_eye.tracer import start_span

    start_span("hanging")
    (run,) = list_runs()
    assert run.status == "running" and run.root.end is None


def test_token_usage_is_stored_and_summed():
    with span("run", kind="agent"):
        with span("llm-1", kind="llm") as s:
            s.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        with span("llm-2", kind="llm") as s:
            s.usage = {"input_tokens": 3, "output_tokens": 2}  # total derived

    (run,) = list_runs()
    assert run.tokens == 20 and run.root.tokens == 0
