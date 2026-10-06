"""The OpenTelemetry adapter, fed by the real OpenTelemetry SDK with spans shaped like the ones
GenAI, OpenInference and OpenLLMetry instrumentations emit. No network, no API key."""

import json

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.trace import SpanKind, Status, StatusCode  # noqa: E402

import deep_eye  # noqa: E402
from deep_eye import trace  # noqa: E402
from deep_eye.integrations import otel  # noqa: E402
from deep_eye.schema import KINDS, validate_file  # noqa: E402
from deep_eye.storage import list_runs  # noqa: E402


@pytest.fixture(autouse=True)
def tracer(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)
    provider = TracerProvider()
    otel.install(provider)
    yield provider.get_tracer("test")
    provider.shutdown()
    otel._open.clear()


def _spans():
    (run,) = list_runs()
    return run, {s.name: s for s in run.spans}


def test_gen_ai_semantic_conventions(tracer):
    with tracer.start_as_current_span("invoke_agent planner", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"}):
        with tracer.start_as_current_span("chat gpt-4o", kind=SpanKind.CLIENT, attributes={
                "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o"}) as llm:
            # instrumentations often only know the result at the end
            llm.set_attribute("gen_ai.input.messages", json.dumps([{"role": "user", "content": "hi"}]))
            llm.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "content": "yo"}]))
            llm.set_attribute("gen_ai.usage.input_tokens", 12)
            llm.set_attribute("gen_ai.usage.output_tokens", 3)
        with tracer.start_as_current_span("execute_tool search", attributes={
                "gen_ai.operation.name": "execute_tool", "gen_ai.tool.call.arguments": '{"q": "x"}'}) as tool:
            tool.set_attribute("gen_ai.tool.call.result", "found it")
        with tracer.start_as_current_span("embeddings", attributes={"gen_ai.operation.name": "embeddings"}):
            pass

    run, spans = _spans()
    assert run.root.name == "invoke_agent planner" and run.root.kind == "agent"
    llm = spans["chat gpt-4o"]
    assert llm.kind == "llm" and llm.parent_id == run.root.span_id
    assert llm.input == [{"role": "user", "content": "hi"}]
    assert llm.output == [{"role": "assistant", "content": "yo"}]
    assert llm.usage == {"input_tokens": 12, "output_tokens": 3} and llm.tokens == 15
    assert llm.attrs["gen_ai.request.model"] == "gpt-4o" and llm.attrs["otel.kind"] == "client"
    assert len(llm.attrs["otel.span_id"]) == 16 and len(llm.attrs["otel.trace_id"]) == 32
    tool = spans["execute_tool search"]
    assert (tool.kind, tool.input, tool.output) == ("tool", {"q": "x"}, "found it")
    assert spans["embeddings"].kind == "embedding"


def test_openinference_conventions(tracer):
    with tracer.start_as_current_span("query", attributes={
            "openinference.span.kind": "CHAIN", "input.value": "what is up?"}) as chain:
        with tracer.start_as_current_span("retrieve", attributes={"openinference.span.kind": "RETRIEVER"}):
            pass
        with tracer.start_as_current_span("rerank", attributes={"openinference.span.kind": "RERANKER"}):
            pass
        with tracer.start_as_current_span("llm", attributes={
                "openinference.span.kind": "LLM", "llm.token_count.prompt": 7,
                "llm.token_count.completion": 2, "llm.token_count.total": 9}):
            pass
        chain.set_attribute("output.value", '{"answer": "not much"}')

    run, spans = _spans()
    assert (run.root.kind, run.root.input, run.root.output) == ("chain", "what is up?", {"answer": "not much"})
    assert spans["retrieve"].kind == "retriever"
    assert (spans["rerank"].kind, spans["rerank"].display_kind) == ("other", "reranker")
    assert spans["llm"].usage == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}


def test_openllmetry_flat_messages(tracer):
    with tracer.start_as_current_span("openai.chat", attributes={
            "llm.request.type": "chat", "gen_ai.system": "openai",
            "gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": "hello",
            "gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "hi there",
            "gen_ai.usage.prompt_tokens": 4, "gen_ai.usage.completion_tokens": 2}):
        pass
    run, spans = _spans()
    span = spans["openai.chat"]
    assert span.kind == "llm"
    assert span.input == [{"role": "user", "content": "hello"}]
    assert span.output == [{"role": "assistant", "content": "hi there"}]
    assert span.tokens == 6
    assert not any(k.startswith("gen_ai.prompt") for k in span.attrs)


def test_plain_spans_and_errors(tracer):
    with pytest.raises(ValueError):
        with tracer.start_as_current_span("GET /weather", kind=SpanKind.CLIENT):
            raise ValueError("timeout")
    with tracer.start_as_current_span("flagged") as span:
        span.set_status(Status(StatusCode.ERROR, "bad answer"))

    runs = {r.name: r for r in list_runs()}
    http = runs["GET /weather"].root
    assert (http.kind, http.error, runs["GET /weather"].status) == ("function", "ValueError: timeout", "error")
    assert runs["flagged"].root.error == "bad answer"


def test_traced_functions_nest_both_ways(tracer):
    @trace
    def inside_otel():
        return 1

    @trace(kind="agent")
    def my_app():
        with tracer.start_as_current_span("otel-step"):
            inside_otel()

    my_app()
    (run,) = list_runs()
    by_name = {s.name: s for s in run.spans}
    assert by_name["otel-step"].parent_id == by_name["my_app"].span_id
    assert by_name["inside_otel"].parent_id == by_name["otel-step"].span_id


def test_otel_timestamps_are_kept(tracer):
    span = tracer.start_span("timed", start_time=1_000_000_000_000_000_000)
    span.end(end_time=1_000_000_002_500_000_000)
    run, spans = _spans()
    assert spans["timed"].duration == pytest.approx(2.5)


def test_contract_output_follows_the_schema(tracer):
    conventions = [
        {"gen_ai.operation.name": "invoke_agent"},
        {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 1, "gen_ai.usage.output_tokens": 1},
        {"gen_ai.operation.name": "some_future_operation"},
        {"openinference.span.kind": "EVALUATOR", "input.value": "{not json"},
        {"traceloop.span.kind": "workflow", "traceloop.entity.input": '{"a": 1}'},
        {"gen_ai.prompt.0.content": "hi", "gen_ai.system": "x"},
        {},
    ]
    with tracer.start_as_current_span("root"):
        for i, attributes in enumerate(conventions):
            with tracer.start_as_current_span(f"span-{i}", attributes=attributes):
                pass
    (run,) = list_runs()
    assert len(run.spans) == len(conventions) + 1
    assert validate_file(run.path) == []
    assert all(s.kind in KINDS for s in run.spans)


def test_processor_never_raises_on_odd_input():
    processor = otel.DeepEyeSpanProcessor()
    processor.on_start(object())
    processor.on_end(None)


def test_install_is_idempotent():
    provider = TracerProvider()
    assert otel.install(provider) is otel.install(provider)


def test_install_works_on_new_providers_after_old_ones_are_gone(tmp_path):
    import gc

    for i in range(30):  # a new provider can reuse a collected one's id()
        deep_eye.configure(trace_dir=tmp_path / str(i))
        provider = TracerProvider()
        otel.install(provider)
        with provider.get_tracer("t").start_as_current_span("x"):
            pass
        assert len(list_runs()) == 1
        provider.shutdown()
        del provider
        gc.collect()
