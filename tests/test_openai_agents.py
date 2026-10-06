"""The OpenAI Agents SDK adapter, fed by the real SDK with a fake model: no network, no API key."""

import asyncio

import pytest

agents = pytest.importorskip("agents")

from agents import Agent, ModelResponse, Runner, Usage, function_tool  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.tracing import (  # noqa: E402
    custom_span,
    function_span,
    generation_span,
    guardrail_span,
    handoff_span,
    set_trace_processors,
)
from agents.tracing import trace as sdk_trace  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import deep_eye  # noqa: E402
from deep_eye import trace  # noqa: E402
from deep_eye.integrations import openai_agents  # noqa: E402
from deep_eye.integrations.openai_agents import DeepEyeProcessor  # noqa: E402
from deep_eye.schema import KINDS, validate_file  # noqa: E402
from deep_eye.storage import list_runs  # noqa: E402


@pytest.fixture(autouse=True)
def processor(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)
    processor = DeepEyeProcessor()
    set_trace_processors([processor])  # only deep-eye: nothing is sent to OpenAI
    yield processor
    set_trace_processors([])
    openai_agents._open.clear()
    openai_agents._merged.clear()


@trace
def lookup(city):
    return f"sunny in {city}"


@function_tool
def get_weather(city: str) -> str:
    """Weather for a city."""
    return lookup(city)


class FakeModel(Model):
    """Asks for the weather tool once, then answers. Records a generation span like real models do."""

    def __init__(self):
        self.calls = 0

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, **kwargs):
        self.calls += 1
        with generation_span(model="fake-model", input=[{"role": "user", "content": str(input)}]) as span:
            if self.calls == 1:
                output = [ResponseFunctionToolCall(type="function_call", id="fc_1", call_id="call_1",
                                                   name="get_weather", arguments='{"city": "Cairo"}')]
            else:
                output = [ResponseOutputMessage(
                    id="msg_1", type="message", role="assistant", status="completed",
                    content=[ResponseOutputText(type="output_text", text="It's sunny.", annotations=[])])]
            span.span_data.output = [item.model_dump() for item in output]
            span.span_data.usage = {"input_tokens": 10, "output_tokens": 5}
        return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=10, output_tokens=5,
                                                       total_tokens=15), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def _tree(run):
    """[(depth, kind, name)] in start order."""
    depth = {None: -1}
    out = []
    for s in sorted(run.spans, key=lambda s: s.start):
        depth[s.span_id] = depth[s.parent_id] + 1
        out.append((depth[s.span_id], s.display_kind, s.name))
    return out


def test_agent_run_with_a_tool_call():
    agent = Agent(name="weather-bot", instructions="Help.", tools=[get_weather], model=FakeModel())
    result = Runner.run_sync(agent, "Weather in Cairo?")
    assert result.final_output == "It's sunny."

    (run,) = list_runs()
    tree = _tree(run)
    names = [(kind, name) for _, kind, name in tree]
    assert tree[0] == (0, "chain", "Agent workflow")  # the SDK trace (and its task) is the run's root
    assert tree[1] == (1, "agent", "weather-bot")
    assert ("agent", "weather-bot") in names
    assert names.count(("llm", "fake-model")) == 2
    assert ("tool", "get_weather") in names

    spans = {s.name: s for s in run.spans}
    tool = spans["get_weather"]
    assert tool.input == {"city": "Cairo"} and tool.output == "sunny in Cairo"
    assert spans["lookup"].parent_id == tool.span_id  # @trace inside a tool nests under it
    assert run.tokens == 30  # two LLM calls; task / turn spans don't count them again
    assert run.status == "ok"


class BrokenModel(FakeModel):
    async def get_response(self, *args, **kwargs):
        raise ConnectionError("model unreachable")


def test_a_failing_run_is_recorded_and_the_error_reaches_the_caller():
    with pytest.raises(ConnectionError, match="model unreachable"):
        Runner.run_sync(Agent(name="bot", model=BrokenModel()), "hi")
    (run,) = list_runs()
    assert run.status == "error"
    assert validate_file(run.path) == []


def test_every_span_type_is_mapped():
    with sdk_trace("workflow", group_id="g1"):
        with custom_span("prep", data={"step": 1}):
            pass
        with function_span("search", input='{"q": "x"}') as fs:
            fs.span_data.output = "found"
        with generation_span(model="gpt-x", input=[{"role": "user", "content": "hi"}],
                             usage={"input_tokens": 3, "output_tokens": 4}) as gs:
            gs.span_data.output = [{"role": "assistant", "content": "hello"}]
        with handoff_span(from_agent="a", to_agent="b"):
            pass
        with guardrail_span("no-pii", triggered=True):
            pass

    (run,) = list_runs()
    spans = {s.name: s for s in run.spans}
    assert run.root.name == "workflow" and run.root.attrs["group_id"] == "g1"
    assert (spans["prep"].kind, spans["prep"].attrs) == ("function", {"step": 1})
    assert (spans["search"].kind, spans["search"].input, spans["search"].output) == ("tool", {"q": "x"}, "found")
    llm = spans["gpt-x"]
    assert (llm.kind, llm.tokens, llm.output) == ("llm", 7, [{"role": "assistant", "content": "hello"}])
    assert (spans["handoff → b"].kind, spans["handoff → b"].display_kind) == ("other", "handoff")
    assert spans["no-pii"].display_kind == "guardrail" and spans["no-pii"].attrs["triggered"] is True


def test_errors_are_recorded():
    with sdk_trace("workflow"):
        with function_span("flaky") as span:
            span.set_error({"message": "Tool failed", "data": {"code": 500}})
    span = next(s for s in list_runs()[0].spans if s.name == "flaky")
    assert span.error == "Tool failed" and span.attrs["error_data"] == {"code": 500}
    assert list_runs()[0].status == "error"


def test_sdk_trace_inside_a_traced_function_nests_under_it():
    @trace(kind="agent")
    def my_app():
        with sdk_trace("inner"):
            with custom_span("work"):
                pass

    my_app()
    (run,) = list_runs()
    assert [(d, n) for d, _, n in _tree(run)] == [(0, "my_app"), (1, "inner"), (2, "work")]


def test_contract_output_follows_the_schema():
    agent = Agent(name="bot", tools=[get_weather], model=FakeModel())
    asyncio.run(Runner.run(agent, "hi"))
    with sdk_trace("all-types"):
        with handoff_span(from_agent="a", to_agent="b"):
            pass
        with guardrail_span("g"):
            pass
    runs = list_runs()
    assert len(runs) == 2
    for run in runs:
        assert validate_file(run.path) == []
        assert all(s.kind in KINDS for s in run.spans)


def test_processor_never_raises_on_odd_input(processor):
    class Odd:
        def __getattr__(self, name):
            raise RuntimeError("odd object")

    for method in (processor.on_trace_start, processor.on_trace_end,
                   processor.on_span_start, processor.on_span_end):
        method(Odd())
        method(None)


def test_install_is_idempotent_and_exclusive_keeps_traces_local(monkeypatch):
    monkeypatch.setattr(openai_agents, "_installed", None)
    first = openai_agents.install(exclusive=True)
    assert openai_agents.install() is first
    with sdk_trace("local"):
        pass
    assert [r.name for r in list_runs()] == ["local"]
