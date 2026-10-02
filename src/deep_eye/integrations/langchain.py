"""LangChain / LangGraph adapter.

    from deep_eye.integrations.langchain import DeepEyeHandler
    agent.invoke(inputs, config={"callbacks": [DeepEyeHandler()]})

Every chain, LLM call, tool call and retriever call becomes a span. Functions decorated with
@trace that run inside a tool are nested under that tool's span.
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.config import var_child_runnable_config

from .. import tracer
from ..models import Span

_open_spans: dict[UUID, Span] = {}  # LangChain run_id -> span


def _langchain_parent() -> Span | None:
    """Find the span of the LangChain run we are currently executing inside."""
    config = var_child_runnable_config.get()
    manager = (config or {}).get("callbacks")
    return _open_spans.get(getattr(manager, "parent_run_id", None))


tracer.register_parent_resolver(_langchain_parent)


def _name(serialized: dict | None, kwargs: dict, default: str) -> str:
    if kwargs.get("name"):
        return kwargs["name"]
    serialized = serialized or {}
    return serialized.get("name") or (serialized.get("id") or [default])[-1]


def _llm_output(response: Any) -> tuple[Any, dict | None]:
    """Split an LLMResult into (output, token usage); providers report usage, no tokenizer needed."""
    try:
        message = response.generations[0][0].message
    except (AttributeError, IndexError):
        return response, None
    output = {"content": message.content, "tool_calls": getattr(message, "tool_calls", None)}
    return output, getattr(message, "usage_metadata", None)


class DeepEyeHandler(BaseCallbackHandler):
    """Records LangChain runs as deep-eye spans."""

    run_inline = True  # run callbacks in the caller's context so ordering is exact

    def _start(self, run_id: UUID, parent_run_id: UUID | None, name: str, kind: str, input: Any):
        parent = _open_spans.get(parent_run_id)
        if parent is not None:
            span = tracer.begin_span(name, kind, input, parent=parent)
        else:  # top-level run: nest under an ambient @trace span if there is one
            span = tracer.begin_span(name, kind, input)
        _open_spans[run_id] = span

    def _end(self, run_id: UUID, output: Any = None, error: BaseException | None = None,
             usage: dict | None = None):
        span = _open_spans.pop(run_id, None)
        if span is not None:
            tracer.finish_span(span, output, error, usage)

    # --- chains / graph nodes -------------------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        kind = "chain" if parent_run_id else "agent"
        self._start(run_id, parent_run_id, _name(serialized, kwargs, "chain"), kind, inputs)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id, outputs)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    # --- LLM calls --------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._start(run_id, parent_run_id, _name(serialized, kwargs, "llm"), "llm", messages)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._start(run_id, parent_run_id, _name(serialized, kwargs, "llm"), "llm", prompts)

    def on_llm_end(self, response, *, run_id, **kwargs):
        output, usage = _llm_output(response)
        self._end(run_id, output, usage=usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    # --- tools --------------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        self._start(run_id, parent_run_id, _name(serialized, kwargs, "tool"), "tool",
                    inputs if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id, getattr(output, "content", output))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    # --- retrievers -------------------------------------------------------------
    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
        self._start(run_id, parent_run_id, _name(serialized, kwargs, "retriever"), "retriever", query)

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end(run_id, [{"content": d.page_content, "metadata": d.metadata} for d in documents])

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)
