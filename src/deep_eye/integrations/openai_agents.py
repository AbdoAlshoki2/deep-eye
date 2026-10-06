"""OpenAI Agents SDK adapter.

    from deep_eye.integrations.openai_agents import install
    install()  # once, before running agents

Each Agents SDK trace becomes a deep-eye run. Agent, tool (function), LLM (generation and
response), handoff and guardrail spans keep their nesting, timing and token usage. Functions
decorated with @trace that run inside a tool are nested under that tool's span.

install() adds deep-eye next to the SDK's other trace processors, so traces still go to the
OpenAI dashboard as well when that's set up. `install(exclusive=True)` keeps them local only.
"""

from datetime import datetime
from typing import Any

from .. import tracer
from ..models import Span
from . import maybe_json, missing_framework, normalize_usage

try:
    from agents.tracing import (
        TracingProcessor,
        add_trace_processor,
        get_current_span,
        get_current_trace,
        set_trace_processors,
    )
except ImportError as exc:
    raise missing_framework("openai_agents", "openai-agents", "openai-agents") from exc

_open: dict[str, Span] = {}  # Agents SDK trace id / span id -> deep-eye span
_merged: set[str] = set()  # task spans recorded as their trace's span
_installed: "DeepEyeProcessor | None" = None

# SDK span types whose deep-eye kind has another name. The rest (handoff, guardrail,
# mcp_tools, ...) are stored as "other", keeping the SDK's name for them.
_KINDS = {
    "agent": "agent",
    "function": "tool",
    "generation": "llm",
    "response": "llm",
    "transcription": "llm",
    "speech": "llm",
    "task": "chain",
    "turn": "chain",
    "custom": "function",
}
_LLM_TYPES = {"generation", "response", "transcription", "speech"}


def _sdk_parent() -> Span | None:
    """The span of the SDK span or trace we are running inside, for @trace functions."""
    current = get_current_span()
    if current is not None and getattr(current, "span_id", None) in _open:
        return _open[current.span_id]
    trace = get_current_trace()
    return _open.get(getattr(trace, "trace_id", None))


tracer.register_parent_resolver(_sdk_parent)


def _time(iso: str | None) -> float | None:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except (TypeError, ValueError):
        return None


def _type(data: Any) -> str:
    return str(getattr(data, "type", None) or "custom")


def _name(data: Any) -> str:
    kind = _type(data)
    if kind == "handoff":
        return f"handoff → {getattr(data, 'to_agent', None) or '?'}"
    if kind == "turn":
        return f"turn {getattr(data, 'turn', '?')}"
    if kind == "response":
        response = getattr(data, "response", None)
        return getattr(response, "model", None) or "response"
    return str(getattr(data, "name", None) or getattr(data, "model", None) or kind)


def _input(data: Any) -> Any:
    return maybe_json(getattr(data, "input", None))


def _output(data: Any) -> Any:
    if _type(data) == "response":
        response = getattr(data, "response", None)
        return getattr(response, "output", None)
    return maybe_json(getattr(data, "output", None))


def _usage(data: Any) -> dict | None:
    if _type(data) not in _LLM_TYPES:
        return None  # task / turn spans repeat their children's tokens: don't count them twice
    usage = getattr(data, "usage", None)
    if usage is None:
        usage = getattr(getattr(data, "response", None), "usage", None)
    return normalize_usage(usage)


def _attrs(data: Any) -> dict:
    """What the SDK exports about the span, minus what has its own field."""
    try:
        exported = data.export() or {}
    except Exception:
        exported = {}
    if exported.get("type") == "custom" and isinstance(exported.get("data"), dict):
        exported = {**exported["data"], "name": exported.get("name")}  # task, turn, custom
    return {k: v for k, v in exported.items()
            if k not in ("type", "name", "input", "output", "usage") and v is not None}


class DeepEyeProcessor(TracingProcessor):
    """An Agents SDK trace processor that records traces as deep-eye runs.
    Its methods never raise."""

    @tracer.never_raises
    def on_trace_start(self, trace: Any) -> None:
        attrs = {"trace_id": trace.trace_id}
        try:
            exported = trace.export() or {}
        except Exception:
            exported = {}
        attrs.update({k: exported[k] for k in ("group_id", "metadata") if exported.get(k)})
        # nests under an enclosing @trace / span() block if there is one, else starts a run
        _open[trace.trace_id] = tracer.start_span(trace.name, "chain", attrs=attrs)

    @tracer.never_raises
    def on_trace_end(self, trace: Any) -> None:
        span = _open.pop(trace.trace_id, None)
        if span is not None:
            tracer.end_span(span)

    @tracer.never_raises
    def on_span_start(self, span: Any) -> None:
        data = span.span_data
        parent = _open.get(span.parent_id) or _open.get(span.trace_id)
        kind = _type(data)
        if kind == "task" and span.parent_id is None and parent is not None:
            # the Runner run that fills the whole trace: one span for both is enough
            _open[span.span_id] = parent
            _merged.add(span.span_id)
            return
        _open[span.span_id] = tracer.start_span(
            _name(data), _KINDS.get(kind, kind), _input(data),
            **({"parent": parent} if parent is not None else {}),
            attrs=_attrs(data) or None,
            start_time=_time(span.started_at),
        )

    @tracer.never_raises
    def on_span_end(self, span: Any) -> None:
        handle = _open.pop(span.span_id, None)
        if handle is None:
            return
        error = span.error
        if span.span_id in _merged:
            _merged.discard(span.span_id)
            if error:  # the trace ends without one: keep it on the run's root
                handle.error = error.get("message") or "error"
            return
        data = span.span_data
        attrs = _attrs(data)
        if error and error.get("data"):
            attrs["error_data"] = error["data"]
        name, input = _name(data), _input(data)
        tracer.end_span(
            handle,
            _output(data),
            error=(error.get("message") or "error") if error else None,
            usage=_usage(data),
            # details the SDK only fills in once the work is done
            **({"input": input} if handle.input is None and input is not None else {}),
            name=name if name != handle.name else None,
            attrs=attrs or None,
            end_time=_time(span.ended_at),
        )

    def shutdown(self) -> None:
        tracer.flush()

    def force_flush(self) -> None:
        tracer.flush()


def install(exclusive: bool = False) -> DeepEyeProcessor:
    """Record every Agents SDK trace with deep-eye. Calling it again does nothing.

    exclusive=True replaces the SDK's other processors, so traces stay on this machine
    instead of also being sent to the OpenAI dashboard.
    """
    global _installed
    if _installed is None:
        _installed = DeepEyeProcessor()
        if not exclusive:
            add_trace_processor(_installed)
    if exclusive:
        set_trace_processors([_installed])
    return _installed
