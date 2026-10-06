"""OpenTelemetry adapter: records OpenTelemetry spans as deep-eye spans.

    from deep_eye.integrations.otel import install
    install()  # once, at startup

Any library that emits OpenTelemetry spans then shows up in deep-eye, with LLM, tool, agent,
retriever and embedding kinds, inputs, outputs and token usage read from the common
conventions:

- OpenTelemetry GenAI semantic conventions (gen_ai.*)
- OpenInference (openinference.span.kind, input.value, llm.token_count.*), used by the
  Arize Phoenix instrumentations for LlamaIndex, CrewAI, DSPy, Haystack, ...
- OpenLLMetry / Traceloop (traceloop.span.kind, traceloop.entity.*, gen_ai.prompt.N.*)

Other spans (HTTP calls, database queries, ...) are kept as "function" spans. Every other
attribute is kept in the span's attrs. Functions decorated with @trace that run inside an
OpenTelemetry span are nested under it.
"""

import logging
import re
import weakref
from typing import Any

from .. import tracer
from ..models import Span
from . import maybe_json, missing_framework, normalize_usage

log = logging.getLogger("deep_eye")

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError as exc:
    raise missing_framework("otel", "opentelemetry-sdk", "otel") from exc

_open: dict[tuple[int, int], Span] = {}  # (trace id, span id) -> deep-eye span
_installed: "weakref.WeakKeyDictionary[Any, DeepEyeSpanProcessor]" = weakref.WeakKeyDictionary()  # provider -> processor

_GEN_AI_OPERATIONS = {
    "chat": "llm", "text_completion": "llm", "generate_content": "llm",
    "embeddings": "embedding", "execute_tool": "tool",
    "invoke_agent": "agent", "create_agent": "agent",
}
_OPENINFERENCE_KINDS = {
    "LLM": "llm", "TOOL": "tool", "AGENT": "agent", "CHAIN": "chain",
    "RETRIEVER": "retriever", "EMBEDDING": "embedding",
}
_TRACELOOP_KINDS = {"workflow": "chain", "task": "function", "agent": "agent", "tool": "tool"}

_INPUT_KEYS = ("input.value", "traceloop.entity.input", "gen_ai.input.messages",
               "gen_ai.tool.call.arguments", "gen_ai.prompt")
_OUTPUT_KEYS = ("output.value", "traceloop.entity.output", "gen_ai.output.messages",
                "gen_ai.tool.call.result", "gen_ai.completion")
_USAGE_KEYS = {
    "input_tokens": ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.token_count.prompt"),
    "output_tokens": ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens",
                      "llm.token_count.completion"),
    "total_tokens": ("llm.token_count.total", "llm.usage.total_tokens"),
}
_INDEXED = re.compile(r"^gen_ai\.(prompt|completion)\.(\d+)\.(.+)$")  # OpenLLMetry's flat messages


def _key(context: Any) -> tuple[int, int]:
    return context.trace_id, context.span_id


def _otel_parent() -> Span | None:
    """The span of the OpenTelemetry span we are running inside, for @trace functions."""
    context = otel_trace.get_current_span().get_span_context()
    return _open.get(_key(context)) if context.is_valid else None


tracer.register_parent_resolver(_otel_parent)


def _kind(attrs: dict) -> str:
    operation = attrs.get("gen_ai.operation.name")
    if operation:
        return _GEN_AI_OPERATIONS.get(operation, str(operation))
    kind = attrs.get("openinference.span.kind")
    if kind:
        return _OPENINFERENCE_KINDS.get(str(kind).upper(), str(kind).lower())
    kind = attrs.get("traceloop.span.kind")
    if kind:
        return _TRACELOOP_KINDS.get(kind, str(kind))
    if "gen_ai.system" in attrs or "gen_ai.request.model" in attrs or "llm.request.type" in attrs:
        return "llm"
    return "function"


def _take(attrs: dict, keys: tuple[str, ...]) -> Any:
    """The value of the first key present, removed from attrs with the other keys."""
    found = [attrs.pop(k) for k in keys if k in attrs]
    return maybe_json(found[0]) if found else None


def _take_messages(attrs: dict, which: str) -> list | None:
    """OpenLLMetry's gen_ai.prompt.0.role / gen_ai.prompt.0.content / ... as a list of dicts."""
    messages: dict[int, dict] = {}
    for key in [k for k in attrs if k.startswith(f"gen_ai.{which}.")]:
        match = _INDEXED.match(key)
        if match:
            messages.setdefault(int(match[2]), {})[match[3]] = maybe_json(attrs.pop(key))
    return [messages[i] for i in sorted(messages)] or None


def _take_usage(attrs: dict) -> dict | None:
    return normalize_usage({key: next((attrs.pop(n) for n in names if n in attrs), None)
                            for key, names in _USAGE_KEYS.items()})


def _error(span: Any) -> str | None:
    if span.status.status_code is not StatusCode.ERROR:
        return None
    for event in span.events or ():
        if event.name == "exception":
            attributes = event.attributes or {}
            kind, message = attributes.get("exception.type"), attributes.get("exception.message")
            if kind:
                return f"{kind}: {message}" if message else str(kind)
    return span.status.description or "error"


def _details(span: Any) -> dict:
    """Kind, input, output and usage read from the span's attributes; the rest go in attrs."""
    attrs = dict(span.attributes or {})
    kind = _kind(attrs)
    input = _take(attrs, _INPUT_KEYS) or _take_messages(attrs, "prompt")
    output = _take(attrs, _OUTPUT_KEYS) or _take_messages(attrs, "completion")
    attrs["otel.trace_id"] = f"{span.context.trace_id:032x}"
    attrs["otel.span_id"] = f"{span.context.span_id:016x}"
    if span.kind is not SpanKind.INTERNAL:
        attrs["otel.kind"] = span.kind.name.lower()
    return {"kind": kind, "input": input, "output": output, "usage": _take_usage(attrs), "attrs": attrs}


class DeepEyeSpanProcessor(SpanProcessor):
    """An OpenTelemetry span processor that records spans as deep-eye spans.
    Its methods never raise."""

    @tracer.never_raises
    def on_start(self, span: Any, parent_context: Any = None) -> None:
        details = _details(span)
        parent = _open.get(_key(span.parent)) if span.parent is not None else None
        _open[_key(span.context)] = tracer.start_span(
            span.name, details["kind"], details["input"],
            # without a known parent: nest under the enclosing @trace block, or start a run
            **({"parent": parent} if parent is not None else {}),
            attrs=details["attrs"],
            start_time=span.start_time / 1e9 if span.start_time else None,
        )

    @tracer.never_raises
    def on_end(self, span: Any) -> None:
        handle = _open.pop(_key(span.context), None)
        if handle is None:
            return
        details = _details(span)  # attributes set after the span started are only known now
        known = handle.attrs or {}
        new_attrs = {k: v for k, v in details["attrs"].items() if k not in known or known[k] != v}
        tracer.end_span(
            handle,
            details["output"],
            error=_error(span),
            usage=details["usage"],
            **({"input": details["input"]} if handle.input is None and details["input"] is not None else {}),
            name=span.name if span.name != handle.name else None,
            kind=details["kind"] if details["kind"] != handle.display_kind else None,
            attrs=new_attrs or None,
            end_time=span.end_time / 1e9 if span.end_time else None,
        )

    def shutdown(self) -> None:
        tracer.flush()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return tracer.flush(timeout_millis / 1000)


def install(tracer_provider: Any = None) -> DeepEyeSpanProcessor:
    """Record the spans of `tracer_provider` (default: the global one) with deep-eye.

    If no OpenTelemetry SDK provider is set up yet, one is created and made global.
    Calling it again for the same provider does nothing.
    """
    provider = tracer_provider or otel_trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):  # the API's default, which records nothing
        provider = TracerProvider()
        otel_trace.set_tracer_provider(provider)
        provider = otel_trace.get_tracer_provider()  # another one may have been set first
    if not hasattr(provider, "add_span_processor"):
        log.warning("deep-eye: the global tracer provider (%s) can't take span processors; "
                    "pass an OpenTelemetry SDK TracerProvider to install()", type(provider).__name__)
        return DeepEyeSpanProcessor()
    if provider not in _installed:
        _installed[provider] = DeepEyeSpanProcessor()
        provider.add_span_processor(_installed[provider])
    return _installed[provider]
