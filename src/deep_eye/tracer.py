"""The tracing API: @trace, span(), and the lower-level begin/finish helpers.

The current span lives in a ContextVar, so nesting works across plain calls,
threads started with copied contexts, and asyncio tasks.
"""

import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from . import storage
from .models import Span

log = logging.getLogger("deep_eye")

_current_span: ContextVar[Span | None] = ContextVar("deep_eye_current_span", default=None)
_parent_resolvers: list[Callable[[], Span | None]] = []
_UNSET: Any = object()


def register_parent_resolver(resolver: Callable[[], Span | None]) -> None:
    """Let an integration tell us the parent span when the ContextVar is empty."""
    _parent_resolvers.append(resolver)


def current_span() -> Span | None:
    span = _current_span.get()
    if span is not None:
        return span
    for resolver in _parent_resolvers:
        span = resolver()
        if span is not None:
            return span
    return None


def _new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:6]}"


def begin_span(name: str, kind: str = "function", input: Any = None, parent: Any = _UNSET) -> Span:
    """Open a span. Pass `parent` explicitly (or None for a new run), else the current span is used."""
    if parent is _UNSET:
        parent = current_span()
    span = Span(
        span_id=uuid.uuid4().hex[:12],
        run_id=parent.run_id if parent else _new_run_id(),
        parent_id=parent.span_id if parent else None,
        name=name,
        kind=kind,
        input=input,
    )
    _safe(storage.write_start, span)
    return span


def finish_span(
    span: Span,
    output: Any = None,
    error: BaseException | str | None = None,
    usage: dict | None = None,
) -> None:
    span.end = time.time()
    span.output = output
    if usage is not None:
        span.usage = usage
    if error is not None:
        span.error = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    _safe(storage.write_end, span)


def _safe(write: Callable[[Span], None], span: Span) -> None:
    """Tracing must never break the traced program."""
    try:
        write(span)
    except Exception:
        log.warning("deep-eye failed to write a trace event", exc_info=True)


@contextmanager
def span(name: str, kind: str = "function", input: Any = None):
    """Trace a block of code. Inside it, set `s.output = ...` (and `s.usage = {...}` for LLM calls)."""
    s = begin_span(name, kind, input)
    token = _current_span.set(s)
    try:
        yield s
    except BaseException as exc:
        finish_span(s, s.output, error=exc)
        raise
    else:
        finish_span(s, s.output)
    finally:
        _current_span.reset(token)


def _capture_args(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        return {"args": args, "kwargs": kwargs}


def trace(fn: Callable | None = None, *, name: str | None = None, kind: str = "function"):
    """Decorator that records a function's inputs, output, errors and timing.

    Usable bare (`@trace`) or configured (`@trace(name="x", kind="tool")`).
    Works on sync and async functions.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with span(span_name, kind, _capture_args(sig, args, kwargs)) as s:
                    s.output = await func(*args, **kwargs)
                    return s.output
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with span(span_name, kind, _capture_args(sig, args, kwargs)) as s:
                s.output = func(*args, **kwargs)
                return s.output
        return wrapper

    return decorator(fn) if fn is not None else decorator
