"""The tracing API: @trace, span(), and the lower-level begin/finish helpers.

The current span lives in a ContextVar, so nesting works across plain calls
and asyncio tasks. Threads don't inherit it on their own: wrap the function you
hand to a thread or executor with `propagate()`.
"""

import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
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
        arguments = dict(bound.arguments)
    except TypeError:
        return {"args": args, "kwargs": kwargs}
    first = next(iter(sig.parameters), None)
    if first in ("self", "cls"):  # the object itself is noise, and its repr can be huge
        arguments.pop(first, None)
    return arguments


def propagate(fn: Callable) -> Callable:
    """Make `fn` nest under the current span when it runs in another thread.

        with ThreadPoolExecutor() as ex:
            ex.map(propagate(work), items)
        threading.Thread(target=propagate(work)).start()
    """
    context = copy_context()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # A Context can only be entered by one thread at a time, so each call gets its own copy.
        return context.copy().run(fn, *args, **kwargs)

    return wrapper


def _run_in_span(s: Span, step: Callable, arg: Any) -> Any:
    """Advance a generator with `s` as the current span, without holding it across yields."""
    token = _current_span.set(s)
    try:
        return step(arg)
    finally:
        _current_span.reset(token)


async def _arun_in_span(s: Span, step: Callable, arg: Any) -> Any:
    token = _current_span.set(s)
    try:
        return await step(arg)
    finally:
        _current_span.reset(token)


def trace(fn: Callable | None = None, *, name: str | None = None, kind: str = "function"):
    """Decorator that records a function's inputs, output, errors and timing.

    Usable bare (`@trace`) or configured (`@trace(name="x", kind="tool")`).
    Works on sync and async functions and generators; a generator's span lasts
    until it is exhausted (or closed) and its output is the list of yielded items.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                s = begin_span(span_name, kind, _capture_args(sig, args, kwargs))
                items: list = []
                agen = func(*args, **kwargs)
                step, arg = agen.asend, None
                try:
                    while True:
                        try:
                            item = await _arun_in_span(s, step, arg)
                        except StopAsyncIteration:
                            break
                        items.append(item)
                        try:
                            step, arg = agen.asend, (yield item)
                        except GeneratorExit:  # the caller stopped early: not an error
                            await agen.aclose()
                            raise
                        except BaseException as exc:  # thrown in by the caller: pass it on
                            step, arg = agen.athrow, exc
                except GeneratorExit:
                    finish_span(s, items)
                    raise
                except BaseException as exc:
                    finish_span(s, items, error=exc)
                    raise
                finish_span(s, items)
            return async_gen_wrapper

        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def gen_wrapper(*args, **kwargs):
                s = begin_span(span_name, kind, _capture_args(sig, args, kwargs))
                items: list = []
                gen = func(*args, **kwargs)
                step, arg = gen.send, None
                try:
                    while True:
                        try:
                            item = _run_in_span(s, step, arg)
                        except StopIteration as stop:
                            result = stop.value
                            break
                        items.append(item)
                        try:
                            step, arg = gen.send, (yield item)
                        except GeneratorExit:  # the caller stopped early: not an error
                            gen.close()
                            raise
                        except BaseException as exc:  # thrown in by the caller: pass it on
                            step, arg = gen.throw, exc
                except GeneratorExit:
                    finish_span(s, items)
                    raise
                except BaseException as exc:
                    finish_span(s, items, error=exc)
                    raise
                finish_span(s, items)
                return result
            return gen_wrapper

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
