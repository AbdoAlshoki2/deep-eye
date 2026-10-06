"""The tracing API: @trace, span(), and the lower-level begin/finish helpers.

The current span lives in a ContextVar, so nesting works across plain calls
and asyncio tasks. Threads don't inherit it on their own: wrap the function you
hand to a thread or executor with `propagate()`.

Tracing never changes what the traced program does: exceptions pass through
unchanged, and a failure inside deep-eye itself is logged once and swallowed.
"""

import atexit
import functools
import inspect
import logging
import os
import random
import signal
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime
from typing import Any

from . import config, storage
from .models import Span

log = logging.getLogger("deep_eye")

NOT_CAPTURED = "[not captured]"

_current_span: ContextVar[Span | None] = ContextVar("deep_eye_current_span", default=None)
_parent_resolvers: list[Callable[[], Span | None]] = []
_UNSET: Any = object()

_open_spans: dict[str, Span] = {}  # span_id -> span, closed as "interrupted" if the process exits
_open_lock = threading.Lock()
_warned = False
_sigterm_installed = False


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


def _safe(action: Callable, *args) -> None:
    """Tracing must never break the traced program: log the first failure, ignore the rest."""
    global _warned
    try:
        action(*args)
    except Exception:
        if not _warned:
            _warned = True
            log.warning("deep-eye failed to write a trace event; later failures are not reported",
                        exc_info=True)


def begin_span(
    name: str,
    kind: str = "function",
    input: Any = None,
    parent: Any = _UNSET,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Span:
    """Open a span. Pass `parent` explicitly (or None for a new run), else the current span is used.

    When tracing is off, or the run wasn't sampled, the span is a stand-in that is never written
    (its children aren't either), so nesting and `s.output = ...` keep working at almost no cost.
    """
    if parent is _UNSET:
        parent = current_span()
    recording = parent.recording if parent else config.tracing_enabled() and _sampled()
    if not recording:
        return Span(span_id=uuid.uuid4().hex[:12], run_id=parent.run_id if parent else "",
                    parent_id=parent.span_id if parent else None, name=name, kind=kind, recording=False)
    span = Span(
        span_id=uuid.uuid4().hex[:12],
        run_id=parent.run_id if parent else _new_run_id(),
        parent_id=parent.span_id if parent else None,
        name=name,
        kind=kind,
        input=input if capture_input else NOT_CAPTURED,
        capture_output=capture_output,
    )
    if parent:
        span.file = parent.file  # a run stays in one file, even if the trace dir changes meanwhile
    else:
        _safe(_start_run, span)
    with _open_lock:
        _open_spans[span.span_id] = span
    _safe(storage.write_start, span)
    return span


def _sampled() -> bool:
    rate = config.get_sample_rate()
    return rate >= 1 or random.random() < rate


def _start_run(span: Span) -> None:
    trace_dir = config.get_trace_dir()
    span.file = storage.run_file(span.run_id, trace_dir)
    _install_sigterm_handler()
    storage.cleanup_if_due(trace_dir, config.get_max_age_days())


def _describe(error: BaseException | str, with_message: bool) -> str:
    if isinstance(error, str):
        return error
    if not with_message:
        return type(error).__name__
    try:
        message = str(error)
    except Exception:  # a broken __str__
        message = ""
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def finish_span(
    span: Span,
    output: Any = None,
    error: BaseException | str | None = None,
    usage: dict | None = None,
) -> None:
    """Close a span. Exceptions that aren't `Exception`s (Ctrl+C, cancellation, exit)
    mark it "interrupted". Closing a span twice is a no-op."""
    with _open_lock:
        if span.end is not None:
            return
        span.end = time.time()
        _open_spans.pop(span.span_id, None)
    if not span.recording:
        return
    span.output = output if span.capture_output else NOT_CAPTURED
    if usage is not None:
        span.usage = usage
    if error is not None:
        # without capture_output the message (often built from the data) is left out too
        span.error = _describe(error, with_message=span.capture_output)
        span.interrupted |= isinstance(error, BaseException) and not isinstance(error, Exception)
    _safe(storage.write_end, span)


def _close_open_spans(reason: str) -> None:
    """Give every span that is still open an end event, so the file shows where it stopped."""
    with _open_lock:
        spans = list(_open_spans.values())
    for s in sorted(spans, key=lambda s: s.start, reverse=True):  # innermost first
        s.interrupted = True
        finish_span(s, s.output, error=reason)


def _at_exit() -> None:
    _close_open_spans("process exited before this span finished")
    _safe(storage.flush)


atexit.register(_at_exit)


def flush(timeout: float | None = 5.0) -> bool:
    """Wait until every trace event so far is written to disk (events are written in the
    background). Happens by itself at exit; call it before reading trace files yourself."""
    return storage.flush(timeout)


def _on_sigterm(signum, frame) -> None:
    _close_open_spans("SIGTERM")
    _safe(storage.flush, 2.0)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)  # carry on with the normal termination


def _install_sigterm_handler() -> None:
    """Close open spans on SIGTERM, but only if nobody else handles it: never replace a handler."""
    global _sigterm_installed
    if _sigterm_installed or threading.current_thread() is not threading.main_thread():
        return
    _sigterm_installed = True
    if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _on_sigterm)


@contextmanager
def span(
    name: str,
    kind: str = "function",
    input: Any = None,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """Trace a block of code. Inside it, set `s.output = ...` (and `s.usage = {...}` for LLM calls).

    With capture_input/capture_output=False the input/output (and the error message) are stored
    as "[not captured]"; name, timing, nesting, status and the error type are still recorded.
    """
    s = begin_span(name, kind, input, capture_input=capture_input, capture_output=capture_output)
    token = _current_span.set(s)
    error: BaseException | None = None
    try:
        yield s
    except BaseException as exc:
        error = exc
        raise
    finally:
        finish_span(s, s.output, error=error)
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


def trace(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    kind: str = "function",
    capture_input: bool = True,
    capture_output: bool = True,
):
    """Decorator that records a function's inputs, output, errors and timing.

    Usable bare (`@trace`) or configured (`@trace(name="x", kind="tool")`).
    Works on sync and async functions and generators; a generator's span lasts
    until it is exhausted (or closed) and its output is the list of yielded items.

    capture_input=False / capture_output=False keep the arguments / return value (and the
    error message) out of the trace file; see `span()`.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        def begin(args: tuple, kwargs: dict) -> Span:
            inputs = _capture_args(sig, args, kwargs) if capture_input else None
            return begin_span(span_name, kind, inputs,
                              capture_input=capture_input, capture_output=capture_output)

        def traced_span(args: tuple, kwargs: dict):
            return span(span_name, kind, _capture_args(sig, args, kwargs) if capture_input else None,
                        capture_input=capture_input, capture_output=capture_output)

        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                s = begin(args, kwargs)
                items: list | None = [] if capture_output and s.recording else None  # only if stored
                agen = func(*args, **kwargs)
                step, arg = agen.asend, None
                try:
                    while True:
                        try:
                            item = await _arun_in_span(s, step, arg)
                        except StopAsyncIteration:
                            break
                        if items is not None:
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
                if not config.tracing_enabled():
                    return (yield from func(*args, **kwargs))
                s = begin(args, kwargs)
                items: list | None = [] if capture_output and s.recording else None
                gen = func(*args, **kwargs)
                step, arg = gen.send, None
                try:
                    while True:
                        try:
                            item = _run_in_span(s, step, arg)
                        except StopIteration as stop:
                            result = stop.value
                            break
                        if items is not None:
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
                if not config.tracing_enabled():
                    return await func(*args, **kwargs)
                with traced_span(args, kwargs) as s:
                    s.output = await func(*args, **kwargs)
                    return s.output
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not config.tracing_enabled():
                return func(*args, **kwargs)
            with traced_span(args, kwargs) as s:
                s.output = func(*args, **kwargs)
                return s.output
        return wrapper

    return decorator(fn) if fn is not None else decorator
