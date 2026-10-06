"""The tracing API: start_span() / end_span(), and @trace and span() built on them.

start_span() returns a handle that end_span() closes, from any thread or task: that is
what framework adapters use, since callbacks rarely nest like Python blocks.

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
from .schema import ORIGINAL_KIND, normalize_kind

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
    """The innermost open span here: the enclosing @trace / span() block, or a framework span
    an adapter knows we are running inside, whichever started last."""
    span = _current_span.get()
    for resolver in _parent_resolvers:
        try:
            candidate = resolver()
        except Exception:
            _warn_once("a parent resolver failed")
            continue
        if candidate is not None and (span is None or candidate.start > span.start):
            span = candidate
    return span


def _new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:6]}"


def _warn_once(what: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning("deep-eye: %s; later failures are not reported", what, exc_info=True)


def _safe(action: Callable, *args) -> None:
    """Tracing must never break the traced program: log the first failure, ignore the rest."""
    try:
        action(*args)
    except Exception:
        _warn_once("failed to write a trace event")


def never_raises(fn: Callable) -> Callable:
    """For adapter callbacks: an exception inside `fn` is logged once and swallowed, so a
    tracing bug can never reach the traced program. The call then returns None."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            _warn_once(f"{fn.__qualname__} failed")
            return None

    return wrapper


def start_span(
    name: str,
    kind: str = "function",
    input: Any = None,
    *,
    parent: Span | None = _UNSET,
    attrs: dict | None = None,
    start_time: float | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Span:
    """Open a span and return its handle. Close it with `end_span(handle)`, from any thread or task.

    parent      the parent span's handle; None starts a new run; left out, the current span
                (the enclosing @trace / span() block) is used. The new span does not become
                the current one: pass it as `parent=` to nest more spans under it.
    kind        one of schema.KINDS; anything else is stored as "other", with the name you
                gave kept in attrs["original_kind"]
    attrs       extra JSON-able details to keep (model name, framework ids, ...)
    start_time  Unix time in seconds, when the work really started (default: now)

    When tracing is off, or the run wasn't sampled, the handle is a stand-in that is never written
    (its children aren't either), so nesting and `handle.output = ...` keep working at almost no
    cost. Never raises: if deep-eye itself fails, you get a stand-in handle.
    """
    try:
        return _start_span(name, kind, input, parent, attrs, start_time, capture_input, capture_output)
    except Exception:
        _warn_once("start_span() failed")
        return Span(span_id=uuid.uuid4().hex[:12], run_id="", name=str(name), recording=False, started=True)


def _start_span(name, kind, input, parent, attrs, start_time, capture_input, capture_output) -> Span:
    if parent is not _UNSET and parent is not None and not isinstance(parent, Span):
        log.debug("deep-eye: start_span() got a parent of type %s, not a span; using the current span",
                  type(parent).__name__)
        parent = _UNSET
    if parent is _UNSET:
        parent = current_span()
    kind, attrs = normalize_kind(kind, dict(attrs) if attrs else None)
    span = Span(
        span_id=uuid.uuid4().hex[:12],
        run_id=parent.run_id if parent else "",
        parent_id=parent.span_id if parent else None,
        name=str(name),
        kind=kind,
        started=True,
    )
    span.recording = parent.recording if parent else config.tracing_enabled() and _sampled()
    if not span.recording:
        return span
    if start_time is not None:
        # clocks of different precision (e.g. a framework's ISO timestamps) must not put a
        # child before its parent
        span.start = max(float(start_time), parent.start) if parent else float(start_time)
    span.input = input if capture_input else NOT_CAPTURED
    span.attrs = attrs
    span.capture_input, span.capture_output = capture_input, capture_output
    if parent:
        span.file = parent.file  # a run stays in one file, even if the trace dir changes meanwhile
    else:
        span.run_id = _new_run_id()
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
    storage.write_header(span)
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


def end_span(
    handle: Span,
    output: Any = _UNSET,
    error: BaseException | str | None = None,
    usage: dict | None = None,
    *,
    input: Any = _UNSET,
    name: str | None = None,
    kind: str | None = None,
    attrs: dict | None = None,
    end_time: float | None = None,
) -> None:
    """Close a span opened by `start_span()`, from any thread or task.

    output      the result (default: whatever was set as `handle.output`)
    error       an exception or a message; exceptions that aren't `Exception`s (Ctrl+C,
                cancellation, exit) mark the span "interrupted"
    usage       token counts, e.g. {"input_tokens": 10, "output_tokens": 5}
    input, name, kind
                for details only known once the work is done; they replace the ones given
                to start_span()
    attrs       merged into the span's attrs
    end_time    Unix time in seconds, when the work really ended (default: now)

    Ending a span twice, or something that isn't a handle from start_span(), is ignored
    (with a debug log). Never raises.
    """
    try:
        _end_span(handle, output, error, usage, input, name, kind, attrs, end_time)
    except Exception:
        _warn_once("end_span() failed")


def _end_span(span, output, error, usage, input, name, kind, attrs, end_time) -> None:
    if not isinstance(span, Span) or not span.started:
        log.debug("deep-eye: end_span() got %s, not a handle from start_span(); ignored",
                  type(span).__name__)
        return
    with _open_lock:
        if span.end is not None:
            log.debug("deep-eye: span %s (%s) was already ended; ignored", span.span_id, span.name)
            return
        span.end = time.time() if end_time is None else max(float(end_time), span.start)
        _open_spans.pop(span.span_id, None)
    if not span.recording:
        return
    if output is not _UNSET:
        span.output = output
    if not span.capture_output:
        span.output = NOT_CAPTURED
    if usage is not None:
        span.usage = usage
    if error is not None:
        # without capture_output the message (often built from the data) is left out too
        span.error = _describe(error, with_message=span.capture_output)
        span.interrupted |= isinstance(error, BaseException) and not isinstance(error, Exception)
    late: dict[str, Any] = {}
    if name is not None:
        span.name = late["name"] = str(name)
    if input is not _UNSET:
        span.input = late["input"] = input if span.capture_input else NOT_CAPTURED
    if attrs:
        late["attrs"] = dict(attrs)
        span.attrs = {**(span.attrs or {}), **attrs}
    if kind is not None:
        span.kind, span.attrs = normalize_kind(kind, span.attrs)
        late["kind"] = span.kind
        if span.kind == "other":
            late["attrs"] = {**late.get("attrs", {}), ORIGINAL_KIND: span.attrs[ORIGINAL_KIND]}
    _safe(storage.write_end, span, late)


def _close_open_spans(reason: str) -> None:
    """Give every span that is still open an end event, so the file shows where it stopped."""
    with _open_lock:
        spans = list(_open_spans.values())
    for s in sorted(spans, key=lambda s: s.start, reverse=True):  # innermost first
        s.interrupted = True
        end_span(s, error=reason)


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
    attrs: dict | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """Trace a block of code. Inside it, set `s.output = ...` (and `s.usage = {...}` for LLM calls).

    With capture_input/capture_output=False the input/output (and the error message) are stored
    as "[not captured]"; name, timing, nesting, status and the error type are still recorded.
    """
    s = start_span(name, kind, input, attrs=attrs, capture_input=capture_input,
                   capture_output=capture_output)
    token = _current_span.set(s)
    error: BaseException | None = None
    try:
        yield s
    except BaseException as exc:
        error = exc
        raise
    finally:
        end_span(s, error=error)
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
    attrs: dict | None = None,
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
            return start_span(span_name, kind, inputs, attrs=attrs,
                              capture_input=capture_input, capture_output=capture_output)

        def traced_span(args: tuple, kwargs: dict):
            return span(span_name, kind, _capture_args(sig, args, kwargs) if capture_input else None,
                        attrs=attrs, capture_input=capture_input, capture_output=capture_output)

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
                    end_span(s, items)
                    raise
                except BaseException as exc:
                    end_span(s, items, error=exc)
                    raise
                end_span(s, items)
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
                    end_span(s, items)
                    raise
                except BaseException as exc:
                    end_span(s, items, error=exc)
                    raise
                end_span(s, items)
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
