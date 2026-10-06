"""Where traces are stored, how much of each value is kept, and what gets redacted."""

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

ENV_DIR = "DEEP_EYE_DIR"
ENV_ENABLED = "DEEP_EYE_ENABLED"  # "0" / "false" / "off" turns tracing off
ENV_SAMPLE_RATE = "DEEP_EYE_SAMPLE_RATE"  # e.g. "0.1" keeps one run in ten
ENV_SYNC_WRITES = "DEEP_EYE_SYNC_WRITES"  # "1" writes each event before the traced call goes on
DEFAULT_DIR = ".deep-eye"
_FALSE = ("0", "false", "no", "off")

# Dict keys / argument names whose values are never written to disk (case-insensitive substrings).
DEFAULT_REDACT_KEYS = (
    "api_key", "apikey", "api-key", "secret", "password", "passwd", "authorization",
    "access_token", "refresh_token", "auth_token", "private_key", "credential", "cookie",
)

_trace_dir: Path | None = None
_max_chars = 20_000
_max_items = 200
_redact = True
_extra_redact_keys: tuple[str, ...] = ()
_redact_fn: Callable[[str], str] | None = None
_max_age_days: float | None = None
_enabled: bool | None = None  # None: use the environment variable
_sample_rate: float | None = None
_sync_writes: bool | None = None

_SETTINGS = ("_trace_dir", "_max_chars", "_max_items", "_redact", "_extra_redact_keys",
             "_redact_fn", "_max_age_days", "_enabled", "_sample_rate", "_sync_writes")
_UNSET: Any = object()


class configure:
    """Change where traces go and what is kept. Arguments left out keep their current value.

    trace_dir     folder for trace files (default: $DEEP_EYE_DIR or ./.deep-eye)
    max_chars     longest string kept before truncating (default 20,000)
    max_items     most list items / dict entries kept per value (default 200)
    redact        hide secrets: keys like `api_key`/`password` and strings like `sk-...` (default True).
                  False turns off all redaction, including `redact_fn`.
    redact_keys   extra key names to hide, on top of the built-in ones
    redact_fn     your own function, called with every string that is stored (after the built-in
                  redaction, before truncation) and returning the string to store instead.
                  If it raises, the string is stored as "[redacted]".
    max_age_days  delete traces older than this when a new run starts (default: keep everything)
    enabled       False turns tracing off: traced code runs untouched (default: $DEEP_EYE_ENABLED or True)
    sample_rate   share of runs to record, 0.0 to 1.0, decided per run (default: $DEEP_EYE_SAMPLE_RATE or 1.0)
    sync_writes   write each event before the traced code continues, instead of in a background
                  thread. Slower, but nothing is lost if the process is killed or segfaults.
                  (default: $DEEP_EYE_SYNC_WRITES or False)

    Called on its own, the change lasts for the rest of the process. Used as a context
    manager, the previous settings come back when the block ends:

        with deep_eye.configure(trace_dir="traces/eval-1"):
            run_agent()

    Runs started inside the block keep writing to their folder even if they outlive it.
    """

    def __init__(
        self,
        trace_dir: str | Path | None = None,
        max_chars: int | None = None,
        max_items: int | None = None,
        redact: bool | None = None,
        redact_keys: Iterable[str] | None = None,
        redact_fn: Callable[[str], str] | None = _UNSET,
        max_age_days: float | None = _UNSET,
        enabled: bool | None = None,
        sample_rate: float | None = None,
        sync_writes: bool | None = None,
    ) -> None:
        global _trace_dir, _max_chars, _max_items, _redact, _extra_redact_keys, _redact_fn, _max_age_days
        global _enabled, _sample_rate, _sync_writes
        self._previous = {name: globals()[name] for name in _SETTINGS}
        if trace_dir is not None:
            _trace_dir = Path(trace_dir)
        if max_chars is not None:
            _max_chars = max_chars
        if max_items is not None:
            _max_items = max_items
        if redact is not None:
            _redact = redact
        if redact_keys is not None:
            _extra_redact_keys = tuple(k.lower() for k in redact_keys)
        if redact_fn is not _UNSET:
            _redact_fn = redact_fn
        if max_age_days is not _UNSET:
            _max_age_days = max_age_days
        if enabled is not None:
            _enabled = enabled
        if sample_rate is not None:
            if not 0 <= sample_rate <= 1:
                raise ValueError("sample_rate must be between 0 and 1")
            _sample_rate = sample_rate
        if sync_writes is not None:
            _sync_writes = sync_writes

    def __enter__(self) -> "configure":
        return self

    def __exit__(self, *exc) -> None:
        globals().update(self._previous)


def get_trace_dir() -> Path:
    """Resolve the trace directory: configure() > $DEEP_EYE_DIR > ./.deep-eye

    Called whenever a run starts (never cached), so changes apply to the next run.
    """
    if _trace_dir is not None:
        return _trace_dir
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)


def get_max_chars() -> int:
    return _max_chars


def get_max_items() -> int:
    return _max_items


def redaction_enabled() -> bool:
    return _redact


def get_redact_keys() -> tuple[str, ...]:
    return DEFAULT_REDACT_KEYS + _extra_redact_keys


def get_redact_fn() -> Callable[[str], str] | None:
    return _redact_fn


def get_max_age_days() -> float | None:
    return _max_age_days


def tracing_enabled() -> bool:
    if _enabled is not None:
        return _enabled
    return os.environ.get(ENV_ENABLED, "").strip().lower() not in _FALSE


def get_sample_rate() -> float:
    if _sample_rate is not None:
        return _sample_rate
    try:
        return min(max(float(os.environ.get(ENV_SAMPLE_RATE, "1")), 0.0), 1.0)
    except ValueError:  # a typo in the variable must not break the traced program
        return 1.0


def sync_writes() -> bool:
    if _sync_writes is not None:
        return _sync_writes
    return os.environ.get(ENV_SYNC_WRITES, "").strip().lower() not in ("", *_FALSE)
