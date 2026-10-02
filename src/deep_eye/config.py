"""Where traces are stored, how much of each value is kept, and what gets redacted."""

import os
from collections.abc import Iterable
from pathlib import Path

ENV_DIR = "DEEP_EYE_DIR"
DEFAULT_DIR = ".deep-eye"

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


def configure(
    trace_dir: str | Path | None = None,
    max_chars: int | None = None,
    max_items: int | None = None,
    redact: bool | None = None,
    redact_keys: Iterable[str] | None = None,
) -> None:
    """Change where traces go and what is kept. Arguments left as None keep their current value.

    trace_dir    folder for trace files (default: $DEEP_EYE_DIR or ./.deep-eye)
    max_chars    longest string kept before truncating (default 20,000)
    max_items    most list items / dict entries kept per value (default 200)
    redact       hide secrets: keys like `api_key`/`password` and strings like `sk-...` (default True)
    redact_keys  extra key names to hide, on top of the built-in ones
    """
    global _trace_dir, _max_chars, _max_items, _redact, _extra_redact_keys
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


def get_trace_dir() -> Path:
    """Resolve the trace directory: configure() > $DEEP_EYE_DIR > ./.deep-eye"""
    if _trace_dir is not None:
        return _trace_dir
    return Path(os.environ.get(ENV_DIR, DEFAULT_DIR))


def get_max_chars() -> int:
    return _max_chars


def get_max_items() -> int:
    return _max_items


def redaction_enabled() -> bool:
    return _redact


def get_redact_keys() -> tuple[str, ...]:
    return DEFAULT_REDACT_KEYS + _extra_redact_keys
