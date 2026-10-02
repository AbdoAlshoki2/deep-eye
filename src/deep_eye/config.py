"""Where traces are stored and how much of each value is kept."""

import os
from pathlib import Path

ENV_DIR = "DEEP_EYE_DIR"
DEFAULT_DIR = ".deep-eye"

_trace_dir: Path | None = None
_max_chars = 20_000


def configure(trace_dir: str | Path | None = None, max_chars: int | None = None) -> None:
    """Override the trace directory and/or the per-string size limit."""
    global _trace_dir, _max_chars
    if trace_dir is not None:
        _trace_dir = Path(trace_dir)
    if max_chars is not None:
        _max_chars = max_chars


def get_trace_dir() -> Path:
    """Resolve the trace directory: configure() > $DEEP_EYE_DIR > ./.deep-eye"""
    if _trace_dir is not None:
        return _trace_dir
    return Path(os.environ.get(ENV_DIR, DEFAULT_DIR))


def get_max_chars() -> int:
    return _max_chars
