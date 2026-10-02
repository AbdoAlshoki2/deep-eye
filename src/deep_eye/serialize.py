"""Turn arbitrary Python objects into JSON-safe, size-limited values."""

import dataclasses
import enum
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import get_max_chars

_MAX_DEPTH = 8


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion; never raises, falls back to repr()."""
    limit = get_max_chars()
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _truncate(obj, limit)
    if _depth >= _MAX_DEPTH:
        return _truncate(repr(obj), limit)
    try:
        if isinstance(obj, bytes):
            return f"<{len(obj)} bytes>"
        if isinstance(obj, enum.Enum):
            return to_jsonable(obj.value, _depth + 1)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Mapping):
            return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [to_jsonable(v, _depth + 1) for v in obj]
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return to_jsonable(dataclasses.asdict(obj), _depth + 1)
        if hasattr(obj, "model_dump"):  # pydantic v2 (incl. LangChain messages)
            return to_jsonable(obj.model_dump(), _depth + 1)
    except Exception:
        pass
    return _truncate(repr(obj), limit)
