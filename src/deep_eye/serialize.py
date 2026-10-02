"""Turn arbitrary Python objects into JSON-safe, size-limited, redacted values."""

import dataclasses
import enum
import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import get_max_chars, get_max_items, get_redact_keys, redaction_enabled

_MAX_DEPTH = 8
REDACTED = "[redacted]"

# Strings that look like credentials, wherever they appear (even under an innocent key).
_SECRET_PATTERNS = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}"            # OpenAI / Anthropic style keys
    r"|\bAIza[0-9A-Za-z_-]{30,}"          # Google API keys
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}"      # GitHub tokens
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"    # Slack tokens
    r"|\bAKIA[0-9A-Z]{16}\b"              # AWS access key ids
    r"|(?i:\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}"
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"


def _clean_str(text: str) -> str:
    if redaction_enabled():
        text = _SECRET_PATTERNS.sub(REDACTED, text)
    return _truncate(text, get_max_chars())


def _is_secret_key(key: str) -> bool:
    key = key.lower()
    return redaction_enabled() and any(k in key for k in get_redact_keys())


def _more(count: int, kept: int, what: str) -> str:
    return f"... [+{count - kept} {what}]"


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion; never raises, falls back to repr()."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _clean_str(obj)
    if _depth >= _MAX_DEPTH:
        return _clean_str(repr(obj))
    limit = get_max_items()
    try:
        if isinstance(obj, bytes):
            return f"<{len(obj)} bytes>"
        if isinstance(obj, enum.Enum):
            return to_jsonable(obj.value, _depth + 1)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Mapping):
            out = {}
            for k, v in itertools.islice(obj.items(), limit):
                k = str(k)
                out[k] = REDACTED if _is_secret_key(k) else to_jsonable(v, _depth + 1)
            if len(obj) > limit:
                out["..."] = _more(len(obj), limit, "entries")
            return out
        if isinstance(obj, (list, tuple, set, frozenset)):
            out = [to_jsonable(v, _depth + 1) for v in itertools.islice(obj, limit)]
            if len(obj) > limit:
                out.append(_more(len(obj), limit, "items"))
            return out
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return to_jsonable(dataclasses.asdict(obj), _depth + 1)
        if hasattr(obj, "model_dump"):  # pydantic v2 (incl. LangChain messages)
            return to_jsonable(obj.model_dump(), _depth + 1)
        if hasattr(obj, "dict") and hasattr(obj, "__fields__"):  # pydantic v1
            return to_jsonable(obj.dict(), _depth + 1)
    except Exception:
        pass
    return _clean_str(repr(obj))
