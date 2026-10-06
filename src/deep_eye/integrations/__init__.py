"""Framework adapters, one module each: langchain, openai_agents, otel.

Conventions every adapter follows (see "Writing an adapter" in the README):

- It lives in deep_eye/integrations/<name>.py and imports its framework only there,
  so `import deep_eye` never imports a framework.
- A missing framework raises `missing_framework(...)`, which names the extra to install.
- It records spans with tracer.start_span() / end_span(), using the kinds in schema.KINDS.
- It never raises into the user's program: callbacks are wrapped in tracer.never_raises.
"""


import json
from typing import Any


def missing_framework(adapter: str, package: str, extra: str) -> ImportError:
    return ImportError(
        f"deep_eye.integrations.{adapter} needs the '{package}' package, which is not installed. "
        f'Install it with: pip install "deep-eye[{extra}]"'
    )


_USAGE_NAMES = {  # deep-eye key: the names frameworks use for it
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "total_tokens": ("total_tokens",),
}


def normalize_usage(usage: Any) -> dict | None:
    """Token counts from a dict or an object (input/prompt, output/completion, total) as
    deep-eye's usage dict, or None when there are none."""
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda name: getattr(usage, name, None)
    out = {}
    for key, names in _USAGE_NAMES.items():
        for name in names:
            value = get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = value
                break
    return out or None


def maybe_json(value: Any) -> Any:
    """Frameworks often pass JSON as a string; decode it so the viewer can expand it."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value
