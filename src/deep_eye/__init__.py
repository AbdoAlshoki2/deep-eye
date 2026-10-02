"""deep-eye: local, file-based tracing for LLM agents."""

from .config import configure
from .tracer import current_span, span, trace

__all__ = ["trace", "span", "current_span", "configure"]
__version__ = "0.1.0"
