"""deep-eye: local, file-based tracing for LLM agents."""

from .config import configure
from .tracer import current_span, end_span, flush, propagate, span, start_span, trace

__all__ = ["trace", "span", "start_span", "end_span", "current_span", "propagate", "configure", "flush"]
__version__ = "0.1.0"
