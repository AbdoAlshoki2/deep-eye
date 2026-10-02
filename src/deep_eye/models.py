"""Plain data objects shared by the tracer, the storage layer and the viewer."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    """One unit of work: a function, a tool call, an LLM call, ..."""

    span_id: str
    run_id: str
    name: str
    kind: str = "function"
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    input: Any = None
    output: Any = None
    error: str | None = None
    usage: dict | None = None  # token counts, e.g. {"input_tokens": 10, "output_tokens": 5}

    @property
    def tokens(self) -> int:
        """Total tokens used by this span itself (0 when none were reported)."""
        u = self.usage or {}
        return u.get("total_tokens") or u.get("input_tokens", 0) + u.get("output_tokens", 0)

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "running" if self.end is None else "ok"


@dataclass
class Run:
    """All spans that share a root span, loaded from one trace file."""

    run_id: str
    path: Path
    spans: list[Span]

    @property
    def root(self) -> Span | None:
        return next((s for s in self.spans if s.parent_id is None), None)

    @property
    def name(self) -> str:
        return self.root.name if self.root else self.run_id

    @property
    def start(self) -> float:
        return self.root.start if self.root else 0.0

    @property
    def duration(self) -> float | None:
        return self.root.duration if self.root else None

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.spans)

    @property
    def status(self) -> str:
        if any(s.error for s in self.spans):
            return "error"
        return "running" if any(s.end is None for s in self.spans) else "ok"

    def children_of(self, span_id: str | None) -> list[Span]:
        return sorted(
            (s for s in self.spans if s.parent_id == span_id), key=lambda s: s.start
        )
