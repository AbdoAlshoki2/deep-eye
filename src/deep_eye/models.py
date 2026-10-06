"""Plain data objects shared by the tracer, the storage layer and the viewer."""

import os
import socket
import sys
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
    interrupted: bool = False  # ended by KeyboardInterrupt, cancellation, exit, ...

    # Only used while tracing, never stored:
    file: Path | None = field(default=None, repr=False, compare=False)  # the run's trace file
    capture_output: bool = field(default=True, repr=False, compare=False)
    recording: bool = field(default=True, repr=False, compare=False)  # False: tracing off / not sampled

    @property
    def tokens(self) -> int:
        """Total tokens used by this span itself (0 when none were reported)."""
        u = self.usage or {}
        try:
            return int(u.get("total_tokens") or u.get("input_tokens", 0) + u.get("output_tokens", 0))
        except (TypeError, ValueError):  # a hand-edited or foreign file
            return 0

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start

    @property
    def status(self) -> str:
        if self.interrupted:
            return "interrupted"
        if self.error:
            return "error"
        return "running" if self.end is None else "ok"


STALE_AFTER = 60  # seconds without writes before an unfinished run from another machine counts as crashed
HOST = socket.gethostname()


def pid_alive(pid: int) -> bool:
    """Is a process with this id running on this machine? Never sends a signal."""
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists, but isn't ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class Run:
    """All spans that share a root span, loaded from one trace file."""

    run_id: str
    path: Path
    spans: list[Span]
    mtime: float = 0.0  # when the file was last written
    incomplete: bool = False  # the last line was cut off mid-write
    pid: int | None = None  # process that wrote the run, and its machine
    host: str | None = None

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
    def last_time(self) -> float:
        """Time of the last recorded event."""
        return max((t for s in self.spans for t in (s.start, s.end) if t is not None), default=0.0)

    @property
    def alive(self) -> bool:
        """Can the process that writes this run still add to it?"""
        if self.pid is not None and self.host == HOST:
            return pid_alive(self.pid)
        return time.time() - self.mtime < STALE_AFTER

    @property
    def status(self) -> str:
        """ok / error / interrupted once the root span ended; otherwise running,
        or crashed / incomplete when the writing process is gone."""
        root = self.root
        if root is None or root.end is None:
            if self.alive:
                return "running"
            return "incomplete" if self.incomplete else "crashed"
        statuses = {s.status for s in self.spans}
        for status in ("error", "interrupted"):
            if status in statuses:
                return status
        return "ok"

    def span_status(self, span: Span) -> str:
        """Like span.status, but an open span of a dead run is 'unfinished', not 'running'."""
        if span.end is None and self.status != "running":
            return "unfinished"
        return span.status

    def children_of(self, span_id: str | None) -> list[Span]:
        return sorted(
            (s for s in self.spans if s.parent_id == span_id), key=lambda s: s.start
        )
