"""Rich renderables shared by the TUI and the plain `show` command.

Every function that draws color takes a `Palette`, so the same code serves all themes.

Traced values are untrusted text: they are always wrapped in `Text` (never parsed as
Rich markup) and passed through `clean()`, so they can't style, click or move anything.
"""

import json
import re
from datetime import datetime

from rich.console import Group, RenderableType
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from .models import Run, Span
from .themes import Palette

# Control characters other than newline and tab: ESC (ANSI sequences), BEL, backspace, CR, C1 codes, ...
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean(text: str) -> str:
    """Show control characters as visible escapes (`\\x1b`) instead of sending them to the terminal."""
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


def plain(text: str, style: str = "") -> Text:
    """Untrusted text (run names, ids, ...) as a renderable that is shown literally."""
    return Text(clean(text), style=style)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "…"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def fmt_tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def fmt_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def status_text(status: str, palette: Palette) -> Text:
    """A colored status, for table cells."""
    return Text(status, style=palette.status_style(status))


def _span_duration(span: Span, run: Run | None) -> str:
    """The duration; for an open span of a run that stopped, the time until its last event."""
    if span.end is None and run is not None and run.span_status(span) == "unfinished":
        return f"≥{fmt_duration(max(run.last_time - span.start, 0))}"
    return fmt_duration(span.duration)


def span_label(span: Span, palette: Palette, run: Run | None = None) -> Text:
    """One tree line: `kind  name  duration  tokens`, highlighted on error."""
    label = Text()
    label.append(f"{clean(span.kind):<9}", style=palette.kind_style(span.kind))
    label.append(clean(span.name), style=palette.error if span.status == "error" else "bold")
    label.append(f"  {_span_duration(span, run)}", style="dim")
    if span.tokens:
        label.append(f"  {fmt_tokens(span.tokens)} tok", style=palette.accent)
    if span.status == "error":
        label.append("  ✗", style=palette.accent)
    elif span.interrupted:
        label.append("  ⏹", style=palette.accent)
    return label


def _value(value, palette: Palette) -> RenderableType:
    if value is None:
        return Text("—", style="dim")
    if isinstance(value, str):
        return Text(clean(value))
    return Syntax(clean(json.dumps(value, indent=2, ensure_ascii=False)), "json",
                  theme=palette.syntax, word_wrap=True, background_color="default")


def span_detail(span: Span, palette: Palette, run: Run | None = None) -> RenderableType:
    """Full detail pane for one span: header, input, output, error."""
    status = run.span_status(span) if run else span.status
    header = Text()
    header.append(f"{clean(span.name)}\n", style="bold")
    header.append(clean(span.kind), style=palette.kind_style(span.kind))
    header.append(f"  ·  {status}", style=palette.status_style(status))
    header.append(f"  ·  {_span_duration(span, run)}  ·  {fmt_time(span.start)}", style="dim")
    if span.tokens:
        u = span.usage or {}
        header.append(
            clean(f"\n{u.get('input_tokens', '?')} in  ·  {u.get('output_tokens', '?')} out"
                  f"  ·  {span.tokens} total tokens"),
            style=palette.accent,
        )

    parts: list[RenderableType] = [
        header,
        Rule("Input", style="dim"), _value(span.input, palette),
        Rule("Output", style="dim"), _value(span.output, palette),
    ]
    if span.error:
        parts += [Rule("Error", style=palette.accent), Text(clean(str(span.error)), style=palette.accent)]
    return Group(*parts)
