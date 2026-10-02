"""Rich renderables shared by the TUI and the plain `show` command.

Every function that draws color takes a `Palette`, so the same code serves all themes.
"""

import json
from datetime import datetime

from rich.console import Group, RenderableType
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from .models import Span
from .themes import Palette


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


def status_markup(status: str, palette: Palette) -> str:
    """Rich console markup, for table cells."""
    return f"[{palette.status_style(status)}]{status}[/]"


def span_label(span: Span, palette: Palette) -> Text:
    """One tree line: `kind  name  duration  tokens`, highlighted on error."""
    label = Text()
    label.append(f"{span.kind:<9}", style=palette.kind_style(span.kind))
    label.append(span.name, style=palette.error if span.error else "bold")
    label.append(f"  {fmt_duration(span.duration)}", style="dim")
    if span.tokens:
        label.append(f"  {fmt_tokens(span.tokens)} tok", style=palette.accent)
    if span.error:
        label.append("  ✗", style=palette.accent)
    return label


def _value(value, palette: Palette) -> RenderableType:
    if value is None:
        return Text("—", style="dim")
    if isinstance(value, str):
        return Text(value)
    return Syntax(json.dumps(value, indent=2, ensure_ascii=False), "json",
                  theme=palette.syntax, word_wrap=True, background_color="default")


def span_detail(span: Span, palette: Palette) -> RenderableType:
    """Full detail pane for one span: header, input, output, error."""
    header = Text()
    header.append(f"{span.name}\n", style="bold")
    header.append(span.kind, style=palette.kind_style(span.kind))
    header.append(f"  ·  {span.status}", style=palette.status_style(span.status))
    header.append(f"  ·  {fmt_duration(span.duration)}  ·  {fmt_time(span.start)}", style="dim")
    if span.tokens:
        u = span.usage or {}
        header.append(
            f"\n{u.get('input_tokens', '?')} in  ·  {u.get('output_tokens', '?')} out"
            f"  ·  {span.tokens} total tokens",
            style=palette.accent,
        )

    parts: list[RenderableType] = [
        header,
        Rule("Input", style="dim"), _value(span.input, palette),
        Rule("Output", style="dim"), _value(span.output, palette),
    ]
    if span.error:
        parts += [Rule("Error", style=palette.accent), Text(span.error, style=palette.accent)]
    return Group(*parts)
