"""Color themes. Each theme has two halves that must agree:

* a Textual `Theme`  - colors the app chrome (header, table, borders, ...)
* a `Palette`        - colors the Rich text we draw ourselves (span labels, details)
"""

from dataclasses import dataclass

from textual.theme import Theme

from .schema import KINDS


@dataclass(frozen=True)
class Palette:
    """Rich style strings for the things we render ourselves."""

    agent: str
    chain: str
    llm: str
    tool: str
    retriever: str
    function: str
    ok: str
    running: str
    error: str  # whole-label style for failed spans / status
    accent: str  # tokens, error text, rules
    syntax: str  # Pygments theme for JSON blocks

    def kind_style(self, kind: str) -> str:
        """Style for a span kind; embeddings look like retrievers, the rest like functions."""
        if kind == "embedding":
            return self.retriever
        return getattr(self, kind) if kind in KINDS and hasattr(self, kind) else self.function

    def status_style(self, status: str) -> str:
        return {
            "ok": self.ok, "running": self.running, "unfinished": self.running,
            "interrupted": self.accent, "error": self.error, "crashed": self.error,
            "incomplete": self.error, "unsupported": self.error,
        }.get(status, self.function)


@dataclass(frozen=True)
class DeepEyeTheme:
    name: str
    textual: Theme
    palette: Palette


CRIMSON = DeepEyeTheme(
    name="crimson",
    textual=Theme(
        name="crimson", primary="#E60000", secondary="#BD0000", accent="#FF6B6B",
        foreground="#FFFFFF", background="#000000", surface="#111111", panel="#1C1C1C",
        warning="#FF9E9E", error="#E60000", success="#FFFFFF", dark=True,
    ),
    palette=Palette(
        agent="bold #E60000", chain="#9A9A9A", llm="bold #FFFFFF", tool="#FF6B6B", retriever="#FFB4B4",
        function="#D9D9D9", ok="#FFFFFF", running="#FF6B6B",
        error="bold #FFFFFF on #E60000", accent="#FF6B6B", syntax="ansi_dark",
    ),
)

MIDNIGHT = DeepEyeTheme(
    name="midnight",
    textual=Theme(
        name="midnight", primary="#4DA3FF", secondary="#2F6FB3", accent="#5EEAD4",
        foreground="#E6EDF7", background="#0B1020", surface="#121A2E", panel="#1A2540",
        warning="#FBBF24", error="#EF4444", success="#5EEAD4", dark=True,
    ),
    palette=Palette(
        agent="bold #4DA3FF", chain="#7F8AA3", llm="bold #E6EDF7", tool="#5EEAD4", retriever="#A78BFA",
        function="#A7B4CC", ok="#E6EDF7", running="#FBBF24",
        error="bold #FFFFFF on #DC2626", accent="#5EEAD4", syntax="ansi_dark",
    ),
)

PAPER = DeepEyeTheme(
    name="paper",
    textual=Theme(
        name="paper", primary="#C00000", secondary="#8F0000", accent="#B45309",
        foreground="#1A1A1A", background="#FAFAF7", surface="#EFEFEA", panel="#E4E4DD",
        warning="#B45309", error="#C00000", success="#2E7D32", dark=False,
    ),
    palette=Palette(
        agent="bold #C00000", chain="#6B6B6B", llm="bold #111111", tool="#B45309", retriever="#6D28D9",
        function="#444444", ok="#1A1A1A", running="#B45309",
        error="bold #FFFFFF on #C00000", accent="#B45309", syntax="ansi_light",
    ),
)

THEMES = {t.name: t for t in (CRIMSON, MIDNIGHT, PAPER)}
DEFAULT_THEME = CRIMSON.name


def get_theme(name: str | None) -> DeepEyeTheme:
    """Look a theme up by name, falling back to the default for unknown names."""
    return THEMES.get(name or "", THEMES[DEFAULT_THEME])


def next_theme_name(current: str) -> str:
    names = list(THEMES)
    return names[(names.index(current) + 1) % len(names)] if current in names else DEFAULT_THEME
