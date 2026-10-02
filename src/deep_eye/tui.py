"""Interactive terminal viewer (Textual): list of runs -> span tree + detail."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree

from .models import Run, Span
from .render import fmt_duration, fmt_time, fmt_tokens, span_detail, span_label, status_markup
from .settings import load_settings, save_setting
from .storage import list_runs, load_run
from .themes import THEMES, Palette, get_theme, next_theme_name


class RunsScreen(Screen):
    """Table of all stored runs, newest first. Refreshes itself while open."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "filter", "Filter"),
        Binding("escape", "clear_filter", "Clear filter", show=False),
    ]

    def __init__(self, trace_dir: Path | None) -> None:
        super().__init__()
        self.trace_dir = trace_dir
        self.filter = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="filter by name, run id or status (Enter: apply, Esc: clear)", id="filter")
        yield DataTable(cursor_type="row", zebra_stripes=True)
        yield Footer()

    def _matches(self, run: Run) -> bool:
        needle = self.filter.lower()
        return not needle or any(needle in field.lower() for field in (run.name, run.run_id, run.status))

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.display = True
        box.focus()

    def action_clear_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.value = ""
        box.display = False
        self.query_one(DataTable).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter = event.value
        self.action_refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.query_one(DataTable).focus()

    def on_mount(self) -> None:
        self.query_one("#filter", Input).display = False
        table = self.query_one(DataTable)
        table.add_columns("Started", "Run", "Duration", "Tokens", "Spans", "Status")
        self.action_refresh()
        self.set_interval(2, self.action_refresh)  # pick up runs from a live agent
        table.focus()

    def action_refresh(self) -> None:
        palette: Palette = self.app.palette
        table = self.query_one(DataTable)
        row = table.cursor_row
        table.clear()
        for run in filter(self._matches, list_runs(self.trace_dir)):
            table.add_row(
                fmt_time(run.start),
                run.name,
                fmt_duration(run.duration),
                fmt_tokens(run.tokens),
                str(len(run.spans)),
                status_markup(run.status, palette),
                key=str(run.path),
            )
        if table.row_count:
            table.move_cursor(row=min(row, table.row_count - 1))

    restyle = action_refresh  # called by the app after a theme change

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(RunScreen(Path(event.row_key.value)))


class RunScreen(Screen):
    """Span tree on the left, details of the highlighted span on the right."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "reload", "Reload"),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._loaded: Run | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Tree("run", id="tree")
            with VerticalScroll(id="detail-pane"):
                yield Static(id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self.action_reload()
        self.query_one(Tree).focus()
        self.set_interval(1, self._reload_if_changed)  # follow a run that is still being written

    def _reload_if_changed(self) -> None:
        if load_run(self.path) is not self._loaded:  # load_run returns a new Run only when the file changed
            self.action_reload()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_reload(self) -> None:
        """(Re)build the tree from the file, keeping the cursor where it was."""
        run = self._loaded = load_run(self.path)
        tree = self.query_one(Tree)
        line = tree.cursor_line
        tree.clear()
        tree.root.set_label(run.name)
        self._add_children(run, tree.root, None)
        tree.root.expand_all()
        self.sub_title = f"{run.run_id} · {run.status}"
        if line > 0:
            tree.call_after_refresh(tree.move_cursor_to_line, line)  # tree is laid out lazily

    restyle = action_reload  # called by the app after a theme change

    def _add_children(self, run: Run, node, parent_id: str | None) -> None:
        for span in run.children_of(parent_id):
            child = node.add(span_label(span, self.app.palette), data=span, expand=True)
            self._add_children(run, child, span.span_id)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        span: Span | None = event.node.data
        if span is not None:
            self.query_one("#detail", Static).update(span_detail(span, self.app.palette))


class DeepEyeApp(App):
    TITLE = "deep-eye"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "cycle_theme", "Theme"),
    ]
    CSS = """
    #tree { width: 45%; border-right: solid $primary; }
    #detail-pane { padding: 0 1; }
    """

    def __init__(self, trace_dir: Path | None = None) -> None:
        super().__init__()
        self.trace_dir = trace_dir
        self.theme_name = get_theme(load_settings().get("theme")).name
        self.palette: Palette = get_theme(self.theme_name).palette

    def on_mount(self) -> None:
        for theme in THEMES.values():
            self.register_theme(theme.textual)
        self.theme = self.theme_name
        self.push_screen(RunsScreen(self.trace_dir))

    def action_cycle_theme(self) -> None:
        """Switch to the next theme, remember it, and redraw what is on screen."""
        self.theme_name = next_theme_name(self.theme_name)
        self.palette = get_theme(self.theme_name).palette
        self.theme = self.theme_name
        save_setting("theme", self.theme_name)
        restyle = getattr(self.screen, "restyle", None)
        if restyle:
            restyle()
        self.notify(f"Theme: {self.theme_name}", timeout=1.5)
