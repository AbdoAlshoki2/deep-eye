"""Interactive terminal viewer (Textual): list of runs -> span tree + detail."""

import os
import time
from collections.abc import Callable
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree

from . import render
from .models import Run, Span
from .render import (clean, parse_rtl_mode, fmt_duration, fmt_time, fmt_tokens, plain, span_detail, span_label, status_text,
                     unsupported_text)
from .settings import load_settings, save_setting
from .storage import delete_run, list_runs, load_run
from .themes import THEMES, Palette, get_theme, next_theme_name

DOUBLE_PRESS = 0.5  # seconds between the two presses of `dd`


class ConfirmDelete(ModalScreen[bool]):
    """Asks before deleting a run. Only y, n and Esc do anything here."""

    BINDINGS = [
        Binding("y", "answer(True)", "Delete"),
        Binding("n", "answer(False)", "Cancel"),
        Binding("escape", "answer(False)", "Cancel", show=False),
        # shadow the app's own keys so a stray press can't quit or restyle behind the prompt
        Binding("q", "ignore", show=False),
        Binding("t", "ignore", show=False),
    ]
    DEFAULT_CSS = """
    ConfirmDelete { align: center middle; }
    ConfirmDelete > Static {
        width: auto; max-width: 80%; padding: 1 2;
        border: thick $error; background: $surface;
    }
    """

    def __init__(self, run: Run) -> None:
        super().__init__()
        self.run = run

    def compose(self) -> ComposeResult:
        text = Text("Delete this run permanently?\n\n", style="bold")
        text.append(clean(self.run.name))
        text.append(f"\n{clean(self.run.run_id)}", style="dim")
        if self.run.status == "running":
            text.append("\n\nThis run is still being written.", style="bold")
        text.append("\n\ny: delete    n / Esc: cancel", style="dim")
        yield Static(text)

    def action_answer(self, delete: bool) -> None:
        self.dismiss(delete)

    def action_ignore(self) -> None:
        pass


class DeleteRunMixin:
    """`dd` on a run asks to delete its trace file."""

    _last_d = 0.0

    def _delete_pressed(self, path: Path | None, after: Callable[[], None]) -> None:
        now = time.monotonic()
        if now - self._last_d > DOUBLE_PRESS:
            self._last_d = now  # first `d`: wait for the second one
            return
        self._last_d = 0.0
        if path is None:
            return

        def done(delete: bool | None) -> None:
            if not delete:
                return
            try:
                delete_run(path, self.app.trace_dir)
            except (OSError, ValueError) as exc:  # Windows refuses to delete a file that is open
                self.app.notify(f"Could not delete the run: {exc}", severity="error")
                return
            self.app.notify("Run deleted", timeout=1.5)
            after()

        self.app.push_screen(ConfirmDelete(load_run(path)), done)


class RunsScreen(DeleteRunMixin, Screen):
    """Table of all stored runs, newest first. Refreshes itself while open."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "filter", "Filter"),
        Binding("d", "delete", "Delete (dd)"),
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
        rows = [
            (
                fmt_time(run.start),
                plain(run.name),
                fmt_duration(run.duration),
                fmt_tokens(run.tokens),
                str(len(run.spans)),
                status_text(run.status, palette),
                str(run.path),
            )
            for run in filter(self._matches, list_runs(self.trace_dir))
        ]
        signature = [(*r[:5], str(r[5]), r[6]) for r in rows]
        if signature == getattr(self, "_signature", None) and palette is getattr(self, "_palette", None):
            return  # nothing changed: rebuilding would flash and jump the scroll position
        self._signature, self._palette = signature, palette
        row, scroll_y = table.cursor_row, table.scroll_y
        with self.app.batch_update():
            table.clear()
            for *cells, key in rows:
                table.add_row(*cells, key=key)
            if table.row_count:
                table.move_cursor(row=min(row, table.row_count - 1), scroll=False)  # after a delete: the next run
            table.scroll_to(y=scroll_y, animate=False, immediate=True)

    restyle = action_refresh  # called by the app after a theme change

    def action_delete(self) -> None:
        table = self.query_one(DataTable)
        path = None
        if table.row_count:
            path = Path(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        self._delete_pressed(path, self.action_refresh)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(RunScreen(Path(event.row_key.value)))


class RunScreen(DeleteRunMixin, Screen):
    """Span tree on the left, details of the highlighted span on the right."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "reload", "Reload"),
        Binding("d", "delete", "Delete (dd)"),
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
        elif self._loaded is not None:
            self._set_sub_title(self._loaded)  # a run can turn from running to crashed without a write

    def _set_sub_title(self, run: Run) -> None:
        self.sub_title = clean(f"{run.run_id} · {run.status}")

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_delete(self) -> None:
        def back_to_list() -> None:
            self.app.pop_screen()
            restyle = getattr(self.app.screen, "restyle", None)
            if restyle:
                restyle()

        self._delete_pressed(self.path, back_to_list)

    def action_reload(self) -> None:
        """(Re)build the tree from the file, keeping the cursor where it was."""
        run = self._loaded = load_run(self.path)
        tree = self.query_one(Tree)
        line = tree.cursor_line
        tree.clear()
        tree.root.set_label(plain(run.name))
        self._add_children(run, tree.root, None)
        tree.root.expand_all()
        self._set_sub_title(run)
        if not run.supported:
            self.query_one("#detail", Static).update(unsupported_text(run))
        if line > 0:
            tree.call_after_refresh(tree.move_cursor_to_line, line)  # tree is laid out lazily

    restyle = action_reload  # called by the app after a theme change

    def _add_children(self, run: Run, node, parent_id: str | None) -> None:
        for span in run.children_of(parent_id):
            child = node.add(span_label(span, self.app.palette, run), data=span, expand=True)
            self._add_children(run, child, span.span_id)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        span: Span | None = event.node.data
        if span is not None:
            self.query_one("#detail", Static).update(span_detail(span, self.app.palette, self._loaded))


class DeepEyeApp(App):
    TITLE = "deep-eye"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "cycle_theme", "Theme"),
        Binding("b", "toggle_rtl", "RTL"),
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
        render.rtl_mode = parse_rtl_mode(os.environ.get("DEEP_EYE_RTL") or load_settings().get("rtl"))

    def action_toggle_rtl(self) -> None:
        """Cycle right-to-left display: off -> words -> full (full needs deep-eye[rtl]); remembered."""
        modes = [m for m in render.RTL_MODES if m != "full" or render.full_rtl_available()]
        i = modes.index(render.rtl_mode) if render.rtl_mode in modes else -1
        render.rtl_mode = modes[(i + 1) % len(modes)]
        save_setting("rtl", render.rtl_mode)
        restyle = getattr(self.screen, "restyle", None)
        if restyle:
            restyle()
        self.notify(f"Right-to-left: {render.rtl_mode}", timeout=1.5)

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
