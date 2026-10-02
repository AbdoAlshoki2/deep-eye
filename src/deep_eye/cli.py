"""Command line entry point: `deep-eye` (viewer), `list`, `show`, and `clear`."""

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from .config import configure, get_trace_dir
from .models import Run
from .render import fmt_duration, fmt_time, fmt_tokens, span_detail, span_label, status_markup
from .settings import load_settings
from .storage import list_runs
from .themes import Palette, get_theme


def _cmd_list(console: Console, palette: Palette) -> None:
    table = Table(box=None, header_style="bold")
    for column in ("Run", "Started", "Name", "Duration", "Tokens", "Spans", "Status"):
        table.add_column(column)
    for run in list_runs():
        table.add_row(run.run_id, fmt_time(run.start), run.name, fmt_duration(run.duration),
                      fmt_tokens(run.tokens), str(len(run.spans)), status_markup(run.status, palette))
    console.print(table)


def _find_runs(ref: str) -> list[Run]:
    """Runs matching `ref`: 'latest', an exact run id, or a run id prefix."""
    runs = list_runs()
    if ref == "latest":
        return runs[:1]
    exact = [r for r in runs if r.run_id == ref]
    return exact or [r for r in runs if r.run_id.startswith(ref)]


def _cmd_show(console: Console, palette: Palette, ref: str, details: bool) -> None:
    matches = _find_runs(ref)
    if not matches:
        console.print(f"[red]No run matching '{ref}'[/]")
        return
    if len(matches) > 1:
        console.print(f"[red]'{ref}' matches {len(matches)} runs; use a longer prefix:[/]")
        for r in matches[:10]:
            console.print(f"  {r.run_id}  {r.name}")
        return
    run = matches[0]

    def add(node: Tree, parent_id: str | None) -> None:
        for span in run.children_of(parent_id):
            add(node.add(span_label(span, palette)), span.span_id)

    tree = Tree(f"[bold]{run.name}[/]  [dim]{run.run_id}  ·  {fmt_tokens(run.tokens)} tokens[/]")
    add(tree, None)
    console.print(tree)
    if details:
        for span in sorted(run.spans, key=lambda s: s.start):
            console.print()
            console.print(span_detail(span, palette))


def _cmd_clear(console: Console, target: Path, yes: bool, older_than: float | None = None) -> None:
    """Delete the trace files in `target`, then the folder itself if nothing else is left.

    Only *.jsonl files are removed, so pointing this at the wrong folder cannot wipe other data.
    With `older_than` (days), only files last written before then are removed.
    """
    if not target.is_dir():
        console.print(f"[red]No trace directory at '{target}'[/]")
        return
    files = list(target.glob("*.jsonl"))
    if older_than is not None:
        cutoff = time.time() - older_than * 86400
        files = [f for f in files if f.stat().st_mtime < cutoff]
    if not files:
        console.print(f"No traces to delete in '{target}'")
    elif yes or input(f"Delete {len(files)} trace file(s) in '{target.resolve()}'? [y/N] ").lower() in ("y", "yes"):
        for f in files:
            f.unlink()
        console.print(f"Deleted {len(files)} trace file(s)")
    else:
        console.print("Cancelled")
        return
    try:
        target.rmdir()  # succeeds only when the folder is empty
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="deep-eye", description="Explore local agent traces.")
    parser.add_argument("--dir", help="trace directory (default: $DEEP_EYE_DIR or ./.deep-eye)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="print all runs as a table")
    show = sub.add_parser("show", help="print one run as a tree")
    show.add_argument("run", nargs="?", default="latest", help="run id prefix, or 'latest'")
    show.add_argument("--details", action="store_true", help="also print every span's input/output")
    show.add_argument("--no-color", action="store_true")
    clear = sub.add_parser("clear", help="delete all traces (and the trace folder)")
    clear.add_argument("path", nargs="?", help="trace directory (default: the current trace directory)")
    clear.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    clear.add_argument("--older-than", type=float, metavar="DAYS",
                       help="only delete traces last written more than DAYS days ago")
    args = parser.parse_args(argv)

    if args.dir:
        configure(trace_dir=Path(args.dir))
    if args.command == "clear":
        _cmd_clear(Console(), Path(args.path) if args.path else get_trace_dir(), args.yes, args.older_than)
        return

    palette = get_theme(load_settings().get("theme")).palette

    if args.command == "list":
        _cmd_list(Console(), palette)
    elif args.command == "show":
        _cmd_show(Console(no_color=args.no_color), palette, args.run, args.details)
    else:
        from .tui import DeepEyeApp  # imported lazily: Textual is slow to import

        DeepEyeApp().run()


if __name__ == "__main__":
    main()
