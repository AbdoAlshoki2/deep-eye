"""Command line entry point: `deep-eye` (viewer), `list`, `show`, and `clear`."""

import argparse
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


def _find_run(ref: str) -> Run | None:
    runs = list_runs()
    if ref == "latest":
        return runs[0] if runs else None
    return next((r for r in runs if r.run_id.startswith(ref)), None)


def _cmd_show(console: Console, palette: Palette, ref: str, details: bool) -> None:
    run = _find_run(ref)
    if run is None:
        console.print(f"[red]No run matching '{ref}'[/]")
        return

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


def _cmd_clear(console: Console, target: Path, yes: bool) -> None:
    """Delete the trace files in `target`, then the folder itself if nothing else is left.

    Only *.jsonl files are removed, so pointing this at the wrong folder cannot wipe other data.
    """
    if not target.is_dir():
        console.print(f"[red]No trace directory at '{target}'[/]")
        return
    files = list(target.glob("*.jsonl"))
    if not files:
        console.print(f"No traces in '{target}'")
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
    args = parser.parse_args(argv)

    if args.dir:
        configure(trace_dir=Path(args.dir))
    if args.command == "clear":  # before the mkdir below: clearing must not create the folder
        _cmd_clear(Console(), Path(args.path) if args.path else get_trace_dir(), args.yes)
        return

    get_trace_dir().mkdir(parents=True, exist_ok=True)  # first run in a folder creates it
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
