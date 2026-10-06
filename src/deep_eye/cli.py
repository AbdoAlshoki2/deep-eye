"""Command line entry point: `deep-eye` (viewer), `list`, `show`, `export` and `clear`."""

import argparse
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from .config import configure, get_trace_dir
from .models import Run
from .render import fmt_duration, fmt_time, fmt_tokens, plain, span_detail, span_label, status_text
from .settings import load_settings
from .storage import list_runs, trace_files
from .themes import Palette, get_theme


def _cmd_list(console: Console, palette: Palette) -> None:
    table = Table(box=None, header_style="bold")
    for column in ("Run", "Started", "Name", "Duration", "Tokens", "Spans", "Status"):
        table.add_column(column)
    for run in list_runs():
        table.add_row(plain(run.run_id), fmt_time(run.start), plain(run.name), fmt_duration(run.duration),
                      fmt_tokens(run.tokens), str(len(run.spans)), status_text(run.status, palette))
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
        console.print(plain(f"No run matching '{ref}'", "red"))
        return
    if len(matches) > 1:
        console.print(plain(f"'{ref}' matches {len(matches)} runs; use a longer prefix:", "red"))
        for r in matches[:10]:
            console.print(plain(f"  {r.run_id}  {r.name}"))
        return
    run = matches[0]

    def add(node: Tree, parent_id: str | None) -> None:
        for span in run.children_of(parent_id):
            add(node.add(span_label(span, palette, run)), span.span_id)

    title = plain(run.name, "bold")
    title.append(plain(f"  {run.run_id}  ·  {run.status}  ·  {fmt_tokens(run.tokens)} tokens", "dim"))
    tree = Tree(title)
    add(tree, None)
    console.print(tree)
    if details:
        for span in sorted(run.spans, key=lambda s: s.start):
            console.print()
            console.print(span_detail(span, palette, run))


def _cmd_export(refs: list[str], output: str | None, kinds: list[str] | None,
                statuses: list[str] | None) -> None:
    """Write one JSON record per span to a file or stdout (all runs, or those matching `refs`)."""
    from .export import span_records, write_jsonl

    if refs:
        runs = {r.run_id: r for ref in refs for r in _find_runs(ref)}.values()  # prefixes may overlap
    else:
        runs = list_runs()
    records = span_records(sorted(runs, key=lambda r: r.start), kinds, statuses)  # oldest first
    if output is None:
        sys.stdout.flush()
        try:
            write_jsonl(records, sys.stdout.buffer)
            sys.stdout.buffer.flush()
        except OSError:  # the reader stopped early (`| head`); Windows reports EINVAL, not EPIPE
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # silence the exit flush
        return
    with open(output, "wb") as f:
        count = write_jsonl(records, f)
    print(f"Wrote {count} span record(s) to {output}", file=sys.stderr)


def _cmd_clear(console: Console, target: Path, yes: bool, older_than: float | None = None) -> None:
    """Delete the trace files in `target`, then the folder itself if nothing else is left.

    Only *.jsonl files are removed (never through a symlink), so pointing this at the wrong
    folder cannot wipe other data. With `older_than` (days), only files last written before
    then are removed.
    """
    if not target.is_dir():
        console.print(plain(f"No trace directory at '{target}'", "red"))
        return
    files = trace_files(target)
    if older_than is not None:
        cutoff = time.time() - older_than * 86400
        files = [f for f in files if f.stat().st_mtime < cutoff]
    if not files:
        console.print(plain(f"No traces to delete in '{target}'"))
    elif yes or input(f"Delete {len(files)} trace file(s) in '{target.resolve()}'? [y/N] ").lower() in ("y", "yes"):
        deleted = 0
        for f in files:
            try:
                f.unlink()
                deleted += 1
            except OSError as exc:  # e.g. still open by a running agent on Windows
                console.print(plain(f"Could not delete {f.name}: {exc}", "red"))
        console.print(f"Deleted {deleted} trace file(s)")
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
    export = sub.add_parser("export", help="write one JSON record per span (JSON Lines), for datasets and other tools")
    export.add_argument("runs", nargs="*", metavar="RUN",
                        help="run id prefixes or 'latest' (default: all runs)")
    export.add_argument("-o", "--output", metavar="FILE", help="write to FILE instead of stdout")
    export.add_argument("--kind", action="append", metavar="KIND",
                        help="only spans of this kind, e.g. llm or tool (repeatable)")
    export.add_argument("--status", action="append", metavar="STATUS",
                        help="only runs with this status, e.g. ok (repeatable)")
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
    if args.command == "export":
        _cmd_export(args.runs, args.output, args.kind, args.status)
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
