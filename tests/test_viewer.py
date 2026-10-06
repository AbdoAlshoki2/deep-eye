import asyncio

import pytest
from textual.widgets import DataTable, Tree

import deep_eye
from deep_eye import trace
from deep_eye.cli import main
from deep_eye.tui import DeepEyeApp


@pytest.fixture(autouse=True)
def sample_run(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)

    @trace
    def helper():
        return "hi"

    @trace(name="my-agent", kind="agent")
    def agent():
        return helper()

    agent()
    return tmp_path


def test_cli_list_and_show(capsys, sample_run):
    main(["list"])
    assert "my-agent" in capsys.readouterr().out
    main(["show", "latest", "--details", "--no-color"])
    out = capsys.readouterr().out
    assert "my-agent" in out and "helper" in out and "hi" in out


def test_tui_opens_run_and_shows_tree(sample_run):
    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.query_one(DataTable).row_count == 1
            await pilot.press("enter")
            await pilot.pause()
            tree = app.screen.query_one(Tree)
            assert len(tree.root.children[0].children) == 1  # agent -> helper

    asyncio.run(scenario())


def test_theme_switch_cycles_and_is_remembered(tmp_path, monkeypatch):
    from deep_eye import settings

    monkeypatch.setattr(settings, "settings_path", lambda: tmp_path / "settings.json")

    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "crimson"
            await pilot.press("t")
            await pilot.pause()
            assert app.theme == "midnight"

    asyncio.run(scenario())
    assert settings.load_settings()["theme"] == "midnight"
    assert DeepEyeApp().theme_name == "midnight"  # next launch starts with it


def test_unknown_saved_theme_falls_back_to_default(tmp_path, monkeypatch):
    from deep_eye import settings

    monkeypatch.setattr(settings, "settings_path", lambda: tmp_path / "settings.json")
    settings.save_setting("theme", "does-not-exist")
    assert DeepEyeApp().theme_name == "crimson"


def test_settings_path_is_per_os(monkeypatch, tmp_path):
    from deep_eye import settings

    monkeypatch.setattr(settings.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(settings.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert settings.settings_path() == tmp_path / "Roaming" / "deep-eye" / "settings.json"
    monkeypatch.setattr(settings.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert settings.settings_path() == tmp_path / "xdg" / "deep-eye" / "settings.json"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert settings.settings_path() == tmp_path / ".config" / "deep-eye" / "settings.json"
    monkeypatch.setattr(settings.sys, "platform", "darwin")
    assert settings.settings_path() == tmp_path / "Library" / "Application Support" / "deep-eye" / "settings.json"


def test_clear_removes_only_trace_files_and_empty_folder(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    (d / "a.jsonl").write_text("{}\n")
    (d / "keep.txt").write_text("x")
    main(["clear", str(d), "-y"])
    assert not (d / "a.jsonl").exists() and (d / "keep.txt").exists()  # folder kept: not empty
    (d / "keep.txt").unlink()
    (d / "b.jsonl").write_text("{}\n")
    main(["clear", str(d), "-y"])
    assert not d.exists()
    main(["clear", str(d), "-y"])  # missing folder is not an error
    assert not d.exists()


def test_filter_hides_non_matching_runs(sample_run):
    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("slash", *"zzz")
            await pilot.pause()
            assert app.screen.query_one(DataTable).row_count == 0
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.query_one(DataTable).row_count == 1

    asyncio.run(scenario())


def test_open_run_picks_up_new_spans(sample_run):
    from deep_eye.storage import list_runs
    from deep_eye.tracer import start_span

    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            start_span("late", parent=list_runs()[0].root)  # a live agent appends to the open run
            await pilot.pause(1.5)
            tree = app.screen.query_one(Tree)
            assert len(tree.root.children[0].children) == 2

    asyncio.run(scenario())


# --- traced text is shown literally (R5) ------------------------------------------------

NASTY = ["[bold red]x[/]", "[@click=app.quit]x[/]", "\x1b[2J\x1b]0;title\x07"]


def _nasty_run() -> str:
    """Trace a run full of markup and escape codes; returns its run id."""
    from deep_eye.storage import list_runs

    @trace(name="[@click=app.quit]agent[/]")
    def agent(a, b, c):
        raise ValueError("\x1b[31mred[/]")

    try:
        agent(*NASTY)
    except ValueError:
        pass
    return next(r.run_id for r in list_runs() if r.name.startswith("[@click"))


def test_cli_shows_markup_and_escapes_literally(capsys):
    run_id = _nasty_run()
    main(["list"])
    main(["show", run_id, "--details"])
    out = capsys.readouterr().out
    assert "[@click=app.quit]agent[/]" in out and r"\x1b[31mred[/]" in out
    assert "\x1b[2J" not in out and "\x1b]0;" not in out and "\x07" not in out


def test_tui_survives_hostile_text():
    from rich.console import Console

    from deep_eye.render import span_detail

    _nasty_run()

    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.query_one(DataTable).row_count == 2
            for row in (0, 1):  # open both runs; one of them is the hostile one
                app.screen.query_one(DataTable).move_cursor(row=row)
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("escape")
            await pilot.press("enter")
            await pilot.pause()
            tree = app.screen.query_one(Tree)
            assert app.is_running
            return [str(n.label) for n in tree.root.children]

    labels = asyncio.run(scenario())
    assert labels

    from deep_eye.storage import list_runs
    run = next(r for r in list_runs() if r.name.startswith("[@click"))
    console = Console(record=True, width=200)
    console.print(span_detail(run.root, DeepEyeApp().palette, run))
    text = console.export_text()
    for value in ("[bold red]x[/]", "[@click=app.quit]x[/]", r"\u001b[2J\u001b]0;title\u0007"):
        assert value in text


# --- deleting a run with dd (R7) --------------------------------------------------------

def _delete_scenario(*keys: str, open_run: bool = False):
    async def scenario():
        app = DeepEyeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            if open_run:
                await pilot.press("enter")
                await pilot.pause()
            await pilot.press(*keys)
            await pilot.pause()
            return app.is_running, type(app.screen).__name__

    return asyncio.run(scenario())


def test_single_d_does_nothing(sample_run):
    assert _delete_scenario("d") == (True, "RunsScreen")
    assert len(list(sample_run.glob("*.jsonl"))) == 1


def test_dd_then_n_keeps_the_run(sample_run):
    assert _delete_scenario("d", "d", "x", "q", "n") == (True, "RunsScreen")  # x, q: ignored
    assert len(list(sample_run.glob("*.jsonl"))) == 1


def test_dd_then_y_deletes_the_run(sample_run):
    (sample_run / "notes.txt").write_text("keep")
    assert _delete_scenario("d", "d", "y") == (True, "RunsScreen")
    assert not list(sample_run.glob("*.jsonl")) and (sample_run / "notes.txt").exists()


def test_dd_inside_a_run_deletes_it_and_returns_to_the_list(sample_run):
    assert _delete_scenario("d", "d", "y", open_run=True) == (True, "RunsScreen")
    assert not list(sample_run.glob("*.jsonl"))


def test_delete_refuses_files_outside_the_trace_folder(sample_run, tmp_path):
    from deep_eye.storage import delete_run

    outside = tmp_path.parent / f"{tmp_path.name}-outside.jsonl"
    outside.write_text("{}\n")
    try:
        with pytest.raises(ValueError):
            delete_run(outside, sample_run)
        assert outside.exists()
    finally:
        outside.unlink()
