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
