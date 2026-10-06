"""Where traces go (R8), interrupted programs (R9) and half-written files (R10)."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

import deep_eye
from deep_eye import span, trace
from deep_eye.cli import main
from deep_eye.storage import list_runs, load_run


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path / "traces")
    return tmp_path / "traces"


def _run_script(code: str, cwd, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd,
                          env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=60)


# --- R8: the trace directory --------------------------------------------------------------

def test_configure_as_context_manager_restores_the_previous_folder(trace_dir, tmp_path):
    other = tmp_path / "other"
    with deep_eye.configure(trace_dir=other):
        with span("inside"):
            pass
    with span("after"):
        pass
    assert [r.name for r in list_runs(other)] == ["inside"]
    assert [r.name for r in list_runs(trace_dir)] == ["after"]


def test_open_run_keeps_its_file_when_the_folder_changes(trace_dir, tmp_path):
    with span("outer"):
        with deep_eye.configure(trace_dir=tmp_path / "other"):
            with span("inner"):
                pass
    (run,) = list_runs(trace_dir)
    assert len(run.spans) == 2 and not (tmp_path / "other").exists()


def test_env_var_is_read_when_a_run_starts(tmp_path, monkeypatch):
    from deep_eye import config

    monkeypatch.setattr(config, "_trace_dir", None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEP_EYE_DIR", str(tmp_path / "from-env"))

    @trace
    def f():
        pass

    f()
    assert len(list_runs(tmp_path / "from-env")) == 1
    monkeypatch.delenv("DEEP_EYE_DIR")
    f()
    assert len(list_runs(tmp_path / ".deep-eye")) == 1  # the default


def test_everything_goes_to_the_configured_folder(tmp_path, monkeypatch):
    pytest.importorskip("langchain_core")
    from langchain_core.runnables import RunnableLambda

    from deep_eye.integrations.langchain import DeepEyeHandler

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "chosen"
    with deep_eye.configure(trace_dir=folder):
        trace(lambda: 1)()
        with span("block"):
            pass
        RunnableLambda(lambda x: x + 1).invoke(1, config={"callbacks": [DeepEyeHandler()]})
    assert len(list_runs(folder)) == 3
    assert not (tmp_path / ".deep-eye").exists()


# --- R9: interrupts -----------------------------------------------------------------------

def test_ctrl_c_closes_every_span_and_is_reraised():
    @trace
    def inner():
        raise KeyboardInterrupt

    @trace
    def outer():
        with span("middle"):
            inner()

    with pytest.raises(KeyboardInterrupt):
        outer()
    (run,) = list_runs()
    assert all(s.end is not None for s in run.spans)
    assert all(s.status == "interrupted" for s in run.spans)
    assert run.status == "interrupted"
    inner_span = next(s for s in run.spans if s.name == "inner")
    assert inner_span.error == "KeyboardInterrupt"


def test_cancelled_task_is_interrupted_not_error():
    @trace
    async def slow():
        await asyncio.sleep(10)

    async def main_():
        task = asyncio.create_task(slow())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main_())
    (run,) = list_runs()
    assert run.status == "interrupted" and run.root.error == "CancelledError"


def test_tracing_failures_never_reach_the_program(trace_dir, monkeypatch, caplog):
    from deep_eye import storage

    def disk_full(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage, "_append", disk_full)
    monkeypatch.setattr("deep_eye.tracer._warned", False)

    @trace
    def f():
        return 42

    assert f() == 42 and f() == 42
    assert sum("failed to write" in r.message for r in caplog.records) == 1


def test_spans_left_open_at_exit_are_closed(tmp_path):
    result = _run_script("""
        import sys
        from deep_eye.tracer import start_span
        root = start_span("agent")
        start_span("step", parent=root)
        sys.exit(0)
    """, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    (run,) = list_runs(tmp_path / ".deep-eye")
    assert all(s.end is not None and s.interrupted for s in run.spans)
    assert run.status == "interrupted"


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM can't be delivered to a Python process")
def test_sigterm_closes_open_spans_then_terminates(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
        import time
        from deep_eye import span
        with span("agent"):
            print("ready", flush=True)
            time.sleep(60)
    """)], cwd=tmp_path, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == -signal.SIGTERM  # still dies the normal way
    (run,) = list_runs(tmp_path / ".deep-eye")
    assert run.root.interrupted and run.root.error == "SIGTERM"


# --- R10: incomplete files ----------------------------------------------------------------

def test_killed_process_shows_as_crashed(tmp_path):
    # sync_writes: with the background writer, a hard kill may lose the last queued events
    result = _run_script("""
        import os
        from deep_eye.tracer import start_span
        start_span("agent")
        os._exit(1)  # like kill -9: no cleanup at all
    """, cwd=tmp_path, env={"DEEP_EYE_SYNC_WRITES": "1"})
    assert result.returncode == 1
    (run,) = list_runs(tmp_path / ".deep-eye")
    assert run.status == "crashed" and run.span_status(run.root) == "unfinished"


def _write(path, lines: list[str], age: float = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    then = time.time() - age
    os.utime(path, (then, then))


def _start(span_id, parent=None, **extra) -> str:
    return json.dumps({"event": "start", "span_id": span_id, "parent_id": parent, "name": span_id,
                       "kind": "function", "start": time.time() - 100, "input": None, **extra}) + "\n"


def test_truncated_file_opens_and_is_marked_incomplete(trace_dir, capsys):
    path = trace_dir / "20240101-000000_abcdef.jsonl"
    full = _start("root", host="another-machine", pid=1) + _start("child", "root")
    _write(path, [full[:-25]], age=120)  # cut in the middle of the last line
    run = load_run(path)
    assert run.incomplete and run.status == "incomplete"
    assert [s.name for s in run.spans] == ["root"]
    main(["show", "latest"])
    assert "incomplete" in capsys.readouterr().out


def test_unfinished_run_from_another_machine_is_crashed_once_stale(trace_dir):
    path = trace_dir / "20240101-000000_abcdef.jsonl"
    _write(path, [_start("root", host="another-machine", pid=1)], age=5)
    assert load_run(path).status == "running"
    _write(path, [_start("root", host="another-machine", pid=1)], age=120)
    assert load_run(path).status == "crashed"
