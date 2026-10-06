"""What ends up on disk: file permissions, the redact_fn hook, capture flags and retention."""

import os
import re
import stat
import sys
import time

import pytest

import deep_eye
from deep_eye import span, trace
from deep_eye.storage import list_runs


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path / "traces")
    return tmp_path / "traces"


def _all_text(folder) -> str:
    deep_eye.flush()
    return "".join(p.read_text(encoding="utf-8") for p in folder.glob("*.jsonl"))


# --- R1: private files ----------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_folder_and_files_are_owner_only(trace_dir):
    with span("x"):
        pass
    deep_eye.flush()
    assert stat.S_IMODE(trace_dir.stat().st_mode) == 0o700
    (f,) = trace_dir.glob("*.jsonl")
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_existing_open_folder_is_left_alone_with_a_warning(tmp_path, caplog):
    folder = tmp_path / "shared"
    folder.mkdir(mode=0o755)
    os.chmod(folder, 0o755)
    deep_eye.configure(trace_dir=folder)
    with span("x"):
        pass
    deep_eye.flush()
    assert stat.S_IMODE(folder.stat().st_mode) == 0o755
    assert sum("readable by other users" in r.message for r in caplog.records) == 1


def test_trace_folder_is_recreated_if_deleted_mid_run(trace_dir):
    with span("first"):
        pass
    deep_eye.flush()
    for f in trace_dir.glob("*.jsonl"):
        f.unlink()
    trace_dir.rmdir()
    with span("second"):
        pass
    assert [r.name for r in list_runs()] == ["second"]


# --- R2: redact_fn ----------------------------------------------------------------------

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def test_redact_fn_masks_inputs_outputs_and_errors(trace_dir):
    deep_eye.configure(redact_fn=lambda text: EMAIL.sub("<email>", text))

    @trace
    def lookup(user):
        return {"contact": "bob@example.com"}

    @trace
    def fail():
        raise ValueError("no account for carol@example.org")

    @trace
    def agent():
        lookup("alice@example.com")
        with span("meta", input={"cc": ["dave@example.net"]}) as s:
            s.output = "sent to erin@example.com"
        fail()

    with pytest.raises(ValueError, match="carol@example.org"):  # the program sees the real message
        agent()
    text = _all_text(trace_dir)
    assert not EMAIL.search(text)
    assert text.count("<email>") == 6  # carol twice: in fail and in agent


def test_redact_fn_runs_after_builtin_redaction_and_before_truncation(trace_dir):
    seen = []
    deep_eye.configure(max_chars=10, redact_fn=lambda t: seen.append(t) or t.upper())

    @trace
    def f(x):
        return None

    f("key sk-ant-0123456789abcdefghij and more text")
    assert seen == ["key [redacted] and more text"]
    assert list_runs()[0].root.input["x"] == "KEY [REDAC... [+18 chars]"


def test_failing_redact_fn_never_stores_the_raw_value(trace_dir, caplog):
    def broken(text):
        raise RuntimeError("bug")

    deep_eye.configure(redact_fn=broken)

    @trace
    def f(secret_ish):
        return "private output"

    f("private input")
    f("private input")
    text = _all_text(trace_dir)
    assert "private" not in text and "[redacted]" in text
    assert sum("redact_fn raised" in r.message for r in caplog.records) <= 1


def test_redact_false_also_turns_off_redact_fn():
    deep_eye.configure(redact=False, redact_fn=lambda t: "masked")

    @trace
    def f(x):
        return x

    f("raw")
    assert list_runs()[0].root.output == "raw"


# --- R3: capture flags ------------------------------------------------------------------

def test_capture_flags_keep_values_out_of_the_file(trace_dir):
    @trace(capture_input=False, capture_output=False)
    def secret(password_hint, data):
        raise KeyError(f"missing {data}")

    @trace(capture_input=False, capture_output=False)
    def stream(data):
        yield from data

    with pytest.raises(KeyError):
        secret("swordfish", "top-secret-arg")
    assert list(stream(["item-one", "item-two"])) == ["item-one", "item-two"]
    with span("block", input="block-input", capture_input=False, capture_output=False) as s:
        s.output = "block-output"

    text = _all_text(trace_dir)
    for value in ("swordfish", "top-secret-arg", "item-one", "block-input", "block-output"):
        assert value not in text
    runs = {r.name: r.root for r in list_runs()}
    assert runs["secret"].error == "KeyError" and runs["secret"].status == "error"
    assert runs["secret"].input == runs["secret"].output == "[not captured]"
    assert runs["stream"].output == "[not captured]" and runs["stream"].end is not None
    assert runs["block"].input == "[not captured]"


def test_capture_output_false_still_records_input():
    @trace(capture_output=False)
    def f(x):
        return "hidden"

    f(1)
    root = list_runs()[0].root
    assert root.input == {"x": 1} and root.output == "[not captured]"


# --- R4: retention ----------------------------------------------------------------------

def test_max_age_days_deletes_old_traces_on_new_run(trace_dir):
    trace_dir.mkdir()
    old, recent, other = trace_dir / "old.jsonl", trace_dir / "recent.jsonl", trace_dir / "notes.txt"
    for f in (old, recent, other):
        f.write_text("{}\n")
    week_ago = time.time() - 8 * 86400
    os.utime(old, (week_ago, week_ago))
    os.utime(other, (week_ago, week_ago))

    deep_eye.configure(max_age_days=7)
    with span("new"):
        pass
    assert not old.exists() and recent.exists() and other.exists()


def test_cleanup_does_not_follow_symlinks(trace_dir, tmp_path):
    trace_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n")
    week_ago = time.time() - 8 * 86400
    os.utime(outside, (week_ago, week_ago))
    try:
        (trace_dir / "link.jsonl").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks needs extra rights on this system")
    deep_eye.configure(max_age_days=7)
    with span("new"):
        pass
    assert outside.exists() and (trace_dir / "link.jsonl").is_symlink()
