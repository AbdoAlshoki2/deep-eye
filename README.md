# deep-eye

Local, file-based tracing for LLM agents, with an interactive terminal viewer.
No server, no Docker, no account.

```
your code  ──►  .deep-eye/*.jsonl  ──►  deep-eye   (terminal viewer)
```

![Run list](docs/runs.svg)

![Span tree and details](docs/run.svg)

> **Heads-up:** I built deep-eye for my own use at work, to debug the agents I was
> building. It covers my workflow: single-process Python agents, mostly
> LangChain/LangGraph, run and inspected on my own machine. It is early (v0.1),
> not published on PyPI, and may not fit your setup. See
> [Limitations](#limitations) before relying on it.

## What it does

- `@trace` and `with span(...)` record each function's inputs, output, errors,
  timing and nesting. Each top-level call becomes one *run*.
- Each run is saved as one JSON Lines file in `./.deep-eye/`. Every event is
  written as it happens (by a background thread, so your code doesn't wait
  on the disk), so a crashed or still-running agent still leaves a readable
  (partial) trace.
- Tracing can be turned off or sampled with an environment variable, and
  `deep-eye export` turns traces into one JSON record per span for datasets
  and other tools.
- Common secrets (API keys, passwords, auth headers) are hidden before
  anything is written to disk.
- A LangChain/LangGraph callback handler traces LLM, tool, retriever and graph
  node calls, including token usage when the provider reports it.
- A terminal viewer (built with Textual) shows the list of runs, the span tree,
  and the full input/output of each span. A plain `list`/`show` CLI is also
  available.

## Install

Requires Python 3.10+. The package is not on PyPI, so install it from GitHub:

```bash
pip install "git+https://github.com/AbdoAlshoki2/deep-eye.git"

# with the LangChain / LangGraph integration
pip install "deep-eye[langchain] @ git+https://github.com/AbdoAlshoki2/deep-eye.git"
```

Or from a local clone:

```bash
git clone https://github.com/AbdoAlshoki2/deep-eye.git
cd deep-eye
pip install -e ".[langchain]"
```

## Trace your code

```python
from deep_eye import trace, span

@trace(kind="tool")
def search(query: str) -> str:
    ...

@trace(name="my-agent", kind="agent")
def run_agent(question: str):
    with span("plan", input=question) as s:
        s.output = "..."
        s.usage = {"input_tokens": 120, "output_tokens": 40}  # optional, shown in the viewer
    return search(question)
```

`@trace` works on normal functions, `async` functions, methods (without
recording `self`), and generators. A generator's span stays open until it
finishes, and its output is the list of yielded items.

To keep a function's data out of the trace but still see that it ran, turn
capture off. The span keeps its name, timing, nesting and status, and the
skipped values are stored as `"[not captured]"`. Without `capture_output`, an
exception is recorded by its type only, since messages often contain the data:

```python
@trace(capture_input=False, capture_output=False)
def load_patient_record(patient_id): ...

with span("decrypt", capture_output=False) as s: ...
```

Traces go to `./.deep-eye/<run>.jsonl`, relative to the folder you run your
code from. Add `.deep-eye/` to your `.gitignore`. To put them somewhere else,
set the `DEEP_EYE_DIR` environment variable, or use `configure()` (see
[Settings](#settings)).

If your program is stopped (Ctrl+C, task cancellation, `sys.exit`, `SIGTERM`),
the open spans are closed and marked `interrupted`, and the exception reaches
your code unchanged. If deep-eye itself fails to write (disk full, bad path,
...), it logs one warning and your program carries on.

### Overhead, sampling and turning it off

Traced code only takes a snapshot of each value (redacted and size-limited)
and queues it. A background thread writes the files, so traced calls, and
the asyncio event loop, never wait on the disk. The queue is written out when
the program exits. Call `deep_eye.flush()` if you read trace files from the
same process that writes them.

| Setting | Environment variable | Effect |
|---|---|---|
| `configure(enabled=False)` | `DEEP_EYE_ENABLED=0` | Turns tracing off. Traced functions run as if undecorated, and nothing is written. |
| `configure(sample_rate=0.1)` | `DEEP_EYE_SAMPLE_RATE=0.1` | Records 1 run in 10. The choice is made per run, so a run is always complete or absent. |
| `configure(sync_writes=True)` | `DEEP_EYE_SYNC_WRITES=1` | Writes each event before the traced code goes on. Slower, but nothing is lost if the process is killed (`kill -9`) or crashes in native code, where the background writer can lose the last few events. |

`configure()` arguments take precedence over environment variables. You can
leave the decorators in your code and switch tracing on or off per
environment without changing any code. If more than 10,000 events are waiting
to be written (the disk can't keep up), new events are dropped with one
warning instead of slowing your program down.

### Threads

Python doesn't pass the current span into new threads, so wrap the function
with `propagate()`. Otherwise each call shows up as a separate run:

```python
from concurrent.futures import ThreadPoolExecutor
from deep_eye import propagate

with ThreadPoolExecutor() as ex:
    results = list(ex.map(propagate(work), items))
```

`asyncio` tasks and LangChain's own threads don't need this.

### Settings

```python
import deep_eye

deep_eye.configure(
    trace_dir="traces",        # or set the DEEP_EYE_DIR environment variable
    max_chars=20_000,          # longest string kept before truncating
    max_items=200,             # most list items / dict entries kept per value
    redact=True,               # hide secrets (on by default)
    redact_keys=["ssn"],       # extra field names to hide
    redact_fn=mask_emails,     # your own redaction, see below
    max_age_days=7,            # delete older traces when a new run starts
)
```

`configure()` also works as a context manager. The settings apply inside the
block and the previous ones come back afterwards, which is handy for sending
each experiment or eval to its own folder:

```python
with deep_eye.configure(trace_dir="traces/eval-1"):
    run_agent()
```

The trace folder is looked up when a run starts: `configure(trace_dir=...)`
first, then `$DEEP_EYE_DIR`, then `./.deep-eye`. It applies to `@trace`,
`span(...)` and the LangChain handler alike. A run that is already open keeps
writing to its own file even if the folder changes.

**Secret redaction.** Two kinds of values are replaced with `[redacted]` before
anything is written: values under keys or argument names containing `api_key`,
`password`, `secret`, `authorization`, `token` variants, `cookie`, and similar,
and strings that look like common API keys (`sk-...`, `AIza...`, `ghp_...`,
`Bearer ...`, AWS key ids). This is a best-effort safety net, not a guarantee.
See [Limitations](#limitations).

**Your own redaction.** `redact_fn` is called with every string that is about
to be stored (span inputs, outputs, error messages, LangChain payloads, one
string at a time) and returns what to store instead. It runs after the
built-in redaction and before truncation. It gets only the text, not the key
it sits under; use `redact_keys` for key-based rules. If it raises, that value
is stored as `[redacted]` (never the raw text) and one warning is logged.
`redact=False` turns off all redaction, `redact_fn` included, which is handy
when you're debugging locally and want the raw values.

```python
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

def mask_emails(text: str) -> str:
    return EMAIL.sub("<email>", text)
```

### LangChain / LangGraph

```python
from deep_eye.integrations.langchain import DeepEyeHandler

agent.invoke(inputs, config={"callbacks": [DeepEyeHandler()]})
```

LLM calls (with token usage), tool calls, retriever calls and graph nodes are
traced. `@trace` helpers called inside a tool are nested under that tool's
span. `examples/langgraph_agent.py` is a full example. It uses Gemini, so it
needs `pip install -e ".[examples]"` and a Google API key in `.env`.

## Explore traces

```bash
deep-eye                        # interactive viewer
deep-eye list                   # plain table of runs
deep-eye show latest            # one run as a tree
deep-eye show <run-id-prefix> --details --no-color
deep-eye export -o spans.jsonl  # one JSON record per span (see Trace files)
deep-eye clear [path] [-y]      # delete all *.jsonl traces
deep-eye clear --older-than 7   # delete traces older than 7 days
deep-eye --dir path/to/traces   # use another trace directory
```

Viewer keys: `Enter` opens a run, `Esc` goes back, `/` filters runs by name,
id or status, `r` reloads, `t` changes the theme, `q` quits. An open run
updates by itself while your agent is still writing to it.

To delete a run, press `d` twice quickly (`dd`), either on a run in the list
or inside an open run, then `y` to confirm or `n`/`Esc` to cancel. It deletes
that run's `.jsonl` file only, and can't be undone.

Run statuses:

| Status | Meaning |
|---|---|
| `ok` | finished normally |
| `error` | finished, and at least one span raised an exception |
| `interrupted` | stopped by Ctrl+C, cancellation, `sys.exit` or `SIGTERM` |
| `running` | the process that writes it is still alive |
| `crashed` | the process died before finishing the run (e.g. `kill -9`) |
| `incomplete` | crashed in the middle of writing a line; the broken line is skipped |

Spans left open by a crashed run are shown as `unfinished`, with the time
until the last recorded event (`≥2.31s`).

## Trace files

### Export

The easiest way to use traces elsewhere (datasets, notebooks, other tools) is
`deep-eye export`. It writes **one self-contained JSON record per span**, with
the start and end events merged and the run's details added:

```bash
deep-eye export -o spans.jsonl                    # every span of every run
deep-eye export --kind llm --status ok -o llm.jsonl   # LLM calls from successful runs only
deep-eye export 20261002 latest                   # runs matching id prefixes, to stdout
deep-eye export --kind llm | jq -c '{prompt: .input, completion: .output}'
```

```json
{"v": 1, "run_id": "20261002-224252_a1b2c3", "run_name": "my-agent", "run_status": "ok", "span_id": "b2c3d4e5f6a7", "parent_id": "a1b2c3d4e5f6", "depth": 1, "name": "ChatGoogleGenerativeAI", "kind": "llm", "status": "ok", "start": 1760000000.1, "end": 1760000001.9, "duration": 1.8, "input": "...", "output": "...", "error": null, "usage": {"input_tokens": 120, "output_tokens": 40}}
```

`--kind` and `--status` can be repeated. From Python, use
`deep_eye.export.span_records(list_runs())` to get the same records as dicts,
for example `pandas.DataFrame(span_records(...))`.

### Raw format

Each run is one [JSON Lines](https://jsonlines.org/) file, and every span
writes two lines:

```json
{"v": 1, "event": "start", "span_id": "a1b2c3d4e5f6", "parent_id": null, "name": "my-agent", "kind": "agent", "start": 1760000000.0, "input": {"question": "..."}, "pid": 1234, "host": "my-laptop"}
{"v": 1, "event": "end", "span_id": "a1b2c3d4e5f6", "end": 1760000002.5, "output": "...", "error": null, "usage": {"input_tokens": 120, "output_tokens": 40}}
```

- `v` is the format version. It changes only when the format changes in a way
  that would break readers. Files from before versioning have no `v` and are
  otherwise the same as version 1.
- `parent_id` is `null` for the root span of the run. `pid` and `host` are
  only on the root span, and let the viewer tell running runs from crashed ones.
- `start`/`end` are Unix timestamps in seconds.
- `error` is `"Type: message"` (or just the type), and an end event of an
  interrupted span has `"interrupted": true`.
- A span without an end event never finished.
- The last line can be cut short if the process died while writing it, so
  skip lines that don't parse.

```python
import json, pathlib

for path in pathlib.Path(".deep-eye").glob("*.jsonl"):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

## Security

Traces are stored as **plain, unencrypted JSON**. That keeps them easy to
read, debug and feed into other tools, but anyone who can read the files can
read your prompts and model outputs. What deep-eye does about it:

- **File permissions.** On Linux and macOS, a new trace folder is created as
  `700` and each trace file as `600` (owner only). An existing folder is left
  as it is, with a one-time warning if other users can read it. Windows has
  no mode bits: keep traces under your user profile, or restrict the folder
  with `icacls`.
- **Redaction.** Built-in secret redaction, plus your own `redact_fn` (see
  [Settings](#settings)).
- **Capture control.** `capture_input=False` / `capture_output=False` keep a
  function's data out of the file entirely (see
  [Trace your code](#trace-your-code)).
- **Retention.** `configure(max_age_days=N)` deletes traces older than N
  days whenever a new run starts. Or schedule `deep-eye clear`:

  ```bash
  # cron (Linux/macOS): every day at 03:00
  0 3 * * * cd /path/to/project && deep-eye clear --older-than 7 -y
  ```

  ```powershell
  # Windows Task Scheduler: every day at 03:00
  schtasks /Create /SC DAILY /ST 03:00 /TN "deep-eye cleanup" /TR "cmd /c cd /d C:\path\to\project && deep-eye clear --older-than 7 -y"
  ```

  Cleanup only deletes `*.jsonl` files directly in the trace folder, and
  never follows symlinks.
- **Viewer.** Traced text is always shown literally: markup like `[bold]` and
  terminal escape codes are displayed, not executed.

Encryption at rest is not planned. On a single-user machine, disk encryption
plus the file permissions above cover the same risk. Don't trace production
traffic with sensitive data, and don't keep traces on shared or backed-up
storage you don't control.

## Limitations

What deep-eye still doesn't do:

- **One process only.** Spans from a child process, another service, or a job
  queue can't be linked to the parent run.
- **Threads need `propagate()`.** Without it, work sent to a thread or thread
  pool appears as separate runs. See [Threads](#threads).
- **LLM SDKs are not traced automatically.** Direct calls to the OpenAI,
  Anthropic, Google, etc. SDKs are only recorded if you wrap them in
  `@trace`/`span`. Token usage is recorded automatically only through the
  LangChain integration; otherwise set `s.usage` yourself.
- **LangChain only.** LlamaIndex, CrewAI, the OpenAI Agents SDK and other
  frameworks have no integration. In the LangChain integration, the top-level
  chain is always labelled `agent`. Only `langchain-core` 0.3+ is tested.
- **Redaction isn't complete.** It catches common key names and key formats,
  not every secret or personal detail. Prompts and model outputs are stored as
  plain text, so don't trace production traffic with sensitive data.
- **Large values are cut short.** Strings, lists and dicts over the size limits
  are truncated, and anything nested more than 8 levels deep is stored as
  `repr()` text. Objects that can't be turned into JSON are stored as `repr()`
  and can't be expanded in the viewer.
- **The viewer is built for hundreds of runs, not tens of thousands.** It
  re-reads only the files that changed, but it has no pagination or full-text
  search.
- **Traces stay on your machine.** There is no built-in export to
  OpenTelemetry, LangSmith or other tools. The files are easy to convert
  yourself (see [Trace files](#trace-files)), but the format may still change
  before 1.0.
- **Crash detection is per machine.** A run written from another machine
  (e.g. on a shared folder) counts as crashed once its file hasn't changed for
  60 seconds, even if it's only waiting on a slow call.
- **Early project.** It was developed and mostly used on Windows. There is no
  CI and no stability guarantee.

## Development

```bash
pip install -e ".[langchain,dev]"
pytest
```
