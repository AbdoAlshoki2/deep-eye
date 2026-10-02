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
  written as it happens, so a crashed or still-running agent still leaves a
  readable (partial) trace.
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

Traces go to `./.deep-eye/<run>.jsonl`, relative to the folder you run your
code from. Add `.deep-eye/` to your `.gitignore`.

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
)
```

**Secret redaction.** Two kinds of values are replaced with `[redacted]` before
anything is written: values under keys or argument names containing `api_key`,
`password`, `secret`, `authorization`, `token` variants, `cookie`, and similar,
and strings that look like common API keys (`sk-...`, `AIza...`, `ghp_...`,
`Bearer ...`, AWS key ids). This is a best-effort safety net, not a guarantee.
See [Limitations](#limitations).

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
deep-eye clear [path] [-y]      # delete all *.jsonl traces
deep-eye clear --older-than 7   # delete traces older than 7 days
deep-eye --dir path/to/traces   # use another trace directory
```

Viewer keys: `Enter` opens a run, `Esc` goes back, `/` filters runs by name,
id or status, `r` reloads, `t` changes the theme, `q` quits. An open run
updates by itself while your agent is still writing to it.

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
- **No automatic cleanup.** Traces build up until you run `deep-eye clear`.
- **The viewer is built for hundreds of runs, not tens of thousands.** It
  re-reads only the files that changed, but it has no pagination or full-text
  search.
- **Traces stay on your machine.** There is no export to OpenTelemetry,
  LangSmith or other tools, and the JSONL format is internal and may change.
- **Early project.** It was developed and mostly used on Windows. There is no
  CI and no stability guarantee.

## Development

```bash
pip install -e ".[langchain,dev]"
pytest
```
