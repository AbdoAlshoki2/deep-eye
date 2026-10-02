# deep-eye

Local, file-based tracing for LLM agents, with an interactive terminal viewer.
No server, no Docker, no account.

```
your code  ──►  .deep-eye/*.jsonl  ──►  deep-eye   (terminal viewer)
```

![Run list](docs/runs.svg)

![Span tree and details](docs/run.svg)

## Install

```bash
pip install deep-eye
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
    return search(question)
```

Inputs, outputs, errors, timing and nesting are recorded automatically. Each
top-level call becomes one run, saved to `./.deep-eye/<run>.jsonl`
(override with `DEEP_EYE_DIR` or `deep_eye.configure(trace_dir=...)`).

### LangChain / LangGraph

```bash
pip install "deep-eye[langchain]"
```

```python
from deep_eye.integrations.langchain import DeepEyeHandler

agent.invoke(inputs, config={"callbacks": [DeepEyeHandler()]})
```

LLM calls, tool calls, retriever calls and graph nodes are traced; `@trace` helpers called inside
a tool nest under it.

## Explore traces

```bash
deep-eye                       # interactive viewer (Enter: open run, Esc: back, q: quit)
deep-eye list                  # plain table of runs
deep-eye show latest           # one run as a tree
deep-eye show <run-id> --details --no-color
deep-eye clear [path] [-y]       # delete all traces (default: the current trace dir)
```

## Development

```bash
pip install -e ".[langchain,dev]"
pytest
```
