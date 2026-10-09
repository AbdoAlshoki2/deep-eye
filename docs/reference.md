# deep-eye reference

This is the full technical reference for deep-eye. For installation and the
first steps, refer to the [README](../README.md).

## Words in this document

| Word | Meaning |
|---|---|
| span | One unit of work: a function call, a tool call or an LLM call. A span has an input, an output, a start time and an end time. |
| run | All the spans of one top-level call. One run is one file. |
| trace file | The file of one run, in `./.deep-eye/<run>.jsonl`. |
| adapter | A module that changes the events of a framework into spans. |
| handle | The object that `start_span()` gives. You use it to close the span. |

## What deep-eye does

- It records the input, output, errors, time and nesting of each traced function.
- It writes each run to one [JSON Lines](https://jsonlines.org/) file. It
  writes each event immediately, in a background thread. Thus, a run that
  stops or crashes also has a trace file.
- It hides usual secrets (API keys, passwords, authorization headers) before
  it writes to the disk.
- It has adapters for LangChain, LangGraph, the OpenAI Agents SDK and
  OpenTelemetry. The adapters record LLM calls, tool calls, agents,
  retrievers and token usage.
- It shows the runs, the span tree and the full data of each span in a
  terminal viewer. It also has the `list`, `show` and `export` commands.

## Install deep-eye

You must have Python 3.10 or a later version. deep-eye is not on PyPI.

1. Install deep-eye from GitHub:

   ```bash
   pip install "git+https://github.com/AbdoAlshoki2/deep-eye.git"
   ```

2. If you use a framework adapter, install it with its extra. The extras are
   `langchain`, `openai-agents` and `otel`:

   ```bash
   pip install "deep-eye[langchain] @ git+https://github.com/AbdoAlshoki2/deep-eye.git"
   pip install "deep-eye[langchain,otel] @ git+https://github.com/AbdoAlshoki2/deep-eye.git"
   ```

To install from a local copy, do these steps:

```bash
git clone https://github.com/AbdoAlshoki2/deep-eye.git
cd deep-eye
pip install -e ".[langchain]"
```

`import deep_eye` does not import a framework. Each adapter needs only its
own extra. If the extra is not installed, the adapter shows the `pip`
command that installs it.

## Ways to connect deep-eye to your code

You can use deep-eye in many ways. Use the table to find the correct method.
You can use more than one method in the same program. Each method puts its
spans in the same span tree.

| Method | Use it when | Section |
|---|---|---|
| `@trace` decorator | You can change the function. The function is sync, async, a method or a generator. | [The @trace decorator](#the-trace-decorator) |
| `with span(...)` | You want to record a part of a function, or you know the input only inside the block. | [The span() block](#the-span-block) |
| `start_span()` and `end_span()` | The start and the end of the work are in different callbacks, threads or tasks. | [Spans from callbacks](#spans-from-callbacks) |
| `propagate()` | You send traced work to a thread or a thread pool. | [Threads](#threads) |
| `DeepEyeHandler` | You use LangChain or LangGraph. | [LangChain and LangGraph](#langchain-and-langgraph) |
| `openai_agents.install()` | You use the OpenAI Agents SDK. | [OpenAI Agents SDK](#openai-agents-sdk) |
| `otel.install()` | Your framework or LLM SDK sends OpenTelemetry spans. | [OpenTelemetry](#opentelemetry) |
| Your own adapter | Your framework has callbacks, but no OpenTelemetry support. | [Write an adapter](#write-an-adapter) |
| Environment variables | You want to change tracing without a change to the code. | [Settings](#settings) |
| `deep-eye export` or `span_records()` | You want to use the traces in a notebook, a dataset or another tool. | [Export](#export) |

## Frameworks that you can use with deep-eye

deep-eye works with all Python code. The framework list has three groups.

**1. Built-in adapters (tested).** deep-eye tests these adapters with fake
models. The tests do not use the network or API keys. The tests ran on
Python 3.10 and Python 3.13.

| Framework | Extra | Tested versions |
|---|---|---|
| LangChain and LangGraph (`langchain-core`) | `langchain` | 0.3.0 to 1.6.7 |
| OpenAI Agents SDK (`openai-agents`) | `openai-agents` | 0.21.0 to 0.23.1 |
| OpenTelemetry (`opentelemetry-sdk`) | `otel` | 1.20.0 to 1.45.1 |

- Other versions can work, but deep-eye does not test them.
- On Python 3.10 with `langchain-core` older than 1.5, `@trace` functions in
  *async* LangChain tools show as separate runs. Sync code has no problem.
- `openai-agents` releases older than 0.21 do not record an error when the
  model call fails.

**2. Frameworks that send OpenTelemetry spans (not tested by deep-eye).**
The `otel` adapter records the spans of all OpenTelemetry instrumentations.
It knows the GenAI semantic conventions, OpenInference and OpenLLMetry. Some
examples of frameworks and SDKs that have such instrumentations:

- Agent frameworks: LlamaIndex, CrewAI, DSPy, Haystack, smolagents, AutoGen,
  PydanticAI, Semantic Kernel, Google ADK.
- LLM SDKs: OpenAI, Anthropic, Google GenAI, Mistral, Groq, AWS Bedrock,
  Vertex AI, LiteLLM.
- Vector databases (with OpenLLMetry): Chroma, Pinecone, Qdrant, Weaviate.

deep-eye does not test these instrumentations. Their attribute names can
change. Make sure that the spans show correctly before you use them.

**3. All other Python code.** Use `@trace`, `span()` or `start_span()` in
plain Python, `asyncio`, threads, scripts, notebooks, tests and web servers
(for example FastAPI, Flask or Django). Each process writes its own runs.

## Trace your code

### Which pattern to use

| Your situation | Use | Why |
|---|---|---|
| A function you can change | `@trace` | It records the arguments, the return value and errors for you. |
| A part of a function (a step, a loop body, a library call) | `with span(...)` | You choose the block. You set the output yourself. |
| The input is known only inside the block | `with span(...)` and `s.input = ...` | deep-eye saves the input when the block ends. |
| Start and end are in different callbacks, threads or tasks | `start_span()` and `end_span()` | You hold the handle and close it later. |
| Work that runs in a thread | `propagate(fn)` | Threads do not get the current span by themselves. |
| A framework (LangChain, OpenAI Agents, OpenTelemetry) | An adapter | The adapter makes the spans for you. |

You can mix all of these in one program. Nested calls build one span tree.

`@trace` and `span()` are not two different tools. `@trace` is `span()` around a
whole function. It fills the input (the arguments) and the output (the return
value) for you. With `span()` you fill them.

### The @trace decorator

```python
from deep_eye import trace

@trace(kind="tool")
def search(query: str) -> str:
    ...

@trace(name="my-agent", kind="agent")
def run_agent(question: str):
    return search(question)
```

- Use it bare (`@trace`) or with arguments: `name=`, `kind=`, `attrs=`,
  `capture_input=`, `capture_output=`. The default name is the function name.
- It works on sync functions, `async` functions, methods and generators.
- For a method, deep-eye does not record `self`.
- The input is a dict of the arguments (with their names). The output is the
  return value.
- If the function raises an exception, the span records the error and the
  exception continues.
- The span of a generator stays open until the generator stops. Its output
  is the list of the items that the generator gave.

### The span() block

```python
from deep_eye import span

with span("plan", kind="llm", input=question) as s:
    s.output = "..."
    s.usage = {"input_tokens": 120, "output_tokens": 40}  # optional, the viewer shows it
```

`span()` gives you a span object, `s`. What deep-eye saves:

| You set | Saved? | Notes |
|---|---|---|
| `span("name", kind=, input=, attrs=)` | Yes | Written when the block starts. |
| `s.output = ...` | Yes | Written when the block ends. Set it before the block ends. |
| `s.usage = {...}` | Yes | Token counts. |
| `s.input = ...` | Yes | Written when the block ends. See below. |
| Any other attribute, for example `s.loaded = ...` | **No** | Python accepts it, but deep-eye ignores it. |
| An exception in the block | Yes | The span records the error. The exception continues. |

**If you know the input only inside the block**, assign it:

```python
with span("login") as s:
    name = input("Enter your name: ")
    s.input = name
    s.output = authenticate(name)
```

- deep-eye saves the new input when the block ends, also when the block raises
  an exception. If the process stops before the block ends (for example
  `kill -9`), the viewer shows no input for that span.
- Assign a new object. If you change the old input in place
  (`s.input["q"] = ...`), deep-eye does not see it.
- `capture_input=False` still applies. deep-eye writes `"[not captured]"`.

**To keep more than one result**, you have these choices:

```python
# 1. Put them in the output:
with span("login", input=name) as s:
    loaded = load(name)
    checked = validate(name)
    s.output = {"loaded": loaded, "validated": checked}

# 2. Give each step its own span, and see them as children:
@trace
def load(name): ...

@trace
def validate(name): ...

with span("login", input=name) as s:
    load(name)
    validate(name)
    s.output = "done"
```

Choice 2 is usually better. Each step has its own time, input, output and
error in the viewer.

Code that runs after the `with` block is not in the span. If it is a traced
call and no other span is open, it starts a new run.

### Spans with kinds

`kind` tells the viewer what the span is (`llm`, `tool`, `agent` and others).
Refer to [Span kinds](#span-kinds). A kind that is not in the list is stored
as `other`.

### Keep data out of the trace

To record that a function ran, but not its data, set `capture_input=False`
or `capture_output=False`. The span keeps its name, time, nesting and status.
deep-eye writes `"[not captured]"` in place of the data.

```python
@trace(capture_input=False, capture_output=False)
def load_patient_record(patient_id): ...

with span("decrypt", capture_output=False) as s: ...
```

Without `capture_output`, deep-eye records only the type of an exception.
The message of an exception often contains the data.

### Where the trace files go

deep-eye writes trace files to `./.deep-eye/`, in the folder where you start
your program. Add `.deep-eye/` to your `.gitignore` file. To use a different
folder, set the `DEEP_EYE_DIR` environment variable or use `configure()`.
Refer to [Settings](#settings).

### Spans from callbacks

Frameworks often tell you about work in two callbacks: one at the start and
one at the end. These callbacks can be in different threads. Use
`start_span()` at the start and `end_span()` at the end:

```python
from deep_eye import start_span, end_span

def on_tool_start(event):
    handles[event.id] = start_span(event.tool_name, kind="tool", input=event.args,
                                   parent=handles.get(event.parent_id))

def on_tool_end(event):
    end_span(handles.pop(event.id), output=event.result, usage=event.usage)
```

- `parent=` is the handle of the parent span. If you do not give `parent=`,
  deep-eye uses the current span (the `@trace` function or `span()` block
  around the call). `parent=None` starts a new run.
- A handle does not become the current span.
- `end_span()` also accepts `error=` (an exception or a message), `attrs=`
  and `end_time=`.
- If you know the `input=`, `name=` or `kind=` only at the end, give them to
  `end_span()`. They replace the values from `start_span()`.
- If you close a span two times, deep-eye ignores the second call. It also
  ignores an object that is not a handle. In the two conditions, it writes a
  debug log message.
- `start_span()` and `end_span()` never raise an exception.
- `attrs={...}` keeps more data, for example a model name or the IDs of a
  framework. You can use `attrs` with `start_span`, `end_span`, `span` and
  `@trace`. The viewer shows `attrs` below the input and the output.

Use these two functions when `span()` does not fit: when the work starts and
ends in different places, or when you need a late `input=`, `name=` or `kind=`
and also want to control the parent yourself. Example without callbacks:

```python
h = start_span("login")
try:
    name = input("Enter your name: ")
    result = authenticate(name)          # a traced call here does NOT nest under h
    end_span(h, output=result, input=name)
except BaseException as exc:
    end_span(h, error=exc)
    raise
```

A handle does not become the current span. To nest a call under it, give
`parent=h` to the `start_span()` of that call. For code in one place, `span()`
is simpler, because it nests the calls and closes the span for you.

`span()` and `@trace` use these two functions.

### Stops and failures

If your program stops because of Ctrl+C, a task cancellation, `sys.exit` or
`SIGTERM`, deep-eye closes the open spans. It marks them `interrupted`. Your
code gets the same exception as before.

If deep-eye cannot write (for example, the disk is full), it writes one
warning to the log. Your program continues.

### Threads

Python does not give the current span to a new thread. Put the function in
`propagate()` before you send it to a thread. If you do not do this, each
call shows as a separate run.

```python
from concurrent.futures import ThreadPoolExecutor
from deep_eye import propagate

with ThreadPoolExecutor() as ex:
    results = list(ex.map(propagate(work), items))
```

You do not need `propagate()` for `asyncio` tasks, for the threads of
LangChain, or for spans with an explicit `parent=` handle.

### Speed, sampling and how to stop tracing

Traced code only makes a copy of each value and puts it in a queue. A
background thread writes the files. Thus, traced calls and the `asyncio`
event loop do not wait for the disk. deep-eye writes the queue when the
program stops. If you read the trace files in the same process, call
`deep_eye.flush()` first.

| Setting | Environment variable | Result |
|---|---|---|
| `configure(enabled=False)` | `DEEP_EYE_ENABLED=0` | Stops tracing. Traced functions run as if they have no decorator. deep-eye writes nothing. |
| `configure(sample_rate=0.1)` | `DEEP_EYE_SAMPLE_RATE=0.1` | Records 1 run in 10. deep-eye decides for each run. Thus, a run is complete or it is not there. |
| `configure(sync_writes=True)` | `DEEP_EYE_SYNC_WRITES=1` | Writes each event before the traced code continues. This is slower. Use it when the process can be killed (`kill -9`) or can crash in native code. |

- `configure()` arguments have priority over the environment variables.
- You can keep the decorators in your code. Use the environment variables
  to start or stop tracing in each environment.
- If more than 10,000 events wait in the queue, deep-eye does not keep the
  new events. It writes one warning. Your program does not become slower.

### Settings

```python
import deep_eye

deep_eye.configure(
    trace_dir="traces",        # or set the DEEP_EYE_DIR environment variable
    max_chars=20_000,          # maximum length of a string before deep-eye cuts it
    max_items=200,             # maximum list items or dict entries in one value
    redact=True,               # hide secrets (on by default)
    redact_keys=["ssn"],       # more field names to hide
    redact_fn=mask_emails,     # your own redaction function, refer to the text below
    max_age_days=7,            # delete older trace files when a new run starts
)
```

You can also use `configure()` as a context manager. The settings apply only
in the block. After the block, the previous settings apply again. For
example, use this to put each test or evaluation in its own folder:

```python
with deep_eye.configure(trace_dir="traces/eval-1"):
    run_agent()
```

deep-eye finds the trace folder when a run starts. It uses the first of these
that you set:

1. `configure(trace_dir=...)`
2. `$DEEP_EYE_DIR`
3. `./.deep-eye`

This applies to `@trace`, `span()` and all adapters. An open run continues
to write to its file if the folder setting changes.

**Secret redaction.** deep-eye writes `[redacted]` in place of these values:

- Values with a key or argument name that contains `api_key`, `password`,
  `secret`, `authorization`, a `token` name, `cookie` or a similar word.
- Strings that look like usual API keys: `sk-...`, `AIza...`, `ghp_...`,
  `Bearer ...` and AWS key IDs.

This redaction does not find all secrets. Refer to [Limitations](#limitations).

**Your own redaction.** deep-eye calls `redact_fn` with each string before it
writes the string. The function gives the string to write.

- deep-eye calls it after the built-in redaction and before it cuts long strings.
- The function gets only the text, not the key of the text. For rules that
  use keys, use `redact_keys`.
- If the function raises an exception, deep-eye writes `[redacted]`. It
  never writes the original text. It writes one warning.
- `redact=False` stops all redaction, `redact_fn` also. Use it only for local
  debugging, when you must see the original values.

```python
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

def mask_emails(text: str) -> str:
    return EMAIL.sub("<email>", text)
```

## Framework adapters

Each adapter is in `deep_eye/integrations/<name>.py`. Each adapter needs its
own extra. All adapters do these things:

- They put `@trace` functions that run in a tool or step of the framework
  below that tool or step.
- They put a framework run that starts in a `@trace` function below that
  function.
- They never raise an exception into your program. If an adapter fails, it
  writes one warning. Your code continues.

### LangChain and LangGraph

Give the handler to each call:

```python
from deep_eye.integrations.langchain import DeepEyeHandler

agent.invoke(inputs, config={"callbacks": [DeepEyeHandler()]})
```

Or attach it one time to a runnable:

```python
agent = agent.with_config(callbacks=[DeepEyeHandler()])
```

deep-eye records LLM calls (with token usage), tool calls, retriever calls
and graph nodes. `examples/langgraph_agent.py` is a full example. It uses
Gemini. To run it, do `pip install -e ".[examples]"` and put a Google API
key in `.env`.

### OpenAI Agents SDK

```python
from deep_eye.integrations.openai_agents import install

install()                 # one time, before you run agents
Runner.run_sync(agent, "What's the weather in Cairo?")
```

- Each Agents SDK trace becomes a run. The run has agent, tool, LLM, handoff
  and guardrail spans, with token usage.
- `install()` adds deep-eye to the other trace processors of the SDK. If you
  also send traces to the OpenAI dashboard, this continues.
- `install(exclusive=True)` keeps the traces only on your computer.
- You can also add the processor yourself:
  `agents.add_trace_processor(DeepEyeProcessor())`.

### OpenTelemetry

```python
from deep_eye.integrations.otel import install

install()   # adds deep-eye to the global TracerProvider, or makes one
```

Each OpenTelemetry span becomes a deep-eye span. deep-eye finds the span
kind, input, output and token usage from these conventions:

- The OpenTelemetry GenAI semantic conventions (`gen_ai.*`).
- [OpenInference](https://github.com/Arize-ai/openinference)
  (`openinference.span.kind`, `input.value`, `llm.token_count.*`).
- [OpenLLMetry](https://github.com/traceloop/openllmetry) (`traceloop.*`).

Example with LlamaIndex. First, do `pip install openinference-instrumentation-llama-index`.
Then:

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from deep_eye.integrations.otel import install

install()                              # makes the global TracerProvider
LlamaIndexInstrumentor().instrument()  # the instrumentation uses it
```

- Call `install()` before the instrumentation, so that the instrumentation
  uses the provider of deep-eye.
- Other spans (HTTP requests, database queries) become `function` spans.
- deep-eye keeps all span attributes in `attrs`, with the OpenTelemetry trace
  ID and span ID.
- To record the spans of only one provider, give it: `install(my_tracer_provider)`.
- You can also add the processor yourself:
  `provider.add_span_processor(DeepEyeSpanProcessor())`.

### Write an adapter

An adapter changes the events of a framework into `start_span()` and
`end_span()` calls. Do these steps to make an adapter like the built-in ones:

1. Put the adapter in `deep_eye/integrations/<name>.py`.
2. Import the framework only in that module.
3. If the framework is not installed, raise
   `missing_framework("<name>", "<package>", "<extra>")`. This function is in
   `deep_eye.integrations`.
4. Add the extra to `pyproject.toml`.
5. Keep a map from the span IDs of the framework to deep-eye handles.
6. Give the handle of the parent span as `parent=`. If there is no parent,
   do not give `parent=`. Then the run goes below a `@trace` function.
7. Use the kinds in [Span kinds](#span-kinds). deep-eye stores a different
   kind as `other`, with your name for it.
8. Put each callback in `deep_eye.tracer.never_raises`. Then a bug in the
   adapter cannot stop the program of the user.
9. Optional: call `deep_eye.tracer.register_parent_resolver(fn)`. `fn` must
   give the handle of the framework span that runs now. Then `@trace`
   functions in the framework go below that span.
10. Test the adapter with recorded or fake events. Do not use the network or
    API keys.
11. Make sure that the trace files agree with the format:

    ```python
    from deep_eye.schema import validate_file

    for run in list_runs():
        assert validate_file(run.path) == []   # [] means no problems
    ```

## See the traces

```bash
deep-eye                        # interactive viewer
deep-eye list                   # table of runs
deep-eye show latest            # one run as a tree
deep-eye show <run-id-prefix> --details --no-color
deep-eye export -o spans.jsonl  # one JSON record per span, refer to "Export"
deep-eye clear [path] [-y]      # delete all *.jsonl trace files
deep-eye clear --older-than 7   # delete trace files older than 7 days
deep-eye --dir path/to/traces   # use a different trace folder
```

Viewer keys:

| Key | Action |
|---|---|
| `Enter` | Open a run. |
| `Esc` | Go back. |
| `/` | Filter the runs by name, ID or status. |
| `r` | Read the files again. |
| `t` | Change the theme. |
| `q` | Quit. |
| `d` `d` | Delete a run (press `d` two times quickly). |

An open run updates automatically while your agent writes to it.

To delete a run, do these steps:

1. Select the run in the list, or open the run.
2. Press `d` two times quickly.
3. Press `y` to delete the run. To cancel, press `n` or `Esc`.

> **CAUTION:** You cannot undo a delete. deep-eye deletes only the `.jsonl`
> file of that run.

Run statuses:

| Status | Meaning |
|---|---|
| `ok` | The run finished correctly. |
| `error` | The run finished, and one or more spans raised an exception. |
| `interrupted` | Ctrl+C, a cancellation, `sys.exit` or `SIGTERM` stopped the run. |
| `running` | The process that writes the run is alive. |
| `crashed` | The process stopped before the run finished (for example, `kill -9`). |
| `incomplete` | The process crashed while it wrote a line. deep-eye ignores the broken line. |
| `unsupported` | The file uses a newer format version. Install a newer deep-eye to see it. |

For a crashed run, the viewer shows the open spans as `unfinished`. It shows
the time until the last recorded event, for example `≥2.31s`.

## Trace files

### Export

To use the traces in a different tool (a dataset, a notebook), use
`deep-eye export`. It writes **one complete JSON record for each span**. Each
record contains the start event, the end event and the data of the run.

```bash
deep-eye export -o spans.jsonl                        # all spans of all runs
deep-eye export --kind llm --status ok -o llm.jsonl   # LLM calls from good runs only
deep-eye export 20261002 latest                       # runs with these ID prefixes, to stdout
deep-eye export --kind llm | jq -c '{prompt: .input, completion: .output}'
```

```json
{"v": 2, "run_id": "20261002-224252_a1b2c3", "run_name": "my-agent", "run_status": "ok", "span_id": "b2c3d4e5f6a7", "parent_id": "a1b2c3d4e5f6", "depth": 1, "name": "ChatGoogleGenerativeAI", "kind": "llm", "status": "ok", "start": 1760000000.1, "end": 1760000001.9, "duration": 1.8, "input": "...", "output": "...", "error": null, "usage": {"input_tokens": 120, "output_tokens": 40}, "attrs": null}
```

- `v` is the version of the trace format. Refer to the next section.
- You can use `--kind` and `--status` more than one time.
- The export is always JSON Lines. If the `-o` file name does not end with
  `.jsonl`, deep-eye shows a warning. It writes the file.
- In Python, `deep_eye.export.span_records(list_runs())` gives the same
  records as dicts. For example: `pandas.DataFrame(span_records(...))`.

### Raw format (version 2)

Each run is one [JSON Lines](https://jsonlines.org/) file. The first line is
the run header. Each span then writes two lines: a `start` line when it opens
and an `end` line when it closes.

```json
{"event": "run", "schema_version": 2, "run_id": "20261002-224252_a1b2c3", "start": 1760000000.0, "pid": 1234, "host": "my-laptop"}
{"event": "start", "span_id": "a1b2c3d4e5f6", "parent_id": null, "name": "my-agent", "kind": "agent", "start": 1760000000.0, "input": {"question": "..."}}
{"event": "start", "span_id": "b2c3d4e5f6a7", "parent_id": "a1b2c3d4e5f6", "name": "gpt-4o", "kind": "llm", "start": 1760000000.1, "input": "...", "attrs": {"model": "gpt-4o"}}
{"event": "end", "span_id": "b2c3d4e5f6a7", "end": 1760000001.9, "output": "...", "error": null, "usage": {"input_tokens": 120, "output_tokens": 40}}
{"event": "end", "span_id": "a1b2c3d4e5f6", "end": 1760000002.5, "output": "...", "error": null, "usage": null}
```

Run header:

| Field | Meaning |
|---|---|
| `schema_version` | The format version, an integer. It changes only when a change in the format can break readers. |
| `run_id` | The name of the file, without `.jsonl`. |
| `start` | The start time of the run (Unix time in seconds). |
| `pid`, `host` | The process that writes the run. The viewer uses them to find crashed runs. |

`start` line:

| Field | Meaning |
|---|---|
| `span_id` | The ID of the span. It is unique in the run. |
| `parent_id` | The ID of the parent span. It is `null` for the root span. Each run has one root span. |
| `name` | The name of the function, tool or model. |
| `kind` | One of the [span kinds](#span-kinds). |
| `start` | Unix time in seconds. |
| `input` | A JSON value (redacted and with a size limit). |
| `attrs` | Optional. An object with more data. |

`end` line:

| Field | Meaning |
|---|---|
| `span_id` | The span that the line closes. |
| `end` | Unix time in seconds. |
| `output` | A JSON value. |
| `error` | `"Type: message"`, only the type, only a message, or `null`. |
| `usage` | `null`, or token counts: `input_tokens`, `output_tokens`, `total_tokens`. Each is an optional integer. |
| `interrupted` | Optional. It is `true` when Ctrl+C, a cancellation, an exit or `SIGTERM` stopped the span. |
| `name`, `kind`, `input` | Optional. They replace the values of the `start` line. |
| `attrs` | Optional. deep-eye adds them to the `attrs` of the `start` line. |

Rules for readers:

- Ignore fields that you do not know. deep-eye can add optional fields
  without a change to `schema_version`.
- A span without an `end` line did not finish.
- The last line can be incomplete if the process stopped while it wrote the
  line. Ignore lines that are not valid JSON.
- Version 1 files (deep-eye 0.1) have no header line. Their lines have
  `"v": 1` or no version. `pid` and `host` are on the `start` line of the
  root span. deep-eye can read these files.
- If the `schema_version` is not known, the viewer, `show` and `export` show
  the version and tell you to install a newer deep-eye. They do not try to
  read the file.

To read the files yourself:

```python
import json, pathlib

for path in pathlib.Path(".deep-eye").glob("*.jsonl"):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

`deep_eye.schema.validate_file(path)` compares a file with this format. It
gives a list of problems. An empty list means that the file is correct.

### Span kinds

`kind` is always one of these values:

| Kind | Use |
|---|---|
| `agent` | An agent, or the top of a framework run. |
| `chain` | A pipeline step, a graph node or a workflow. |
| `llm` | A model call (chat, completion, transcription). |
| `tool` | A tool or function call that a model asked for. |
| `retriever` | A document search. |
| `embedding` | An embedding call. |
| `function` | All other code. This is the default for `@trace`. |
| `other` | All other kinds. `attrs.original_kind` keeps the original name. |

deep-eye stores a kind that is not in this list as `other`. Examples are
`@trace(kind="planner")` and the `handoff` span of a framework. The file then
contains `"attrs": {"original_kind": "planner"}`. The viewer shows the
original name.

## Security

deep-eye stores traces as **plain JSON without encryption**. Thus, the files
are easy to read and to use in other tools. But a person who can read the
files can read your prompts and model outputs. deep-eye does these things to
decrease this risk:

- **File permissions.** On Linux and macOS, deep-eye makes a new trace folder
  with mode `700` and each trace file with mode `600`. Only the owner can
  read them. deep-eye does not change a folder that exists. If other users
  can read that folder, deep-eye writes one warning. Windows does not use
  these modes. On Windows, keep traces in your user profile, or use `icacls`
  to limit access to the folder.
- **Redaction.** deep-eye has built-in secret redaction, and you can add your
  own `redact_fn`. Refer to [Settings](#settings).
- **Capture control.** `capture_input=False` and `capture_output=False` keep
  the data of a function out of the file. Refer to
  [Keep data out of the trace](#keep-data-out-of-the-trace).
- **Retention.** `configure(max_age_days=N)` deletes trace files older than N
  days when a new run starts. You can also run `deep-eye clear` on a
  schedule:

  ```bash
  # cron (Linux/macOS): each day at 03:00
  0 3 * * * cd /path/to/project && deep-eye clear --older-than 7 -y
  ```

  ```powershell
  # Windows Task Scheduler: each day at 03:00
  schtasks /Create /SC DAILY /ST 03:00 /TN "deep-eye cleanup" /TR "cmd /c cd /d C:\path\to\project && deep-eye clear --older-than 7 -y"
  ```

  The cleanup deletes only `*.jsonl` files directly in the trace folder. It
  never follows symlinks.
- **Viewer.** The viewer shows traced text exactly as it is. It does not
  apply markup (for example `[bold]`) or terminal escape codes.

deep-eye will not encrypt the files. On a computer with one user, disk
encryption and the file permissions above give the same protection.

> **CAUTION:** Do not trace production traffic that has sensitive data. Do
> not keep traces on shared storage or backups that you do not control.

## Limitations

deep-eye does not do these things:

- **It works in one process only.** It cannot connect spans from a child
  process, a different service or a job queue to the parent run.
- **Threads need `propagate()`.** Without it, work in a thread or a thread
  pool shows as separate runs. Refer to [Threads](#threads).
- **It does not trace LLM SDKs automatically.** deep-eye records direct calls
  to the OpenAI, Anthropic or Google SDKs only in these conditions:
  - A framework adapter makes the call.
  - An OpenTelemetry instrumentation of the SDK sends spans. Refer to
    [OpenTelemetry](#opentelemetry).
  - You put the call in `@trace` or `span()`. Then you must set `s.usage`
    yourself.
- **It has three adapters.** The adapters are for LangChain and LangGraph,
  the OpenAI Agents SDK and OpenTelemetry. deep-eye tests only the versions
  in [Frameworks that you can use with deep-eye](#frameworks-that-you-can-use-with-deep-eye).
  Other frameworks need an OpenTelemetry instrumentation or your own adapter.
  The LangChain adapter always shows the top-level chain as `agent`.
- **Redaction is not complete.** It finds usual key names and key formats.
  It does not find all secrets or personal data. deep-eye stores prompts and
  model outputs as plain text.
- **It cuts large values.** It cuts strings, lists and dicts that are larger
  than the limits. It stores data that is more than 8 levels deep as
  `repr()` text. It also stores objects that it cannot change into JSON as
  `repr()` text. You cannot expand these objects in the viewer.
- **The viewer is for hundreds of runs, not tens of thousands.** It reads
  only the files that changed. It has no pagination and no full-text search.
- **Traces stay on your computer.** deep-eye can read OpenTelemetry spans,
  but it cannot export to OpenTelemetry, LangSmith or other tools. You can
  change the files yourself. Refer to [Trace files](#trace-files). The format
  has a version. A change that can break readers gets a new `schema_version`.
- **It finds crashes only on one computer.** A run from a different
  computer (for example, on a shared folder) shows as crashed after its file
  does not change for 60 seconds. This also occurs when the run only waits
  for a slow call.
- **It is an early project.** It was made and used mostly on Windows. It
  has no CI and no stability guarantee.

## Development

```bash
pip install -e ".[langchain,openai-agents,otel,dev]"
pytest
```

The tests skip the adapters of the frameworks that are not installed.
