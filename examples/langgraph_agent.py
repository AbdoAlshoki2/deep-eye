"""A simple LangGraph agent with silly tools, powered by Gemini 3 via Google AI."""

import random
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from deep_eye import trace
from deep_eye.integrations.langchain import DeepEyeHandler

load_dotenv()


def _log(tool_name: str, msg: str) -> None:
    print(f"\n  [tool:{tool_name}] {msg}", flush=True)


# --- Hand-written helper functions (plain Python, not tools) ---

@trace
def _vowel_count(text: str) -> int:
    n = sum(1 for c in text.lower() if c in "aeiou")
    print(f"    [helper:_vowel_count] {text!r} -> {n}", flush=True)
    return n


@trace
def _pig_latin_word(word: str) -> str:
    out = word + "way" if word[0].lower() in "aeiou" else word[1:] + word[0] + "ay"
    print(f"    [helper:_pig_latin_word] {word!r} -> {out!r}", flush=True)
    return out


@trace
def _mock_case(text: str) -> str:
    out = "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(text))
    print(f"    [helper:_mock_case] {text!r} -> {out!r}", flush=True)
    return out


# --- Atomic tools: do one thing, no helpers ---

@tool
def roll_dice(sides: int = 6) -> int:
    """Roll a single die with the given number of sides."""
    _log("roll_dice", f"called with sides={sides}")
    result = random.randint(1, sides)
    _log("roll_dice", f"returning {result}")
    return result


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    _log("add", f"called with a={a}, b={b}")
    result = a + b
    _log("add", f"returning {result}")
    return result


@tool
def reverse_text(text: str) -> str:
    """Reverse a string."""
    _log("reverse_text", f"called with text={text!r}")
    result = text[::-1]
    _log("reverse_text", f"returning {result!r}")
    return result


# --- Composite tools: call hand-written helper functions ---

@tool
def pig_latin(sentence: str) -> str:
    """Translate a sentence into Pig Latin."""
    _log("pig_latin", f"called with sentence={sentence!r}")
    result = " ".join(_pig_latin_word(w) for w in sentence.split())
    _log("pig_latin", f"returning {result!r}")
    return result


@tool
def silly_text_report(text: str) -> dict:
    """Analyze text: vowel count, mocking-case version, and Pig Latin version."""
    _log("silly_text_report", f"called with text={text!r}")
    result = {
        "vowels": _vowel_count(text),
        "mocking": _mock_case(text),
        "pig_latin": pig_latin.invoke({"sentence": text}),  # tool calling a tool
    }
    _log("silly_text_report", f"returning {result}")
    return result


TOOLS = [roll_dice, add, reverse_text, pig_latin, silly_text_report]


def build_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    return create_agent(
        llm,
        TOOLS,
        system_prompt="You are a playful assistant. Use your tools whenever they apply.",
    )


def _text_parts(content) -> list[tuple[str, str]]:
    """Normalize message content into (kind, text) pairs; Gemma returns content blocks."""
    if isinstance(content, str):
        return [("text", content)] if content else []
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(("text", block))
        elif block.get("type") == "thinking":
            parts.append(("thinking", block.get("thinking", "")))
        elif block.get("type") == "text":
            parts.append(("text", block.get("text", "")))
    return parts


def stream_agent(agent, prompt: str) -> None:
    """Stream tokens, tool calls and tool results in real time."""
    current = None  # which kind of output we are mid-line in
    for chunk, meta in agent.stream(
        {"messages": [("user", prompt)]},
        config={"callbacks": [DeepEyeHandler()]},
        stream_mode="messages",
    ):
        kind = type(chunk).__name__

        if kind == "AIMessageChunk":
            for part_kind, text in _text_parts(chunk.content):
                if part_kind != current:
                    label = "💭 thinking" if part_kind == "thinking" else "🤖 agent"
                    print(f"\n\n{label}: ", end="", flush=True)
                    current = part_kind
                print(text, end="", flush=True)
            for tc in chunk.tool_call_chunks:
                if tc.get("name"):  # name only arrives on the first chunk of a call
                    print(f"\n\n🔧 calling {tc['name']}: ", end="", flush=True)
                    current = "tool_call"
                if tc.get("args"):
                    print(tc["args"], end="", flush=True)

        elif kind == "ToolMessage":
            print(f"\n\n📦 result from {chunk.name}: {chunk.content}", flush=True)
            current = None

    print()


if __name__ == "__main__":
    agent = build_agent()
    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Roll a d20, add the result to 100, then give me a silly text "
        "report of the phrase 'tracing agents is fun'."
    )
    print(f"👤 user: {prompt}")
    stream_agent(agent, prompt)
