"""The LangChain handler must nest correctly for both sync and async execution (no LLM calls)."""

import asyncio

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import deep_eye
from deep_eye import trace
from deep_eye.integrations.langchain import DeepEyeHandler
from deep_eye.storage import list_runs


@pytest.fixture(autouse=True)
def trace_dir(tmp_path):
    deep_eye.configure(trace_dir=tmp_path)


@trace
def helper(x):
    return x + 1


@trace
async def ahelper(x):
    await asyncio.sleep(0)
    return x * 2


@tool
def my_tool(x: int) -> int:
    """Add one using a traced helper."""
    return helper(x)


@tool
async def my_async_tool(x: int) -> int:
    """Double using a traced async helper."""
    return await ahelper(x)


def _shape(run):
    """[(name, depth)] in start order, to compare tree shapes."""
    depth = {None: -1}
    out = []
    for s in sorted(run.spans, key=lambda s: s.start):
        depth[s.span_id] = depth[s.parent_id] + 1
        out.append((s.name, depth[s.span_id]))
    return out


def test_sync_chain_nests_traced_helper_under_tool():
    my_tool.invoke({"x": 1}, config={"callbacks": [DeepEyeHandler()]})
    (run,) = list_runs()
    assert _shape(run) == [("my_tool", 0), ("helper", 1)]


def test_async_tool_nests_traced_async_helper():
    asyncio.run(my_async_tool.ainvoke({"x": 2}, config={"callbacks": [DeepEyeHandler()]}))
    (run,) = list_runs()
    assert _shape(run) == [("my_async_tool", 0), ("ahelper", 1)]


def test_async_chain_with_mixed_steps_is_one_run():
    chain = RunnableLambda(lambda x: helper(x)) | RunnableLambda(ahelper)
    assert asyncio.run(chain.ainvoke(1, config={"callbacks": [DeepEyeHandler()]})) == 4
    (run,) = list_runs()  # one run, not several
    names = [name for name, _ in _shape(run)]
    assert "helper" in names and "ahelper" in names
    assert all(depth > 0 or name == run.root.name for name, depth in _shape(run))


def test_retriever_becomes_a_span_with_documents():
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class FakeRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return [Document(page_content=f"about {query}", metadata={"id": 1})]

    FakeRetriever().invoke("cats", config={"callbacks": [DeepEyeHandler()]})
    span = list_runs()[0].spans[0]
    assert span.kind == "retriever"
    assert span.input == "cats"
    assert span.output == [{"content": "about cats", "metadata": {"id": 1}}]
