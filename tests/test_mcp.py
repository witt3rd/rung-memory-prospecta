"""The provider over streamable HTTP, through the reference MCP client."""

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from rung_memory_prospecta.contract import MARKER, RECALL_TOOL, RETAIN_TOOL

NOTE = "The deploy branch is release/x; never deploy from main."


def test_marker_tools_retain_and_recall(http_server):
    async def run():
        async with streamable_http_client(http_server) as (r, w, _), ClientSession(r, w) as s:
            init = await s.initialize()
            assert init.capabilities.experimental[MARKER] == {
                "recall": True,
                "retain": True,
                "budget": {"max_records": 5, "max_chars": 4000, "max_cost_usd": 0.02},
            }
            assert {t.name for t in (await s.list_tools()).tools} == {RECALL_TOOL, RETAIN_TOOL}
            kept = await s.call_tool(
                RETAIN_TOOL,
                {"scope": "m1", "observation": {"kind": "note", "text": NOTE, "attrs": {"session": "a"}}},
            )
            assert not kept.isError and kept.structuredContent["status"] == "stored"
            got = await s.call_tool(RECALL_TOOL, {"scope": "m1", "prompt": "deploy branch", "context": []})
            assert [x["text"] for x in got.structuredContent["records"]] == [NOTE]
            other = await s.call_tool(RECALL_TOOL, {"scope": "m2", "prompt": "deploy branch"})
            assert other.structuredContent["records"] == []
            bad = await s.call_tool(RETAIN_TOOL, {"scope": "m1", "observation": {"kind": "x"}})
            assert bad.isError

    anyio.run(run)
