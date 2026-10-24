"""Depth is a pure pass-through to Prospecta: a fake Memory records what it is called with (no DB)."""

from rung_memory_prospecta import server
from rung_memory_prospecta.contract import Budget, RecallCall
from rung_memory_prospecta.provider import Provider, Settings, bank_id


class Hit:
    document_id, original_chunk, metadata, score = "d1", "a note", {}, 1.0


class Res:
    synthesis, citations = "the list [1]", ["d1"]


class FakeMemory:
    def __init__(self):
        self.calls = []

    def formulate_queries(self, prompt, context=None):
        return [prompt]

    def recall(self, queries, **kw):
        self.calls.append(("recall", kw))
        return [Hit()]

    def search(self, query, **kw):
        self.calls.append(("search", kw))
        return [Hit()]

    def recall_mapreduce(self, message, **kw):
        self.calls.append(("gather", kw))
        return Res()


def make():
    p = Provider.__new__(Provider)
    p.settings = Settings(database_url="x")
    mem = FakeMemory()
    p._memories = {bank_id("s"): mem}
    import threading

    p._lock = threading.Lock()
    p.meter = type("M", (), {"reset": lambda s, limit=None: None, "cost": 0.0, "calls": 1})()
    return p, mem


def test_hook_recall_is_standard():
    p, mem = make()
    p.recall(RecallCall(scope="s", prompt="q", context=(), budget=Budget()))
    assert mem.calls == [("recall", {"limit": 10, "depth": "standard"})]


def test_search_default_and_deep():
    p, mem = make()
    p.search("s", "q")
    p.search("s", "q", depth="deep")
    assert [c[1]["depth"] for c in mem.calls] == ["standard", "deep"]


def test_gather_passes_depth_and_returns_cited_result():
    p, mem = make()
    out = p.gather("s", "all the x", entity="X", aliases=["Ex"])
    p.gather("s", "all the x", entity="X", depth="standard")
    assert [c[1]["depth"] for c in mem.calls] == ["deep", "standard"]
    assert mem.calls[0][1]["entity"] == "X" and mem.calls[0][1]["aliases"] == ["Ex"]
    assert out == {"synthesis": "the list [1]", "citations": ["d1"], "truncated": False}


def test_gather_is_cut_to_the_tool_result_budget():
    p, _ = make()
    out = p.gather("s", "q", entity="X", budget=Budget(max_chars=4))
    assert out["synthesis"] == "the " and out["truncated"]


def test_tool_schemas_carry_depth():
    tools = {t.name: t for t in server.TOOLS}
    for name in ("memory_search", "memory_gather"):
        depth = tools[name].inputSchema["properties"]["depth"]
        assert depth["enum"] == ["standard", "deep"]
    assert (
        "deep" in tools["memory_search"].description
        and "everything about" in tools["memory_search"].description
    )
    assert tools["memory_search"].inputSchema["properties"]["depth"]["default"] == "standard"
    assert "set" in tools["memory_gather"].description.lower()
