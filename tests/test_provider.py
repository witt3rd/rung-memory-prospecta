"""The provider over a real pgvector Postgres, with the fake model and embedder."""

from rung_memory_prospecta.contract import Budget, RecallCall, RetainCall

NOTE = "The deploy branch is release/x; never deploy from main."


def retain(p, scope, text, **attrs):
    return p.retain(RetainCall(scope=scope, kind="note", text=text, attrs=attrs))


def recall(p, scope, prompt, budget=None):
    return p.recall(RecallCall(scope=scope, prompt=prompt, context=(), budget=budget or Budget()))


def test_retain_then_recall_returns_the_whole_record_with_provenance(provider):
    out = retain(provider, "s1", NOTE, session="a", line="1")
    assert out["status"] == "stored" and out["id"] and out["cost_usd"] > 0
    got = recall(provider, "s1", "Which branch do we deploy from? deploy branch")
    assert got["calls"] >= 1 and got["cost_usd"] > 0
    [r] = got["records"]
    assert r["text"] == NOTE and r["id"] == out["id"]
    assert r["attrs"] == {"session": "a", "line": "1"}
    assert r["observed_at"].endswith("Z") and r["score"] > 0


def test_scopes_are_isolated_and_an_unknown_scope_is_empty(provider):
    retain(provider, "s-a", NOTE)
    assert recall(provider, "s-b", "deploy branch")["records"] == []
    assert recall(provider, "never-seen", "deploy branch")["records"] == []


def test_recall_stays_within_the_budget(provider):
    for i in range(4):
        retain(provider, "s-budget", f"deploy branch rule number {i}: " + "x" * 40)
    got = recall(provider, "s-budget", "deploy branch rule", Budget(max_records=2))
    assert len(got["records"]) == 2 and got["left_out"] == 2
    got = recall(provider, "s-budget", "deploy branch rule", Budget(max_chars=60))
    assert all(len(r["text"]) <= 60 for r in got["records"]) and len(
        {r["id"] for r in got["records"]}
    ) == len(got["records"])


def test_same_text_twice_is_one_record(provider):
    a = retain(provider, "s-dup", NOTE)["id"]
    b = retain(provider, "s-dup", NOTE)["id"]
    assert a == b
    assert len(recall(provider, "s-dup", "deploy branch")["records"]) == 1


def test_empty_observation_is_declined(provider):
    assert retain(provider, "s-e", "   ")["status"] == "declined"
