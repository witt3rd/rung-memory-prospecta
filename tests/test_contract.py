import pytest
from rung_memory_prospecta.contract import Budget, ContractError, fit, marker, read_recall, read_retain


def test_marker_declares_both_hooks_and_budget():
    m = marker(Budget())["rung-memory/1"]
    assert m["recall"] and m["retain"]
    assert m["budget"] == {"max_records": 5, "max_chars": 4000, "max_cost_usd": 0.02}


def test_a_call_can_only_tighten_the_budget():
    b = Budget().within({"max_records": 2, "max_chars": 99999, "max_cost_usd": -1})
    assert (b.max_records, b.max_chars, b.max_cost_usd) == (2, 4000, 0.02)


def test_retain_turn_and_note():
    t = read_retain(
        {"scope": "s", "observation": {"kind": "turn", "user": "q", "assistant": "a", "attrs": {"line": 3}}}
    )
    assert t.text == "User: q\nAssistant: a" and t.attrs == {"line": "3"}
    n = read_retain({"scope": "s", "observation": {"kind": "note", "text": " hi "}})
    assert n.text == "hi"
    with pytest.raises(ContractError):
        read_retain({"scope": "s", "observation": {"kind": "other"}})
    with pytest.raises(ContractError):
        read_recall({"prompt": "x"}, Budget())


def test_fit_keeps_whole_records_and_counts_the_rest():
    recs = [{"text": "a" * 30}, {"text": "b" * 30}, {"text": "c" * 5}]
    kept, left = fit(recs, Budget(max_records=5, max_chars=40))
    assert [r["text"][0] for r in kept] == ["a", "c"] and left == 1
    kept, left = fit(recs, Budget(max_records=1, max_chars=4000))
    assert len(kept) == 1 and left == 2
