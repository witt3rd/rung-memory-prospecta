"""Bad model output, from recorded replies: it must never be indexed as questions or break a hook.

Runs without a database; the real Prospecta parsers (`generate_index_text`, `formulate_queries`) read
the replies after our guard.
"""

import json
import threading
from pathlib import Path

import pytest
from prospecta._formulate import formulate_queries
from prospecta._index_text import generate_index_text
from rung_memory_prospecta import provider as provider_mod
from rung_memory_prospecta.contract import Budget, RecallCall, RetainCall
from rung_memory_prospecta.provider import Provider, Settings, bank_id
from rung_memory_prospecta.questions import BadModelOutput, checked, questions_text

REC = json.loads((Path(__file__).parent / "recorded" / "bad_llm_outputs.json").read_text())


def _llm_returning(value):
    def llm(messages, *, json_mode=False):
        return value

    return llm


@pytest.mark.parametrize("name", REC["write_side"])
def test_bad_write_side_reply_is_rejected(name):
    with pytest.raises(BadModelOutput):
        questions_text(REC["write_side"][name])


@pytest.mark.parametrize("name", REC["write_side"])
def test_bad_write_side_reply_never_reaches_the_index(name):
    llm = checked(_llm_returning(REC["write_side"][name]))
    with pytest.raises(BadModelOutput):  # Prospecta is never handed a reply to index
        generate_index_text("The deploy branch is release/x.", llm)


@pytest.mark.parametrize("name", REC["write_side_ok_but_noisy"])
def test_noisy_but_usable_reply_gives_clean_questions(name):
    lines, _, _ = generate_index_text(
        "The deploy branch is release/x.", checked(_llm_returning(REC["write_side_ok_but_noisy"][name]))
    )
    assert lines == ["Which branch do we deploy from?", "Who may deploy?"]


def test_none_content_unguarded_would_be_indexed_as_a_question():
    """Why the guard exists: Prospecta itself turns None into the question 'None'."""
    lines, _, _ = generate_index_text("x", _llm_returning(None))
    assert lines == ["None"]


@pytest.mark.parametrize("name", REC["read_side_json"])
def test_bad_read_side_reply_falls_back_to_the_prompt(name):
    llm = checked(_llm_returning(REC["read_side_json"][name]))
    queries, outcome = formulate_queries("which branch do we deploy from?", llm=llm)
    assert [q.text for q in queries] == ["which branch do we deploy from?"]
    assert outcome.parse_fallback


def test_a_good_reply_passes_untouched():
    assert questions_text("What does this record say?") == "What does this record say?"


class _Mem:
    def __init__(self, llm):
        self.llm = llm

    def retain(self, text, source, metadata):
        generate_index_text(text, self.llm)
        return "doc"

    def formulate_queries(self, prompt, context=None):
        return formulate_queries(prompt, llm=self.llm, context=context)[0]

    def recall(self, queries, **kw):
        return []


def _provider(value, monkeypatch):
    p = Provider.__new__(Provider)
    p.settings = Settings(database_url="x")
    p.llm = checked(_llm_returning(value))
    p.embed = None
    p.meter = provider_mod.Meter()
    p._lock = threading.Lock()
    p._memories = {bank_id("s"): _Mem(p.llm)}
    return p


@pytest.mark.parametrize("name", REC["write_side"])
def test_retain_declines_instead_of_storing_garbage(name, monkeypatch):
    p = _provider(REC["write_side"][name], monkeypatch)
    out = p.retain(RetainCall(scope="s", kind="note", text="The deploy branch is release/x.", attrs={}))
    assert out["status"] == "declined" and "id" not in out


@pytest.mark.parametrize("name", REC["read_side_json"])
def test_recall_survives_bad_read_side_reply(name, monkeypatch):
    p = _provider(REC["read_side_json"][name], monkeypatch)
    out = p.recall(RecallCall(scope="s", prompt="deploy branch", context=(), budget=Budget()))
    assert out["records"] == []
