"""Live-path deadlines, offline: LiteLLM is replaced by a stub, so no network, DB or keys are needed."""

import sys
import time
import types

import pytest
from rung_memory_prospecta.cli import settings_from_env
from rung_memory_prospecta.live import CallTimeout, make_embed, make_llm
from rung_memory_prospecta.meter import Meter


def _stub(delay: float, seen: list):
    mod = types.ModuleType("litellm")

    def completion(**kw):
        seen.append(kw)
        time.sleep(delay)
        msg = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    def embedding(**kw):
        seen.append(kw)
        time.sleep(delay)
        return types.SimpleNamespace(data=[{"embedding": [1.0]}])

    mod.completion, mod.embedding = completion, embedding
    mod.completion_cost = lambda **kw: 0.001
    return mod


def test_slow_model_times_out_within_deadline(monkeypatch):
    seen: list = []
    monkeypatch.setitem(sys.modules, "litellm", _stub(5.0, seen))
    llm = make_llm("m", Meter(), timeout=0.3)
    t0 = time.monotonic()
    with pytest.raises(CallTimeout):
        llm([{"role": "user", "content": "hi"}])
    assert time.monotonic() - t0 < 2.0


def test_slow_embedder_times_out(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", _stub(5.0, []))
    embed = make_embed("litellm:m", Meter(), timeout=0.3)
    with pytest.raises(CallTimeout):
        embed(["x"])


def test_timeout_and_no_retries_are_passed_to_litellm(monkeypatch):
    seen: list = []
    monkeypatch.setitem(sys.modules, "litellm", _stub(0.0, seen))
    assert make_llm("m", Meter())([{"role": "user", "content": "hi"}]) == "ok"
    make_embed("litellm:m", Meter())(["x"])
    assert [(k["timeout"], k["num_retries"]) for k in seen] == [(4.0, 0), (3.0, 0)]


def test_settings_are_configurable():
    env = {
        "PROSPECTA_DATABASE_URL": "postgresql://x",
        "RMP_LLM_TIMEOUT_SECS": "2.5",
        "RMP_EMBED_TIMEOUT_SECS": "1",
        "RMP_NUM_RETRIES": "1",
    }
    s = settings_from_env(env)
    assert (s.llm_timeout_secs, s.embed_timeout_secs, s.num_retries) == (2.5, 1.0, 1)
    d = settings_from_env({"PROSPECTA_DATABASE_URL": "postgresql://x"})
    assert (d.llm_timeout_secs, d.embed_timeout_secs, d.num_retries) == (4.0, 3.0, 0)
