"""Cost: unknown prices fail closed (or use the operator's price), and a call's budget is respected.

These run without a database: Provider's Postgres edges are stubbed, the model and embedder are counting fakes.
"""

from types import SimpleNamespace

import pytest
from rung_memory_prospecta import provider as provider_mod
from rung_memory_prospecta.cli import settings_from_env
from rung_memory_prospecta.contract import Budget, RecallCall, RetainCall
from rung_memory_prospecta.live import Price, UnknownPrice, _cost, make_llm
from rung_memory_prospecta.meter import BudgetExceeded, Meter
from rung_memory_prospecta.provider import Provider, Settings


def _resp(tin=1000, tout=500):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


def _no_price(**_):
    raise Exception("This model isn't mapped yet")  # what LiteLLM raises for an unpriced model


def test_unknown_price_fails_closed(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: _resp())
    monkeypatch.setattr(litellm, "completion_cost", _no_price)
    with pytest.raises(UnknownPrice, match="RMP_PRICE_INPUT_PER_MTOK"):
        make_llm("brand/new-model", Meter())([{"role": "user", "content": "x"}])


def test_operator_price_is_used_when_litellm_has_none(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: _resp(1000, 500))
    monkeypatch.setattr(litellm, "completion_cost", _no_price)
    meter = Meter()
    meter.reset()
    llm = make_llm("brand/new-model", meter, Price(3.0, 15.0))
    assert llm([{"role": "user", "content": "x"}]) == "ok"
    assert meter.cost == pytest.approx((1000 * 3.0 + 500 * 15.0) / 1e6)
    assert llm.estimate_usd > 0


def test_known_price_comes_from_litellm(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0042)
    assert _cost(litellm, _resp(), "m", None) == 0.0042


def test_price_env():
    base = {"PROSPECTA_DATABASE_URL": "postgresql://x/y"}
    assert settings_from_env(base).price is None
    s = settings_from_env({**base, "RMP_PRICE_INPUT_PER_MTOK": "3", "RMP_PRICE_OUTPUT_PER_MTOK": "15"})
    assert s.price == Price(3.0, 15.0)


def test_meter_limit():
    m = Meter()
    m.reset(limit=0.01)
    m.check(0.005)
    m.add(0.008)
    with pytest.raises(BudgetExceeded):
        m.check(0.005)
    m.reset(limit=0.0)
    with pytest.raises(BudgetExceeded):
        m.check(0.0)
    m.reset()
    m.check(1e9)  # no limit: nothing refused


class _Mem:
    """Stands in for prospecta.Memory: formulate and recall each make one paid call, as the real one does."""

    def __init__(self, llm, embed, **_):
        self.llm, self.embed = llm, embed
        self.retained = []

    def formulate_queries(self, prompt, context=None):
        self.llm([{"role": "user", "content": prompt}], json_mode=True)
        return [provider_mod.Query(text=prompt)]

    def recall(self, queries, limit, depth):
        self.embed([q.text for q in queries])
        return []

    def retain(self, text, source, metadata):
        self.llm([{"role": "user", "content": text}])
        self.embed([text])
        return "doc-1"


@pytest.fixture
def counted(monkeypatch):
    calls = {"llm": 0, "embed": 0}
    monkeypatch.setattr(provider_mod, "run_migrations", lambda url: None)

    def memory(**kw):
        return _Mem(kw["llm"], kw["embed"])

    monkeypatch.setattr(provider_mod, "Memory", memory)

    def llm(messages, *, json_mode=False):
        calls["llm"] += 1
        return "{}"

    llm.estimate_usd = 0.004

    def embed(texts):
        calls["embed"] += 1
        return [[1.0, 0.0]]

    embed.estimate_usd = 0.0001
    p = Provider(Settings(database_url="x", model="m", embedder="e"), llm=llm, embed=embed)
    monkeypatch.setattr(p, "_memory", lambda scope, create: memory(llm=p.llm, embed=p.embed))
    calls["embed"] = 0
    return p, calls


def _recall(p, max_cost):
    return p.recall(
        RecallCall(scope="s", prompt="deploy branch", context=(), budget=Budget(max_cost_usd=max_cost))
    )


def test_recall_with_a_budget_below_one_model_call_skips_the_model(counted):
    p, calls = counted
    out = _recall(p, 0.001)  # the model call is estimated at 0.004
    assert calls["llm"] == 0  # no model call was made
    assert calls["embed"] == 1  # the prompt itself was searched
    assert out["cost_usd"] <= 0.001


def test_recall_with_a_budget_that_covers_it_makes_the_model_call(counted):
    p, calls = counted
    _recall(p, 0.01)
    assert calls["llm"] == 1 and calls["embed"] == 1


def test_recall_with_a_zero_budget_buys_nothing(counted):
    p, calls = counted
    out = _recall(p, 0.0)
    assert calls == {"llm": 0, "embed": 0}
    assert out["records"] == [] and out["cost_usd"] == 0


def test_retain_over_the_declared_budget_is_declined_not_half_done(counted):
    p, calls = counted
    p.settings.budget = Budget(max_cost_usd=0.001)
    out = p.retain(RetainCall(scope="s", kind="note", text="a fact", attrs={}))
    assert out["status"] == "declined" and calls["llm"] == 0
