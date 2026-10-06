"""rung-memory/1 over Prospecta: one bank per scope, retain -> Memory.retain, recall -> formulate + recall."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory

from . import fakes
from .contract import Budget, RecallCall, RetainCall, fit
from .meter import Meter


def bank_id(scope: str) -> str:
    """The scope key is opaque; a hash of it names the bank, so any key is a valid bank id."""
    return "s-" + hashlib.sha256(scope.encode()).hexdigest()[:32]


@dataclass
class Settings:
    database_url: str
    model: str = "openrouter/anthropic/claude-sonnet-5.5"
    embedder: str = "litellm:openai/text-embedding-3-small"
    budget: Budget = field(default_factory=Budget)


class Provider:
    def __init__(self, settings: Settings, llm=None, embed=None, meter: Meter | None = None) -> None:
        self.settings = settings
        self.meter = meter or Meter()
        if llm is None:
            if settings.model == "fake":
                llm = self._fake_llm()
            else:
                from .live import make_llm

                llm = make_llm(settings.model, self.meter)
        if embed is None:
            if settings.embedder == "fake":
                embed = self._fake_embed()
            else:
                from .live import make_embed

                embed = make_embed(settings.embedder, self.meter)
        self.llm, self.embed = llm, embed
        self.dim = len(embed(["dimension probe"])[0])
        run_migrations(settings.database_url)
        self._memories: dict[str, Memory] = {}
        self._lock = threading.Lock()

    def _fake_llm(self):
        def llm(messages, *, json_mode=False):
            self.meter.add(fakes.FAKE_COST_USD)
            return fakes.fake_llm(messages, json_mode=json_mode)

        return llm

    def _fake_embed(self):
        def embed(texts):
            self.meter.add(fakes.FAKE_COST_USD)
            return fakes.fake_embed(texts)

        return embed

    def _memory(self, scope: str, create: bool) -> Memory | None:
        bid = bank_id(scope)
        with self._lock:
            mem = self._memories.get(bid)
            if mem is None:
                mem = Memory(
                    database_url=self.settings.database_url, bank_id=bid, llm=self.llm, embed=self.embed
                )
                if create:
                    mem.create_bank(bid, embedding_dim=self.dim)
                else:
                    try:
                        mem.bank_stats(bid)
                    except ValueError:
                        mem.close()
                        return None
                self._memories[bid] = mem
            return mem

    def retain(self, call: RetainCall) -> dict:
        if not call.text.strip():
            return {"status": "declined", "reason": "nothing to retain"}
        self.meter.reset()
        mem = self._memory(call.scope, create=True)
        # Same text in the same scope is the same source: Prospecta replaces it rather than conflicting.
        source = "rung:" + hashlib.sha256(call.text.encode()).hexdigest()[:16]
        doc_id = mem.retain(
            call.text,
            source=source,
            metadata={
                "kind": call.kind,
                "attrs": call.attrs,
                "observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        return {"status": "stored", "id": doc_id, "cost_usd": self.meter.cost}

    def recall(self, call: RecallCall) -> dict:
        self.meter.reset()
        mem = self._memory(call.scope, create=False)
        records: list[dict] = []
        left_out = 0
        if mem is not None and call.prompt.strip():
            queries = mem.formulate_queries(call.prompt, context="\n".join(call.context) or None)
            hits = mem.recall(queries, limit=call.budget.max_records * 2)
            best: dict[str, dict] = {}
            for h in hits:  # one record per document: its best-scoring item
                if h.document_id not in best or h.score > best[h.document_id]["score"]:
                    meta = h.metadata or {}
                    best[h.document_id] = {
                        "id": h.document_id,
                        "text": h.original_chunk,
                        "observed_at": meta.get("observed_at", ""),
                        "attrs": meta.get("attrs", {}),
                        "score": h.score,
                    }
            ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)
            records, left_out = fit(ranked, call.budget)
        return {
            "records": records,
            "left_out": left_out,
            "cost_usd": self.meter.cost,
            "calls": max(self.meter.calls, 1),
        }

    def close(self) -> None:
        for m in self._memories.values():
            m.close()
