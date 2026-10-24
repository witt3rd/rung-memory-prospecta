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
    llm_timeout_secs: float = 4.0
    embed_timeout_secs: float = 3.0
    num_retries: int = 0


class Provider:
    def __init__(self, settings: Settings, llm=None, embed=None, meter: Meter | None = None) -> None:
        self.settings = settings
        self.meter = meter or Meter()
        if llm is None:
            if settings.model == "fake":
                llm = self._fake_llm()
            else:
                from .live import make_llm

                llm = make_llm(settings.model, self.meter, settings.llm_timeout_secs, settings.num_retries)
        if embed is None:
            if settings.embedder == "fake":
                embed = self._fake_embed()
            else:
                from .live import make_embed

                embed = make_embed(
                    settings.embedder, self.meter, settings.embed_timeout_secs, settings.num_retries
                )
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
            # Automatic per-turn recall is always `standard` (Prospecta's default, passed explicitly).
            hits = mem.recall(queries, limit=call.budget.max_records * 2, depth="standard")
            records, left_out = self._fit(hits, call.budget)
        return {
            "records": records,
            "left_out": left_out,
            "cost_usd": self.meter.cost,
            "calls": max(self.meter.calls, 1),
        }

    @staticmethod
    def _fit(hits, budget: Budget) -> tuple[list[dict], int]:
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
        return fit(sorted(best.values(), key=lambda r: r["score"], reverse=True), budget)

    def search(self, scope: str, query: str, depth: str = "standard", budget: Budget | None = None) -> dict:
        """Agent tool: Memory.search with the caller's `depth` passed straight through."""
        budget = budget or self.settings.budget
        mem = self._memory(scope, create=False)
        if mem is None or not query.strip():
            return {"records": [], "left_out": 0}
        hits = mem.search(query, limit=budget.max_records * 2, depth=depth)
        records, left_out = self._fit(hits, budget)
        return {"records": records, "left_out": left_out}

    def gather(
        self,
        scope: str,
        question: str,
        entity: str,
        aliases: list[str] | None = None,
        depth: str = "deep",
        budget: Budget | None = None,
    ) -> dict:
        """Agent tool: Memory.recall_mapreduce (the cited set answer), cut to the tool-result budget."""
        budget = budget or self.settings.budget
        mem = self._memory(scope, create=False)
        if mem is None:
            return {"synthesis": "", "citations": []}
        res = mem.recall_mapreduce(question, entity=entity, aliases=aliases, depth=depth)
        text = res.synthesis or ""
        citations = list(res.citations or [])
        cut = len(text) > budget.max_chars
        return {
            "synthesis": text[: budget.max_chars],
            "citations": citations[: budget.max_records * 10],
            "truncated": cut or len(citations) > budget.max_records * 10,
        }

    def forget(self, scope: str, document_id: str) -> dict:
        """Delete one record (a Prospecta document) from the scope's bank."""
        mem = self._memory(scope, create=False)
        if mem is None:
            return {"forgotten": 0}
        with mem._pool.cursor() as cur:
            cur.execute(
                "SELECT source FROM documents WHERE bank_id = %s AND id::text = %s",
                (bank_id(scope), document_id),
            )
            row = cur.fetchone()
        return {"forgotten": mem.remove_documents([row[0]]) if row else 0}

    def close(self) -> None:
        for m in self._memories.values():
            m.close()
