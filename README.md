# rung-memory-prospecta

[Prospecta](https://github.com/witt3rd/prospecta) as a [rung](https://github.com/witt3rd/rung) memory
provider: a small MCP server that speaks `rung-memory/1` (retain and recall) over the Prospecta library
(Postgres + pgvector, anticipated-question retrieval).

This is the plain provider: one retrieval method, one bank per scope. It is a building block and a
comparison point for richer providers, not a hybrid.

## How it maps

| rung | Prospecta |
|---|---|
| scope key (opaque) | one bank per scope: `s-` + first 32 hex of `sha256(scope)`; created on first retain |
| `rung_memory_retain` turn/note | `Memory.retain(text, source=…, metadata=…)`. A turn is `User: …\nAssistant: …`. The write-side LLM authors the anticipated questions. `metadata` carries `kind`, the observation `attrs` (provenance) and `observed_at`. The same text in the same scope replaces itself (same source), so a retry is idempotent |
| `rung_memory_recall` | `formulate_queries(prompt, context)` (read-side LLM) then `Memory.recall(queries)`. Hits are collapsed to one record per document (best score), ranked, and cut whole-record to the budget. A record is `{id, text (the original body), observed_at, attrs, score}`; `left_out` counts what the budget dropped |
| `calls`, `cost_usd` | every model and embedder call in the hook, costed by LiteLLM when it knows the price |

| *(no contract verb)* `memory_forget(id)` | `Memory.remove_documents` for that record. `rung-memory/1` has no delete concept, so this is an ordinary **agent tool**, not a hook: rung gives it to the agent unchanged. It acts on the scope of its session's latest hook call (rung sends agent tools no scope) |

| *(no contract verb)* `memory_search(query, depth)` | `Memory.search(query, depth=…)`. `depth` is `standard` (default) or `deep`; the tool description tells the agent to use `deep` for discovery questions (everything about X, all the …, every time …, lists and sets). Agent tool, bound to the session's latest hook scope |
| *(no contract verb)* `memory_gather(question, entity, aliases?, depth)` | `Memory.recall_mapreduce(…, depth=…)` (default `deep`) for set questions: a cited merged list, cut to the tool-result budget |

The automatic `rung_memory_recall` hook always passes `depth="standard"`. `depth` is a pure pass-through: its meaning (rerank pool cut) lives in Prospecta, see its `docs/recall-stages.md` "Recall depth".

An unknown scope recalls nothing and creates nothing. The repo is deliberately thin: no retrieval logic of its own. Hybrid retrieval belongs inside Prospecta.

## Run it

```bash
# a scratch pgvector (no volume, loopback). Name and port are overridable.
docker compose -f docker/compose.yml up -d --wait

pip install -e .                      # prospecta, LiteLLM, mcp, uvicorn (add .[st] for local MiniLM embeddings)
export PROSPECTA_DATABASE_URL=postgresql://rmp:rmp@127.0.0.1:55432/postgres   # needs CREATE EXTENSION vector
export OPENROUTER_API_KEY OPENAI_API_KEY      # by name: LiteLLM reads them from the environment
rung-memory-prospecta serve --port 9000       # mcp:http://127.0.0.1:9000/mcp

rung-agent --memory-check mcp:http://127.0.0.1:9000/mcp
```

Configuration is environment only; no key is read from, or written to, a file.

| env | default | |
|---|---|---|
| `PROSPECTA_DATABASE_URL` (or `DATABASE_URL`) | required | Postgres with pgvector; migrations run at start |
| `RMP_MODEL` | `openrouter/anthropic/claude-sonnet-5.5` | LiteLLM model for both spine calls; `fake` for offline |
| `RMP_EMBEDDER` | `litellm:openai/text-embedding-3-small` | `fake`, `st:<model>` (offline sentence-transformers) or `litellm:<model>`. A bank's dimension is fixed at creation: do not change embedders on a populated database |
| `RMP_MAX_COST_USD` | `0.02` | declared per-recall cost budget. rung treats a recall over it as unavailable |

## Tests and conformance

```bash
pip install -e '.[dev]'
export PROSPECTA_TEST_DATABASE_URL=postgresql://rmp:rmp@127.0.0.1:55432/postgres   # admin URL; each run uses its own scratch database
pytest
```

Everything is offline: a deterministic fake embedder and fake LLM stand in for the models, against a real
pgvector. The conformance test runs `rung-agent --memory-check` over HTTP against the server and requires
all 7 clauses to pass; it skips only when `rung-agent` is not on `PATH`. CI runs it (Postgres service
container, `rung-agent` built from the pinned rung tag `v0.1.13`, cached), so conformance is checked in CI.
Without `PROSPECTA_TEST_DATABASE_URL` the DB tests skip.

Live (real model) behaviour is not exercised by tests; it needs `OPENROUTER_API_KEY` (and an embedder key).

## Status

Prototype: the simplest thing that passes conformance. No auth, no hardening, no stdio transport, no
delete verb. Prospecta is pinned by commit in `pyproject.toml`.

### Cost

A model LiteLLM cannot price fails closed: its call raises `UnknownPrice` and the hook errors. Set `RMP_PRICE_INPUT_PER_MTOK` (and `RMP_PRICE_OUTPUT_PER_MTOK`) to price it yourself. A recall stays within the call's `max_cost_usd`: if the model step does not fit, the prompt is searched as is; if nothing fits, no records come back. A retain that does not fit the declared budget is `declined`.
