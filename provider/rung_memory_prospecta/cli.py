"""rung-memory-prospecta serve: configuration is by environment; keys by name only.

PROSPECTA_DATABASE_URL    Postgres+pgvector URL (or DATABASE_URL)            required
RMP_MODEL                 LiteLLM model, or `fake`   default openrouter/anthropic/claude-sonnet-5.5
RMP_EMBEDDER              fake | st:<model> | litellm:<model>   default litellm:openai/text-embedding-3-small
RMP_MAX_COST_USD          declared per-recall cost budget       default 0.02
RMP_LLM_TIMEOUT_SECS      deadline for one model call           default 4 (rung's hook limit is 10 s)
RMP_EMBED_TIMEOUT_SECS    deadline for one embedding call       default 3
RMP_NUM_RETRIES           LiteLLM retries per call              default 0
OPENROUTER_API_KEY, OPENAI_API_KEY ...  read by LiteLLM from the environment, never from a file
"""

from __future__ import annotations

import argparse
import os
import sys

from .contract import Budget
from .provider import Provider, Settings


def settings_from_env(env=os.environ) -> Settings:
    url = env.get("PROSPECTA_DATABASE_URL") or env.get("DATABASE_URL")
    if not url:
        raise SystemExit("PROSPECTA_DATABASE_URL (or DATABASE_URL) is required")
    return Settings(
        database_url=url,
        model=env.get("RMP_MODEL", "openrouter/anthropic/claude-sonnet-5.5"),
        embedder=env.get("RMP_EMBEDDER", "litellm:openai/text-embedding-3-small"),
        budget=Budget(max_cost_usd=float(env.get("RMP_MAX_COST_USD", "0.02"))),
        llm_timeout_secs=float(env.get("RMP_LLM_TIMEOUT_SECS", "4")),
        embed_timeout_secs=float(env.get("RMP_EMBED_TIMEOUT_SECS", "3")),
        num_retries=int(env.get("RMP_NUM_RETRIES", "0")),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rung-memory-prospecta")
    ap.add_argument("command", choices=["serve"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args(argv)
    import uvicorn

    from .server import http_app

    provider = Provider(settings_from_env())
    print(f"rung-memory-prospecta: serving mcp:http://{args.host}:{args.port}/mcp", file=sys.stderr)
    uvicorn.run(http_app(provider), host=args.host, port=args.port, log_level="warning")
    return 0
