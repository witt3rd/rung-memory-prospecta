"""Offline stand-ins for the model and the embedder: deterministic, no network, no keys."""

from __future__ import annotations

import hashlib
import json
import math
import re

FAKE_DIM = 32
FAKE_COST_USD = 0.0001  # per call, so the cost path is exercised offline


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vec = [0.0] * FAKE_DIM
        for tok in _tokens(text):
            h = hashlib.sha256(tok.encode()).digest()
            vec[h[0] % FAKE_DIM] += 1.0 if h[1] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        out.append(vec)
    return out


def fake_llm(messages: list[dict], *, json_mode: bool = False) -> str:
    """Write side: the content's own words as a question. Read side: the message as the one query."""
    prompt = messages[-1]["content"]
    if json_mode:
        # The prompt quotes the user's message; the fake queries by the words of its last line of text.
        words = _tokens(prompt)[-12:]
        return json.dumps({"queries": [{"text": " ".join(words)}]})
    return "What does this record say?"
