"""The real model and embedder, through LiteLLM. Keys are read from the environment by LiteLLM, by name."""

from __future__ import annotations

import threading

from .meter import Meter

# rung gives each hook 10 s (`timeout_secs`); one overrun stops the provider for the rest of the run.
# Recall is an LLM call then an embedding, so the defaults leave room for the DB query.
DEFAULT_LLM_TIMEOUT_SECS = 4.0
DEFAULT_EMBED_TIMEOUT_SECS = 3.0
DEFAULT_NUM_RETRIES = 0  # a retry would spend the hook's time twice


class CallTimeout(TimeoutError):
    """A model or embedder call did not finish within its deadline."""


def call_with_deadline(fn, secs: float | None, what: str):
    """Run `fn()` and give up after `secs` wall-clock seconds, whatever the client library does.

    The library's own `timeout` bounds one HTTP request; this bounds the whole call. An abandoned call
    finishes (or fails) on its daemon thread and its result is dropped.
    """
    if not secs or secs <= 0:
        return fn()
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # re-raised on the caller's thread
            box["error"] = e

    t = threading.Thread(target=run, name=f"rmp-{what}", daemon=True)
    t.start()
    t.join(secs)
    if t.is_alive():
        raise CallTimeout(f"{what} call exceeded {secs:g}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def make_llm(
    model: str,
    meter: Meter,
    timeout: float = DEFAULT_LLM_TIMEOUT_SECS,
    num_retries: int = DEFAULT_NUM_RETRIES,
):
    import litellm

    def llm(messages: list[dict], *, json_mode: bool = False) -> str:
        kwargs: dict = {"model": model, "messages": messages, "timeout": timeout, "num_retries": num_retries}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = call_with_deadline(lambda: litellm.completion(**kwargs), timeout, "llm")
        meter.add(_cost(litellm, resp))
        return resp.choices[0].message.content

    return llm


def make_embed(
    spec: str,
    meter: Meter,
    timeout: float = DEFAULT_EMBED_TIMEOUT_SECS,
    num_retries: int = DEFAULT_NUM_RETRIES,
):
    """`st:<sentence-transformers model>` (offline weights) or `litellm:<model>`."""
    kind, _, model = spec.partition(":")
    if kind == "st":
        from prospecta.embed.sentence_transformers import sentence_transformers

        inner = sentence_transformers(model or "all-MiniLM-L6-v2")

        def embed(texts):
            meter.add(0.0)
            return inner(texts)

        return embed
    if kind == "litellm":
        import litellm

        def embed(texts):
            resp = call_with_deadline(
                lambda: litellm.embedding(model=model, input=texts, timeout=timeout, num_retries=num_retries),
                timeout,
                "embedding",
            )
            meter.add(_cost(litellm, resp))
            return [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]

        return embed
    raise ValueError(f"unknown embedder {spec!r}: use fake, st:<model> or litellm:<model>")


def _cost(litellm, resp) -> float:
    try:
        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # unknown model price: report no cost rather than fail the hook
        return 0.0
