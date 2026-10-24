"""The real model and embedder, through LiteLLM. Keys are read from the environment by LiteLLM, by name."""

from __future__ import annotations

from dataclasses import dataclass

from .meter import Meter

# Rough size of one call, to decide before it runs whether the budget can pay for it.
EST_LLM_TOKENS = (1500, 400)
EST_EMBED_TOKENS = 500


class UnknownPrice(RuntimeError):
    """LiteLLM has no price for the model and the operator set none: the cost cannot be reported."""


@dataclass(frozen=True)
class Price:
    """Operator-set price, USD per million tokens, for a model LiteLLM cannot price."""

    input_per_mtok: float
    output_per_mtok: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int = 0) -> float:
        return (tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok) / 1e6


def make_llm(model: str, meter: Meter, price: Price | None = None):
    import litellm

    def llm(messages: list[dict], *, json_mode: bool = False) -> str:
        kwargs: dict = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(**kwargs)
        meter.add(_cost(litellm, resp, model, price))
        return resp.choices[0].message.content

    llm.estimate_usd = _estimate(litellm, model, price, *EST_LLM_TOKENS)
    return llm


def make_embed(spec: str, meter: Meter, price: Price | None = None):
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
            resp = litellm.embedding(model=model, input=texts)
            meter.add(_cost(litellm, resp, model, price))
            return [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]

        embed.estimate_usd = _estimate(litellm, model, price, EST_EMBED_TOKENS, 0)
        return embed
    raise ValueError(f"unknown embedder {spec!r}: use fake, st:<model> or litellm:<model>")


def _usage(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    g = (
        (lambda k: u.get(k) if isinstance(u, dict) else getattr(u, k, None))
        if u is not None
        else (lambda k: None)
    )
    return int(g("prompt_tokens") or 0), int(g("completion_tokens") or 0)


def _cost(litellm, resp, model: str, price: Price | None) -> float:
    """What the call cost. An operator price wins; else LiteLLM's; no price at all fails closed."""
    if price is not None:
        return price.cost(*_usage(resp))
    try:
        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception as e:
        raise UnknownPrice(
            f"no price for model {model!r}: set RMP_PRICE_INPUT_PER_MTOK (and RMP_PRICE_OUTPUT_PER_MTOK)"
        ) from e


def _estimate(litellm, model: str, price: Price | None, tokens_in: int, tokens_out: int) -> float:
    """Expected cost of one call, or 0.0 when it cannot be known (the call's real cost is then checked)."""
    if price is not None:
        return price.cost(tokens_in, tokens_out)
    try:
        p, c = litellm.cost_per_token(model=model, prompt_tokens=tokens_in, completion_tokens=tokens_out)
        return float(p + c)
    except Exception:
        return 0.0
