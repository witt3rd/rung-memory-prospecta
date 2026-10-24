"""The real model and embedder, through LiteLLM. Keys are read from the environment by LiteLLM, by name."""

from __future__ import annotations

from .meter import Meter


def make_llm(model: str, meter: Meter):
    import litellm

    def llm(messages: list[dict], *, json_mode: bool = False) -> str:
        kwargs: dict = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(**kwargs)
        meter.add(_cost(litellm, resp))
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            if json_mode:
                return ""  # Prospecta falls back to searching the prompt itself
            from .questions import BadModelOutput

            raise BadModelOutput("the model refused")
        return msg.content  # may be None: `questions.checked` deals with it

    return llm


def make_embed(spec: str, meter: Meter):
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
            meter.add(_cost(litellm, resp))
            return [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]

        return embed
    raise ValueError(f"unknown embedder {spec!r}: use fake, st:<model> or litellm:<model>")


def _cost(litellm, resp) -> float:
    try:
        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # unknown model price: report no cost rather than fail the hook
        return 0.0
