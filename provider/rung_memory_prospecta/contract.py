"""The `rung-memory/1` contract, as this provider reads and answers it.

Source of truth: rung's `docs/rung-memory.md` (rung v0.1.13).
"""

from __future__ import annotations

from dataclasses import dataclass

MARKER = "rung-memory/1"
RECALL_TOOL = "rung_memory_recall"
RETAIN_TOOL = "rung_memory_retain"

MAX_PROMPT_CHARS = 2000
MAX_CONTEXT_ITEMS = 3
MAX_CONTEXT_CHARS = 500
MAX_TURN_SIDE_CHARS = 2000


class ContractError(ValueError):
    """A hook call that does not have the shape the contract gives it."""


@dataclass(frozen=True)
class Budget:
    max_records: int = 5
    max_chars: int = 4000
    max_cost_usd: float = 0.02

    def declared(self) -> dict:
        return {
            "max_records": self.max_records,
            "max_chars": self.max_chars,
            "max_cost_usd": self.max_cost_usd,
        }

    def within(self, asked: object) -> Budget:
        """The tighter of this (declared) budget and the one a call asks for."""
        if not isinstance(asked, dict):
            return self

        def tighter(key, mine, cast):
            v = asked.get(key)
            if isinstance(v, bool) or not isinstance(v, int | float) or v < 0:
                return mine
            return min(mine, cast(v))

        return Budget(
            tighter("max_records", self.max_records, int),
            tighter("max_chars", self.max_chars, int),
            tighter("max_cost_usd", self.max_cost_usd, float),
        )


def marker(budget: Budget) -> dict:
    return {MARKER: {"recall": True, "retain": True, "budget": budget.declared()}}


def _text(value: object, name: str, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContractError(f"`{name}` must be a string")
    return value[:limit]


def _scope(args: dict) -> str:
    scope = args.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ContractError("`scope` must be a non-empty string")
    return scope


@dataclass(frozen=True)
class RecallCall:
    scope: str
    prompt: str
    context: tuple[str, ...]
    budget: Budget


def read_recall(args: dict, declared: Budget) -> RecallCall:
    context = args.get("context") or []
    if not isinstance(context, list):
        raise ContractError("`context` must be a list of strings")
    return RecallCall(
        scope=_scope(args),
        prompt=_text(args.get("prompt"), "prompt", MAX_PROMPT_CHARS),
        context=tuple(_text(c, "context[]", MAX_CONTEXT_CHARS) for c in context[:MAX_CONTEXT_ITEMS]),
        budget=declared.within(args.get("budget")),
    )


@dataclass(frozen=True)
class RetainCall:
    scope: str
    kind: str
    text: str
    attrs: dict


def read_retain(args: dict) -> RetainCall:
    o = args.get("observation")
    if not isinstance(o, dict):
        raise ContractError("`observation` must be an object")
    attrs = o.get("attrs") or {}
    if not isinstance(attrs, dict):
        raise ContractError("`observation.attrs` must be an object")
    attrs = {str(k): v if isinstance(v, str) else str(v) for k, v in attrs.items()}
    kind = o.get("kind")
    if kind == "turn":
        user = _text(o.get("user"), "user", MAX_TURN_SIDE_CHARS).strip()
        assistant = _text(o.get("assistant"), "assistant", MAX_TURN_SIDE_CHARS).strip()
        text = f"User: {user}\nAssistant: {assistant}" if user or assistant else ""
    elif kind == "note":
        text = _text(o.get("text"), "text", 2 * MAX_TURN_SIDE_CHARS).strip()
    else:
        raise ContractError("`observation.kind` must be `turn` or `note`")
    return RetainCall(scope=_scope(args), kind=kind, text=text, attrs=attrs)


def fit(records: list[dict], budget: Budget) -> tuple[list[dict], int]:
    """Keep whole records, in order, within the budget; count those left out."""
    kept, chars, left_out = [], 0, 0
    for r in records:
        if len(kept) < budget.max_records and chars + len(r["text"]) <= budget.max_chars:
            kept.append(r)
            chars += len(r["text"])
        else:
            left_out += 1
    return kept, left_out
