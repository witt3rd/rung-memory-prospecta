"""Guard the model's output before Prospecta reads it.

Prospecta turns the write-side reply into anticipated questions, one per line, and would index
whatever comes back: `None` becomes the question "None", a refusal becomes its sentences. This module
checks the reply first and raises `BadModelOutput` when no usable question is in it, so a retain
fails visibly instead of storing a record that can never be found.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_REFUSAL = re.compile(
    r"^\W*(i\s*['\u2019]?m sorry|i am sorry|sorry,|i can['\u2019]?t|i cannot|i can not|i['\u2019]m unable|"
    r"i am unable|i['\u2019]m not able|as an ai|unable to (?:help|assist|comply))",
    re.I,
)


class BadModelOutput(ValueError):
    """The model's reply cannot be used (empty, None, a refusal, or no questions in it)."""


def questions_text(raw: object) -> str:
    """The reply as one question per line, or BadModelOutput."""
    if raw is None:
        raise BadModelOutput("the model returned no content")
    if not isinstance(raw, str):
        raise BadModelOutput(f"the model returned {type(raw).__name__}, not text")
    text = _FENCE.sub("", raw.strip()).strip()
    if not text:
        raise BadModelOutput("the model returned an empty reply")
    if text[0] in "[{":
        text = "\n".join(_json_questions(text))
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if _REFUSAL.match(first):
        raise BadModelOutput("the model refused")
    if not any(ln.rstrip().endswith("?") for ln in text.splitlines()):
        raise BadModelOutput("the reply holds no question")
    return text


def _json_questions(text: str) -> list[str]:
    try:
        v = json.loads(text)
    except ValueError as e:
        raise BadModelOutput("the reply is malformed JSON") from e
    if isinstance(v, dict):
        v = next((x for x in v.values() if isinstance(x, list)), None)
    if not isinstance(v, list):
        raise BadModelOutput("the JSON reply holds no list of questions")
    out = [x.get("text") or x.get("question") if isinstance(x, dict) else x for x in v]
    out = [x.strip() for x in out if isinstance(x, str) and x.strip()]
    if not out:
        raise BadModelOutput("the JSON reply holds no questions")
    return out


def json_text(raw: object) -> str:
    """A JSON-mode reply as text. Prospecta falls back to the plain prompt on unparsable JSON, so only
    a non-string (None) needs turning into one that fails its parse."""
    return raw if isinstance(raw, str) else ""


def checked(llm):
    """Wrap an LLM callable: write-side replies are validated, read-side (json_mode) ones made safe."""

    def wrapped(messages, *, json_mode: bool = False):
        raw = llm(messages, json_mode=json_mode)
        if json_mode:
            return json_text(raw)
        return questions_text(raw)

    for attr in ("estimate_usd",):
        if hasattr(llm, attr):
            setattr(wrapped, attr, getattr(llm, attr))
    return wrapped


def caused_by_bad_output(e: BaseException | None) -> bool:
    while e is not None:
        if isinstance(e, BadModelOutput):
            return True
        e = e.__cause__ or e.__context__
    return False
