#!/usr/bin/env python3
"""Repository guard: no secrets, no consumer names, no datasets in the tree.

    scripts/guard.py            # scan every tracked file (what CI runs)
    scripts/guard.py PATH...    # scan these files instead

Three rules, each a hard failure:

1. **Secrets.** No API key, token or private key is committed. The patterns
   below match the credential shapes this repository could plausibly meet
   (OpenRouter, OpenAI-style, GitHub, Doppler, AWS, Hugging Face, Slack,
   PEM private keys) plus `NAME=value` assignments of key-like variables.
2. **Consumer names.** This repository is a provider; it names rung and
   Jev-Mem and no product that consumes either. The banned words are kept
   as SHA-256 digests of lowercase word tokens so that the guard itself
   does not spell them.
3. **Datasets.** Benchmark data (LoCoMo is CC BY-NC 4.0 and may not be
   redistributed) lives in the gitignored `eval/cache/` and is never tracked.

A hit prints `path:line: rule` and never the matched text, so the guard's
own output cannot leak what it found.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

SECRET_PATTERNS = {
    "openrouter key": re.compile(r"sk-or-(?:v1-)?[0-9a-f]{32,}"),
    "openai-style key": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}"),
    "github token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "doppler token": re.compile(r"\bdp\.(?:st|ct|sa|scim|audit|pt)\.[A-Za-z0-9_.-]{20,}"),
    "aws access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "hugging face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    # An env-style NAME=value (upper-case name), or a quoted literal assigned to
    # a key-like name in code. Placeholders ($VAR, <...>, {...}, test/fixture
    # values) are not credentials.
    "key assignment": re.compile(
        r"""\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)\s*[:=]\s*["']?"""
        r"""(?!\$|<|\{|(?i:your|changeme|none|null|test|fixture|example|redacted|offline)|x+\b)[A-Za-z0-9_./+-]{16,}"""
    ),
    "quoted key literal": re.compile(
        r"""(?i)\b\w*(?:api_key|secret|token|password)\w*\s*[:=]\s*["']"""
        r"""(?!your|changeme|test|fixture|example|redacted|offline)[^"'\s$<{]{16,}["']"""
    ),
}

# sha256 of each banned lowercase word token (see rule 2).
BANNED_WORD_DIGESTS = frozenset(
    {
        "8aa27f40201f446bbbc28266c61fc3ada12522422ef34f51ce733bd65960cf54",
        "850ca8e0ec3f2106b6a230cd36a86cf163b3b005dd2b6c94eafaf604cbdf03a8",
    }
)
WORD = re.compile(r"[a-z0-9]+")

DATASET_PATH = re.compile(r"(?i)(^|/)(eval/cache/|locomo\d*\.json|longmemeval_[a-z0-9_]*\.json)")
DATASET_EXEMPT = ("eval/datasets.md",)


def tracked(root: Path) -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)  # noqa: S607
    return [f for f in out.decode().split("\0") if f]


def scan_text(text: str) -> list[tuple[int, str]]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for rule, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                hits.append((n, "secret: " + rule))
        for word in WORD.findall(line.lower()):
            if hashlib.sha256(word.encode()).hexdigest() in BANNED_WORD_DIGESTS:
                hits.append((n, "consumer name"))
                break
    return hits


def scan(root: Path, files: list[str]) -> list[str]:
    hits = []
    for f in files:
        if DATASET_PATH.search(f) and f not in DATASET_EXEMPT:
            hits.append(f"{f}:0: dataset file must not be tracked")
            continue
        try:
            text = (root / f).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend(f"{f}:{n}: {rule}" for n, rule in scan_text(text))
    return hits


def main(argv: list[str]) -> int:
    root = Path.cwd()
    files = argv or tracked(root)
    hits = scan(root, files)
    for h in hits:
        print(h)
    if hits:
        print(f"guard: {len(hits)} hit(s)", file=sys.stderr)
        return 1
    print(f"guard: ok ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
