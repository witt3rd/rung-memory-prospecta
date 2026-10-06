"""rung's own conformance check, `rung-agent --memory-check`, against this provider over HTTP.

Needs rung-agent on PATH (CI builds it from rung v0.1.13). Every one of the 7 clauses must print `pass`.
"""

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(shutil.which("rung-agent") is None, reason="rung-agent not on PATH")


def test_conformance_over_http(http_server):
    out = subprocess.run(
        ["rung-agent", "--memory-check", "mcp:" + http_server], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stdout + out.stderr
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("pass", "FAIL"))]
    assert len(lines) == 7 and all(ln.startswith("pass") for ln in lines), out.stdout
