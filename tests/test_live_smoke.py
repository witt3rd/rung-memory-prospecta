"""Opt-in live smoke: `rung-agent --memory-check` with the real model and embedder.

Skipped unless RMP_LIVE_SMOKE=1 (the `live-smoke` workflow sets it, with secrets). Costs a few cents.
Needs PROSPECTA_TEST_DATABASE_URL, rung-agent on PATH, RMP_MODEL / RMP_EMBEDDER and the provider's keys.
"""

import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RMP_LIVE_SMOKE") != "1" or shutil.which("rung-agent") is None,
    reason="opt-in: set RMP_LIVE_SMOKE=1 and have rung-agent on PATH",
)


@pytest.fixture
def live_server(db_url):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, PROSPECTA_DATABASE_URL=db_url)  # RMP_MODEL / RMP_EMBEDDER as given
    proc = subprocess.Popen(
        [sys.executable, "-m", "rung_memory_prospecta", "serve", "--port", str(port)], env=env
    )
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("provider did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=10)


def test_memory_check_with_the_real_model(live_server):
    out = subprocess.run(
        ["rung-agent", "--memory-check", "mcp:" + live_server], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stdout + out.stderr
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("pass", "FAIL"))]
    assert len(lines) == 7 and all(ln.startswith("pass") for ln in lines), out.stdout
