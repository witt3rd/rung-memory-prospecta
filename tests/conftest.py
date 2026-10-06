"""Fixtures. The DB tests need a pgvector Postgres: set PROSPECTA_TEST_DATABASE_URL to an admin URL
(see docker/compose.yml); without it they skip. Each session gets its own scratch database."""

import os
import socket
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse, urlunparse

import pytest

ADMIN = os.environ.get("PROSPECTA_TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def db_url():
    if not ADMIN:
        pytest.skip("PROSPECTA_TEST_DATABASE_URL not set")
    import psycopg

    name = "rmp_" + uuid.uuid4().hex[:12]
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(ADMIN)._replace(path="/" + name))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    yield url
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
        c.execute(f'DROP DATABASE "{name}"')


@pytest.fixture
def provider(db_url):
    from rung_memory_prospecta.provider import Provider, Settings

    p = Provider(Settings(database_url=db_url, model="fake", embedder="fake"))
    yield p
    p.close()


@pytest.fixture(scope="session")
def http_server(db_url):
    """The provider over HTTP on a free loopback port, fake model and embedder."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, PROSPECTA_DATABASE_URL=db_url, RMP_MODEL="fake", RMP_EMBEDDER="fake")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rung_memory_prospecta", "serve", "--port", str(port)], env=env
    )
    for _ in range(100):
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
