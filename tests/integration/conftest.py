"""Integration tests need Postgres with pgvector: `docker compose up -d` locally, a service in CI.

Each test gets its own schema, so the tests never touch indexed data in the public schema.
"""

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from rag_eval_lab import store

BASE_URL = os.environ.get("APP_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")


@pytest.fixture
def database_url() -> Iterator[str]:
    """A URL whose search_path points at a fresh schema, dropped after the test."""
    try:
        admin = psycopg.connect(BASE_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres with pgvector is not running (docker compose up -d)")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    with admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield f"{BASE_URL}?options=-csearch_path%3D{schema}%2Cpublic"
    finally:
        with psycopg.connect(BASE_URL, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def conn(database_url: str) -> Iterator[store.Connection]:
    with store.connect(database_url) as connection:
        yield connection
