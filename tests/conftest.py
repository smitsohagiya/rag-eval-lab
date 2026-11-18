"""Shared test fixtures."""

from collections.abc import Iterator

import pytest

from rag_eval_lab.config import get_settings


@pytest.fixture(autouse=True)
def fresh_settings() -> Iterator[None]:
    """Clear cached settings so each test sees its own environment."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
