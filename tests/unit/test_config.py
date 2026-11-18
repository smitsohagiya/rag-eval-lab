import pytest

from rag_eval_lab.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.environment == "local"


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")
    assert Settings(_env_file=None).log_level == "DEBUG"
