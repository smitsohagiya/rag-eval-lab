"""Application settings, read from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Each value can be overridden with an APP_* environment variable."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO", description="Python logging level name.")
    environment: str = Field(default="local", description="local, ci or production.")
    data_dir: Path = Field(default=Path("data"), description="Corpora and generated files.")

    @property
    def corpus_dir(self) -> Path:
        """Where the FastAPI docs pages are stored."""
        return self.data_dir / "corpus" / "fastapi"

    @property
    def chunks_dir(self) -> Path:
        """Where chunk files are written, one JSONL file per configuration."""
        return self.data_dir / "chunks"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once per process."""
    return Settings()
