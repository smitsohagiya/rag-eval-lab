"""Application settings, read from environment variables and an optional .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Each value can be overridden with an APP_* environment variable."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO", description="Python logging level name.")
    environment: str = Field(default="local", description="local, ci or production.")


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once per process."""
    return Settings()
