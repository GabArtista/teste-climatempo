"""Application settings loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable values for the Weather LLM Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama / LLM
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_api_key: str = "ollama"  # Required by OpenAI client, not used by Ollama

    # Open-Meteo
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"

    # HTTP
    request_timeout: int = 30

    # App
    app_name: str = "Weather LLM Agent"
    app_version: str = "1.0.0"
    app_description: str = (
        "LLM agent with function calling for weather forecasts via Open-Meteo API. "
        "Built for the Climatempo technical challenge."
    )
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
