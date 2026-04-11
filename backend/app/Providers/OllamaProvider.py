"""Ollama client provider — singleton factory."""
import logging

from openai import AsyncOpenAI

from config.settings import Settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_ollama_client(settings: Settings) -> AsyncOpenAI:
    """
    Return the singleton OpenAI-compatible client pointed at Ollama.

    Args:
        settings: Application settings with Ollama URL and API key.

    Returns:
        Configured AsyncOpenAI client.
    """
    global _client
    if _client is None:
        logger.info("Creating Ollama client: base_url=%s", settings.ollama_base_url)
        _client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        )
    return _client
