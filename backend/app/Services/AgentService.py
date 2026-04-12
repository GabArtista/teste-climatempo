"""
LLM Agent Service — Hybrid intent detection + tool calling.

Architecture decision: qwen2.5:1.5b (1.5B parameters, CPU-only) has low
recall for tool calling in Portuguese. To guarantee reliable weather data
retrieval, we use a hybrid approach:

  1. Detect weather intent via keyword matching (deterministic)
  2. Extract city name via fuzzy matching against capitals database
  3. If weather intent + known city → call Open-Meteo directly, use LLM only
     to format the natural-language response
  4. If weather intent + unknown city → inform user of supported cities
  5. If weather intent + no city mentioned → ask the LLM to ask for city
  6. If no weather intent → normal LLM conversation without tools

This eliminates hallucinated weather data (the model making up forecasts
instead of calling the API) while preserving natural conversation for
non-weather topics.
"""
import json
import logging
import re

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from app.Models.ChatMessage import ChatMessage, ChatResponse, MessageRole
from app.Models.WeatherForecast import WeatherResponse
from app.Repositories.CapitalsRepository import CapitalsRepository
from app.Services.WeatherService import CityNotFoundError, WeatherService
from app.Tools.WeatherTool import WEATHER_TOOL
from config.settings import Settings

logger = logging.getLogger(__name__)

# Keywords that indicate a weather-related question (Portuguese + English)
_WEATHER_KEYWORDS = {
    'tempo', 'clima', 'temperatura', 'chuva', 'previsão', 'previsao',
    'chover', 'choverá', 'chovera', 'calor', 'frio', 'precipitação',
    'precipitacao', 'graus', 'umidade', 'vento', 'nublado', 'sol',
    'ensolarado', 'nuvem', 'nuvens', 'quente', 'fria', 'quentes',
    'forecast', 'weather', 'rain', 'temperature',
}

_WEATHER_PHRASES = [
    'como está o tempo', 'como esta o tempo',
    'como vai estar', 'vai chover', 'está chovendo', 'esta chovendo',
    'como está o clima', 'como esta o clima',
    'previsão do tempo', 'previsao do tempo',
]

_SYSTEM_PROMPT = (
    "Você é um assistente de previsão do tempo para capitais brasileiras. "
    "Responda sempre em português, de forma clara e amigável.\n\n"
    "IMPORTANTE:\n"
    "- Quando receber dados de previsão do tempo, formate-os de forma legível "
    "com datas, temperaturas em °C e precipitação em mm.\n"
    "- Quando o usuário perguntar sobre uma cidade que não é capital estadual, "
    "explique gentilmente que só temos dados das 26 capitais brasileiras.\n"
    "- Quando precisar pedir a cidade, seja direto e amigável."
)

_FORMAT_PROMPT = (
    "O usuário perguntou: {question}\n\n"
    "Aqui estão os dados reais de previsão do tempo obtidos da API:\n"
    "{weather_data}\n\n"
    "Por favor, formate esses dados de forma clara e amigável em português, "
    "incluindo as datas, temperaturas máxima e mínima em °C, e precipitação em mm. "
    "Use emojis para deixar mais visual. NÃO invente dados — use apenas os fornecidos acima."
)


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not running or not reachable."""


class AgentService:
    """
    Hybrid weather agent: deterministic intent detection + LLM for formatting.
    """

    def __init__(
        self,
        settings: Settings,
        weather_service: WeatherService,
        repo: CapitalsRepository,
    ) -> None:
        self._settings = settings
        self._weather_service = weather_service
        self._repo = repo
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(self, message: str, history: list[ChatMessage]) -> ChatResponse:
        """
        Process a user message through the hybrid agent pipeline.

        Args:
            message: Current user message.
            history: Previous conversation turns.

        Returns:
            ChatResponse with reply text and tool metadata.
        """
        logger.info("Chat message: %r", message[:60])

        try:
            if self._is_weather_query(message):
                return await self._handle_weather_query(message, history)
            else:
                return await self._handle_general_chat(message, history)

        except APIConnectionError as exc:
            logger.error("Ollama connection failed: %s", exc)
            raise OllamaUnavailableError(
                "Ollama não está disponível. Certifique-se que está rodando: ollama serve"
            ) from exc
        except APITimeoutError as exc:
            logger.error("Ollama timeout: %s", exc)
            raise OllamaUnavailableError("Ollama demorou demais para responder.") from exc

    async def check_health(self) -> dict:
        """Return health status of the Ollama backend."""
        try:
            models = await self._client.models.list()
            available = any(self._settings.ollama_model in m.id for m in models.data)
            return {"status": "ok" if available else "model_not_found",
                    "model": self._settings.ollama_model, "available": available}
        except Exception as exc:
            return {"status": "unavailable", "model": self._settings.ollama_model,
                    "available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _is_weather_query(self, message: str) -> bool:
        """Return True if the message is weather-related."""
        lower = message.lower()
        words = set(re.sub(r'[?!.,;]', ' ', lower).split())
        if words & _WEATHER_KEYWORDS:
            return True
        return any(phrase in lower for phrase in _WEATHER_PHRASES)

    def _extract_city(self, message: str) -> dict | None:
        """
        Try to find a known capital city mentioned in the message.
        Tries n-grams (3, 2, 1 words) against the capitals database.
        """
        text = re.sub(r'[?!.,;]', ' ', message)
        words = text.split()

        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i: i + n])
                result = self._repo.find_city(phrase)
                if result:
                    return result
        return None

    def _extract_days(self, message: str) -> int:
        """Extract forecast days from message, default 3."""
        match = re.search(r'(\d+)\s*dia', message.lower())
        if match:
            return max(1, min(7, int(match.group(1))))
        if 'semana' in message.lower():
            return 7
        if 'amanhã' in message.lower() or 'amanha' in message.lower():
            return 1
        return 3

    # ------------------------------------------------------------------
    # Weather flow
    # ------------------------------------------------------------------

    async def _handle_weather_query(
        self,
        message: str,
        history: list[ChatMessage],
    ) -> ChatResponse:
        """Handle a detected weather query."""
        city = self._extract_city(message)

        if city is None:
            # Check history for a city mentioned previously
            for msg in reversed(history):
                city = self._extract_city(msg.content)
                if city:
                    break

        if city is None:
            # Weather intent but no city found — ask for it
            logger.info("Weather intent detected, no city found — asking user")
            response = await self._llm_ask_for_city(message, history)
            return ChatResponse(response=response, tool_called=False, city_queried=None, reason="no_city")

        days = self._extract_days(message)
        logger.info("Weather query: city=%s days=%d", city['name'], days)

        try:
            forecast = await self._weather_service.get_forecast(
                city=city['name'], forecast_days=days
            )
        except CityNotFoundError as exc:
            return ChatResponse(
                response=str(exc),
                tool_called=False,
                city_queried=city['name'],
                reason="non_capital",
            )

        # Use LLM only to format the response — data is real, from the API
        formatted = await self._llm_format_weather(message, forecast)
        return ChatResponse(
            response=formatted,
            tool_called=True,
            city_queried=city['name'],
            reason="success",
        )

    async def _llm_format_weather(
        self, question: str, forecast: WeatherResponse
    ) -> str:
        """Ask the LLM to format real weather data into natural language."""
        weather_text = forecast.to_text()

        prompt = _FORMAT_PROMPT.format(
            question=question,
            weather_data=weather_text,
        )

        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        result = resp.choices[0].message.content or weather_text
        logger.info("LLM formatted weather response for %s", forecast.city)
        return result

    async def _llm_ask_for_city(
        self, message: str, history: list[ChatMessage]
    ) -> str:
        """Ask the LLM to request the city from the user."""
        cities_sample = ", ".join(self._repo.list_cities()[:6])
        messages = [
            {"role": "system", "content": (
                f"{_SYSTEM_PROMPT}\n\n"
                f"Cidades disponíveis (26 capitais estaduais): {cities_sample} e outras.\n"
                "Peça ao usuário que informe uma capital estadual brasileira."
            )},
        ]
        for msg in history[-4:]:
            messages.append({"role": msg.role.value, "content": msg.content})
        messages.append({"role": "user", "content": message})

        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=messages,
        )
        return resp.choices[0].message.content or "Por favor, informe o nome de uma capital estadual brasileira."

    # ------------------------------------------------------------------
    # General chat (non-weather)
    # ------------------------------------------------------------------

    async def _handle_general_chat(
        self, message: str, history: list[ChatMessage]
    ) -> ChatResponse:
        """Handle non-weather messages with plain LLM conversation."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for msg in history[-6:]:
            messages.append({"role": msg.role.value, "content": msg.content})
        messages.append({"role": "user", "content": message})

        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=messages,
        )
        content = resp.choices[0].message.content or "Como posso ajudar?"
        return ChatResponse(response=content, tool_called=False, city_queried=None, reason="non_weather")
