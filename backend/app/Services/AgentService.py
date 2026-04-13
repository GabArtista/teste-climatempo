"""
LLM Agent Service — Cascading intent detection + tool calling.

Architecture decision: qwen2.5:1.5b (1.5B parameters, CPU-only) has low
recall for tool calling in Portuguese. To guarantee reliable weather data
retrieval, we use a hybrid approach:

  1. Detect weather intent via a 5-stage cascading pipeline (deterministic first,
     LLM only as last resort)
  2. Extract city name via fuzzy matching against capitals database
  3. If weather intent + known city → call Open-Meteo directly, use LLM only
     to format the natural-language response
  4. If weather intent + unknown city → inform user of supported cities
  5. If weather intent + no city mentioned → ask the LLM to ask for city
  6. If no weather intent → normal LLM conversation without tools

Cascading Intent Pipeline (cheapest to most expensive):
  Stage 1 — Strong keywords (~0ms): "previsão", "temperatura", etc.
  Stage 2 — Fixed phrases (~0ms): "vai chover", "como está o tempo", etc.
  Stage 3 — Exclusion patterns (~1ms): personal comments like "tô com frio aqui"
  Stage 4 — Multi-signal scoring (~1-5ms): weighted signals with thresholds
  Stage 5 — LLM binary classifier (~30-60s): last resort for genuine ambiguity

This eliminates hallucinated weather data while preserving natural conversation
for non-weather topics. The LLM is expected to be needed for < 10% of cases
that pass Stage 1 vocabulary filtering.
"""
import json
import logging
import re
from typing import Literal

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from app.Models.ChatMessage import ChatMessage, ChatResponse, MessageRole
from app.Models.WeatherForecast import WeatherResponse
from app.Repositories.CapitalsRepository import CapitalsRepository
from app.Services.WeatherService import CityNotFoundError, WeatherService
from app.Tools.WeatherTool import WEATHER_TOOL
from config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1 — Strong keywords: almost always a weather query
# ---------------------------------------------------------------------------
_STRONG_KEYWORDS = {
    'previsão', 'previsao', 'temperatura', 'precipitação', 'precipitacao',
    'forecast', 'weather', 'temperature',
    'clima',   # "clima" in PT almost exclusively means meteorological climate
}

# ---------------------------------------------------------------------------
# Stage 2 — Fixed phrases that unambiguously express weather intent
# ---------------------------------------------------------------------------
_WEATHER_PHRASES = [
    'como está o tempo', 'como esta o tempo',
    'como vai estar', 'vai chover', 'está chovendo', 'esta chovendo',
    'como está o clima', 'como esta o clima',
    'previsão do tempo', 'previsao do tempo',
    'vai fazer frio', 'vai fazer calor',
    'como vai o tempo', 'como vai o clima',
]

# ---------------------------------------------------------------------------
# Stage 3 — Exclusion patterns: personal/non-weather uses of weather words
# ---------------------------------------------------------------------------
_EXCLUSION_PATTERNS = [
    # "tô/estou com frio/calor/febre" — physical sensation, not weather query
    re.compile(r'\b(t[oô]|estou|fiquei|sinto|me sinto|fico)\b.{0,20}\bcom\b.{0,10}\b(frio|calor|febre|quente|fria)\b', re.I),
    # "ar-condicionado", "ventilador", "aquecedor" — indoor climate devices
    re.compile(r'\b(ar[\s-]condicionado|ventilador|aquecedor|cobertor|blusa|casaco)\b', re.I),
    # "ela/ele estava quente/fria" — third-person physical description
    re.compile(r'\b(ela|ele|você|vc|a\s+\w+|o\s+\w+)\b.{0,20}\b(estava|está|ficou|ficava)\b.{0,15}\b(quente|fria|frio|gelad)\b', re.I),
    # "aqui no escritório/em casa/no trabalho" — personal locatives
    re.compile(r'\b(aqui\s+(no|em|na|dentro)\b|em\s+casa\b|no\s+(escritório|escritorio|trabalho|apartamento|quarto)\b)', re.I),
    # "febre e frio" / "resfriado" — illness context
    re.compile(r'\b(febre|resfriado|gripe|doente).{0,20}\b(frio|calor|quente)\b', re.I),
    re.compile(r'\b(frio|calor|quente).{0,20}\b(febre|resfriado|gripe|doente)\b', re.I),
]

# ---------------------------------------------------------------------------
# Stage 4 — Multi-signal scoring thresholds
# ---------------------------------------------------------------------------
_THRESHOLD_WEATHER = 0.55       # score above → weather
_THRESHOLD_NOT_WEATHER = 0.20   # score below → not weather
_SCORE_BASELINE = 0.10          # slight positive bias (FP better than FN)

# Interrogatives that at the start of a sentence strongly suggest a question.
# Note: "que" is intentionally excluded — it is also used as an exclamation
# ("Que sol bonito!") which is NOT a weather query, making it too ambiguous.
_INTERROGATIVES = {
    'como', 'qual', 'quais', 'quando', 'onde', 'quanto', 'quanta',
    'quantos', 'quantas', 'vai', 'irá', 'será',
}

# Temporal references that suggest forecasting (future orientation)
_TEMPORAL_MARKERS = {
    'amanhã', 'amanha', 'semana', 'hoje', 'sábado', 'sabado',
    'domingo', 'segunda', 'terça', 'terca', 'quarta', 'quinta',
    'sexta', 'próxim', 'proxim', 'fim de semana',
}

# Personal subject markers (indicate personal comment, not weather query)
_PERSONAL_SUBJECTS = {
    'eu', 'tô', 'to', 'estou', 'me', 'meu', 'minha',
    'a gente', 'nós', 'nos',
}

# Geographic prepositions followed by a proper noun (uppercase first letter).
# This avoids false positives like "na foto", "em casa", "no trabalho".
# Brazilian city and state names are always capitalized in text.
_GEO_PREPOSITIONS = re.compile(r'\b(em|no|na|para|pra)\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ]')

# Future tense meteorological verbs
_FUTURE_WEATHER_VERBS = re.compile(
    r'\b(vai|irá|sera|será)\s+(chover|nevar|ventar|fazer|ter)\b', re.I
)

# Weak weather vocabulary — need context to confirm intent
_WEAK_KEYWORDS = {
    'clima', 'chuva', 'chover', 'choverá', 'chovera',
    'calor', 'frio', 'umidade', 'vento', 'nublado', 'sol',
    'ensolarado', 'nuvem', 'nuvens', 'quente', 'fria', 'quentes',
    'rain', 'graus',
}

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
    Hybrid weather agent: cascading intent detection + LLM for formatting.
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
            intent = await self._classify_intent(message)
            if intent == "weather":
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
    # Cascading intent classification
    # ------------------------------------------------------------------

    async def _classify_intent(self, message: str) -> Literal["weather", "not_weather"]:
        """
        5-stage cascading pipeline. Each stage returns "weather", "not_weather",
        or "inconclusive". Only advances to the next (more expensive) stage when
        the current stage cannot decide.

        Expected flow for most messages:
          - ~60% resolve at Stage 1 (off-topic, no weather vocab)
          - ~25% resolve at Stage 1-2 (clear weather queries)
          - ~10% resolve at Stage 3-4 (personal comments / borderline)
          - ~5% need Stage 5 (LLM) for genuine ambiguity
        """
        lower = message.lower()
        words = set(re.sub(r'[?!.,;]', ' ', lower).split())

        # Stage 1 — Strong keywords: near-certain weather intent
        if words & _STRONG_KEYWORDS:
            logger.debug("Intent: weather (stage 1 — strong keyword)")
            return "weather"

        # Check for any weather vocabulary at all — if none, skip to not_weather
        has_weather_vocab = bool(words & _WEAK_KEYWORDS) or any(
            phrase in lower for phrase in _WEATHER_PHRASES
        )
        if not has_weather_vocab:
            logger.debug("Intent: not_weather (stage 1 — no weather vocab)")
            return "not_weather"

        # Stage 2 — Fixed phrases: unambiguous weather expressions
        if any(phrase in lower for phrase in _WEATHER_PHRASES):
            logger.debug("Intent: weather (stage 2 — fixed phrase)")
            return "weather"

        # Stage 3 — Exclusion patterns: personal/non-weather uses
        if self._is_non_weather_context(message):
            logger.debug("Intent: not_weather (stage 3 — exclusion pattern)")
            return "not_weather"

        # Stage 4 — Multi-signal scoring
        score = self._score_weather_intent(message)
        logger.debug("Intent: stage 4 score=%.2f", score)
        if score >= _THRESHOLD_WEATHER:
            return "weather"
        if score <= _THRESHOLD_NOT_WEATHER:
            return "not_weather"

        # Stage 5 — LLM binary classifier (last resort)
        logger.info("Intent: ambiguous (score=%.2f), escalating to LLM", score)
        return await self._llm_classify_intent(message)

    def _is_non_weather_context(self, message: str) -> bool:
        """
        Stage 3: Check if the message matches known non-weather patterns.
        These are personal comments or domestic contexts that happen to use
        weather-related words but are clearly not weather forecast requests.
        """
        for pattern in _EXCLUSION_PATTERNS:
            if pattern.search(message):
                return True
        return False

    def _score_weather_intent(self, message: str) -> float:
        """
        Stage 4: Score the message based on multiple linguistic signals.
        Returns a float in [0.0, 1.0]. Positive signals push toward weather,
        negative signals push toward personal comment.

        Baseline is slightly positive (0.10) — false positive is preferable
        to false negative in this domain (missing a weather query is worse
        than asking for a city name unnecessarily).
        """
        score = _SCORE_BASELINE
        lower = message.lower()
        words_raw = lower.split()
        words = set(re.sub(r'[?!.,;]', ' ', lower).split())

        # --- Positive signals ---

        # Question mark: strong indicator of a request
        if '?' in message:
            score += 0.40

        # Starts with interrogative word
        if words_raw and words_raw[0].strip('?!.,') in _INTERROGATIVES:
            score += 0.35

        # Geographic preposition followed by a proper noun (uppercase).
        # Search original message — lowercased text loses the case signal.
        if _GEO_PREPOSITIONS.search(message):
            score += 0.15

        # Temporal markers (future orientation)
        if any(marker in lower for marker in _TEMPORAL_MARKERS):
            score += 0.20

        # Future tense meteorological verb
        if _FUTURE_WEATHER_VERBS.search(lower):
            score += 0.30

        # Known capital city in message
        if self._extract_city(message) is not None:
            score += 0.30

        # --- Negative signals ---

        # Personal subject at start of message
        first_word = words_raw[0].strip('?!.,') if words_raw else ''
        if first_word in _PERSONAL_SUBJECTS:
            score -= 0.45

        # Body sensation pattern (estou/tô com frio/calor)
        if re.search(r'\b(estou|t[oô]|fiquei)\b.{0,15}\bcom\b.{0,10}\b(frio|calor|febre)\b', lower):
            score -= 0.50

        # Personal locative (aqui, em casa, no trabalho)
        if re.search(r'\b(aqui|em\s+casa|no\s+(trabalho|escritório|escritorio|quarto|apartamento))\b', lower):
            score -= 0.30

        # Third-person description of a person's temperature
        if re.search(r'\b(ela|ele)\b.{0,20}\b(estava|está|ficou)\b.{0,15}\b(quente|fria|frio)\b', lower):
            score -= 0.35

        return max(0.0, min(1.0, score))

    async def _llm_classify_intent(self, message: str) -> Literal["weather", "not_weather"]:
        """
        Stage 5 (last resort): Ask the LLM to make a binary decision.
        Only called when Stages 1-4 cannot reach a confident conclusion.
        Uses a minimal prompt to get a yes/no answer quickly.
        """
        resp = await self._client.chat.completions.create(
            model=self._settings.ollama_model,
            messages=[{
                "role": "user",
                "content": (
                    "Responda apenas 'sim' ou 'não', sem mais nada.\n"
                    f"A mensagem a seguir é uma consulta de previsão do tempo? "
                    f'"{message}"'
                ),
            }],
        )
        answer = (resp.choices[0].message.content or "").lower().strip()
        result: Literal["weather", "not_weather"] = "weather" if "sim" in answer else "not_weather"
        logger.info("LLM intent classification: %r → %s", answer[:20], result)
        return result

    # ------------------------------------------------------------------
    # City and days extraction
    # ------------------------------------------------------------------

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
