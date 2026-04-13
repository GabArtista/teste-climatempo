"""
Unit tests for the 5-stage cascading intent classification pipeline.

Tests AgentService._classify_intent() directly — no HTTP, no Ollama required.
Each test documents which stage is expected to handle the decision.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.Services.AgentService import AgentService
from config.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> AgentService:
    """AgentService with mocked dependencies — no Ollama, no HTTP calls."""
    settings = MagicMock(spec=Settings)
    settings.ollama_base_url = "http://localhost:11434/v1"
    settings.ollama_api_key = "ollama"
    settings.ollama_model = "qwen2.5:1.5b"

    weather_service = MagicMock()
    repo = MagicMock()

    # _extract_city is used inside _score_weather_intent to detect capitals.
    # Default: return None (no city found). Override per test when needed.
    repo.find_city.return_value = None
    repo.list_cities.return_value = ["Sao Paulo - Sao Paulo", "Rio de Janeiro - Rio de Janeiro"]

    with patch("app.Services.AgentService.AsyncOpenAI"):
        service = AgentService(settings, weather_service, repo)

    return service


def city_result(name: str = "Manaus - Amazonas") -> dict:
    """Helper to simulate a city found in the repository."""
    return {"name": name, "latitude": -3.119, "longitude": -60.022}


# ---------------------------------------------------------------------------
# Stage 1 — Strong keywords: should resolve as "weather" immediately
# ---------------------------------------------------------------------------

class TestStage1StrongKeywords:
    """Messages with unambiguous forecast vocabulary — never reach Stage 2+."""

    @pytest.mark.asyncio
    async def test_previsao(self, agent):
        assert await agent._classify_intent("Previsão para Brasília essa semana") == "weather"

    @pytest.mark.asyncio
    async def test_temperatura(self, agent):
        assert await agent._classify_intent("Temperatura amanhã em Curitiba?") == "weather"

    @pytest.mark.asyncio
    async def test_precipitacao(self, agent):
        assert await agent._classify_intent("Precipitação acumulada em Belém") == "weather"

    @pytest.mark.asyncio
    async def test_forecast_english(self, agent):
        assert await agent._classify_intent("forecast for tomorrow in Recife") == "weather"

    @pytest.mark.asyncio
    async def test_weather_english(self, agent):
        assert await agent._classify_intent("weather in São Paulo today") == "weather"


# ---------------------------------------------------------------------------
# Stage 1 (negative path) — No weather vocabulary at all
# ---------------------------------------------------------------------------

class TestStage1NoVocab:
    """Messages with zero weather vocabulary — resolved as not_weather at Stage 1."""

    @pytest.mark.asyncio
    async def test_greeting(self, agent):
        assert await agent._classify_intent("Olá, tudo bem?") == "not_weather"

    @pytest.mark.asyncio
    async def test_capital_question(self, agent):
        assert await agent._classify_intent("Qual a capital do Brasil?") == "not_weather"

    @pytest.mark.asyncio
    async def test_math(self, agent):
        assert await agent._classify_intent("Quanto é 2 + 2?") == "not_weather"

    @pytest.mark.asyncio
    async def test_joke(self, agent):
        assert await agent._classify_intent("Me conta uma piada") == "not_weather"

    @pytest.mark.asyncio
    async def test_machine_learning(self, agent):
        assert await agent._classify_intent("O que é machine learning?") == "not_weather"


# ---------------------------------------------------------------------------
# Stage 2 — Fixed phrases: unambiguous weather expressions
# ---------------------------------------------------------------------------

class TestStage2FixedPhrases:
    """Messages that match known weather phrases — resolved at Stage 2."""

    @pytest.mark.asyncio
    async def test_vai_chover(self, agent):
        assert await agent._classify_intent("Vai chover em Manaus amanhã?") == "weather"

    @pytest.mark.asyncio
    async def test_como_esta_o_tempo(self, agent):
        assert await agent._classify_intent("Como está o tempo em Fortaleza?") == "weather"

    @pytest.mark.asyncio
    async def test_previsao_do_tempo(self, agent):
        assert await agent._classify_intent("previsão do tempo para essa semana") == "weather"

    @pytest.mark.asyncio
    async def test_esta_chovendo(self, agent):
        assert await agent._classify_intent("Está chovendo em Porto Alegre?") == "weather"

    @pytest.mark.asyncio
    async def test_como_esta_o_clima(self, agent):
        assert await agent._classify_intent("Como está o clima em Florianópolis?") == "weather"


# ---------------------------------------------------------------------------
# Stage 3 — Exclusion patterns: personal comments with weather words
# ---------------------------------------------------------------------------

class TestStage3ExclusionPatterns:
    """Messages that contain weather words in personal/non-forecast contexts."""

    @pytest.mark.asyncio
    async def test_to_com_frio(self, agent):
        assert await agent._classify_intent("Tô com frio aqui no escritório") == "not_weather"

    @pytest.mark.asyncio
    async def test_estou_com_calor(self, agent):
        assert await agent._classify_intent("Estou com calor, preciso de água") == "not_weather"

    @pytest.mark.asyncio
    async def test_fiquei_com_frio(self, agent):
        assert await agent._classify_intent("Fiquei com frio ontem à noite") == "not_weather"

    @pytest.mark.asyncio
    async def test_ar_condicionado(self, agent):
        assert await agent._classify_intent("O ar-condicionado está com frio demais") == "not_weather"

    @pytest.mark.asyncio
    async def test_ela_estava_quente(self, agent):
        assert await agent._classify_intent("Ela estava muito quente naquela festa") == "not_weather"

    @pytest.mark.asyncio
    async def test_em_casa(self, agent):
        assert await agent._classify_intent("Que vento em casa hoje") == "not_weather"

    @pytest.mark.asyncio
    async def test_no_escritorio(self, agent):
        assert await agent._classify_intent("Que frio no escritório hoje") == "not_weather"

    @pytest.mark.asyncio
    async def test_febre_e_frio(self, agent):
        assert await agent._classify_intent("Estou com febre e frio") == "not_weather"

    @pytest.mark.asyncio
    async def test_ventilador(self, agent):
        assert await agent._classify_intent("O ventilador está fazendo muito frio") == "not_weather"


# ---------------------------------------------------------------------------
# Stage 4 — Scoring: borderline cases resolved by multi-signal scoring
# ---------------------------------------------------------------------------

class TestStage4Scoring:
    """Messages that pass Stages 1-3 but are resolved by the scoring system."""

    @pytest.mark.asyncio
    async def test_chuva_com_interrogacao_e_cidade(self, agent):
        # city found → +0.30, question mark → +0.40 = 0.80 > threshold
        agent._repo.find_city.return_value = city_result("Recife - Pernambuco")
        assert await agent._classify_intent("Chuva em Recife?") == "weather"

    @pytest.mark.asyncio
    async def test_sol_bonito_foto(self, agent):
        # no question mark, no city, no temporal, no geo prep → score stays low
        assert await agent._classify_intent("Que sol bonito na foto") == "not_weather"

    @pytest.mark.asyncio
    async def test_vento_bateu_janela(self, agent):
        # Stage 3 catches "aqui"; if not, scoring: no ?, no city, personal context
        result = await agent._classify_intent("O vento bateu a janela aqui")
        assert result == "not_weather"

    @pytest.mark.asyncio
    async def test_vai_fazer_frio_com_cidade(self, agent):
        # future weather verb → +0.30, city → +0.30, geo prep → +0.15 = 0.85
        agent._repo.find_city.return_value = city_result("Curitiba - Parana")
        assert await agent._classify_intent("Vai fazer frio em Curitiba essa semana?") == "weather"

    @pytest.mark.asyncio
    async def test_interrogativa_no_inicio(self, agent):
        # starts with "como" → +0.35, question → +0.40, weak keyword → baseline
        assert await agent._classify_intent("Como vai o clima amanhã?") == "weather"


# ---------------------------------------------------------------------------
# Scoring unit tests — verify individual signal weights
# ---------------------------------------------------------------------------

class TestScoringSignals:
    """Direct tests of _score_weather_intent() signal weights."""

    def test_question_mark_adds_score(self, agent):
        score_with = agent._score_weather_intent("Chuva amanhã?")
        score_without = agent._score_weather_intent("Chuva amanhã")
        assert score_with > score_without

    def test_personal_subject_reduces_score(self, agent):
        score_personal = agent._score_weather_intent("Eu sinto frio")
        score_neutral = agent._score_weather_intent("Frio em Manaus?")
        assert score_personal < score_neutral

    def test_city_increases_score(self, agent):
        # Use a message that won't max out the score without a city
        agent._repo.find_city.return_value = None
        score_no_city = agent._score_weather_intent("Chuva hoje")

        agent._repo.find_city.return_value = city_result()
        score_with_city = agent._score_weather_intent("Chuva em Manaus hoje")

        assert score_with_city > score_no_city

    def test_body_sensation_heavily_reduces_score(self, agent):
        score = agent._score_weather_intent("Estou com frio agora")
        assert score < 0.20

    def test_baseline_is_positive(self, agent):
        # Even with no signals at all, baseline > 0
        score = agent._score_weather_intent("chuva")
        assert score > 0.0

    def test_score_clamped_between_0_and_1(self, agent):
        # Extremely positive message
        agent._repo.find_city.return_value = city_result()
        score_high = agent._score_weather_intent("Como vai estar a temperatura amanhã em Manaus?")
        assert 0.0 <= score_high <= 1.0

        # Extremely negative message
        score_low = agent._score_weather_intent("Eu tô com frio em casa no trabalho")
        assert 0.0 <= score_low <= 1.0


# ---------------------------------------------------------------------------
# Stage 5 — LLM fallback (mocked)
# ---------------------------------------------------------------------------

class TestStage5LLMFallback:
    """Verify Stage 5 is called for genuinely ambiguous messages and respects LLM answer."""

    @pytest.mark.asyncio
    async def test_llm_sim_returns_weather(self, agent):
        agent._llm_classify_intent = AsyncMock(return_value="weather")
        # "Que calor hoje hein?" — weak signal, passes stages 1-4 inconclusively
        # Force stage 5 by patching _score_weather_intent to return mid-range
        with patch.object(agent, '_score_weather_intent', return_value=0.35):
            with patch.object(agent, '_is_non_weather_context', return_value=False):
                result = await agent._classify_intent("Que calor hoje hein?")
        assert result == "weather"
        agent._llm_classify_intent.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_nao_returns_not_weather(self, agent):
        agent._llm_classify_intent = AsyncMock(return_value="not_weather")
        with patch.object(agent, '_score_weather_intent', return_value=0.35):
            with patch.object(agent, '_is_non_weather_context', return_value=False):
                result = await agent._classify_intent("Que calor hoje hein?")
        assert result == "not_weather"
        agent._llm_classify_intent.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_not_called_for_strong_keyword(self, agent):
        """LLM must NOT be called when Stage 1 resolves the intent."""
        agent._llm_classify_intent = AsyncMock(return_value="weather")
        result = await agent._classify_intent("Previsão do tempo para São Paulo")
        agent._llm_classify_intent.assert_not_called()
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_llm_not_called_for_exclusion_pattern(self, agent):
        """LLM must NOT be called when Stage 3 resolves the intent."""
        agent._llm_classify_intent = AsyncMock(return_value="not_weather")
        result = await agent._classify_intent("Tô com frio aqui no escritório")
        agent._llm_classify_intent.assert_not_called()
        assert result == "not_weather"
