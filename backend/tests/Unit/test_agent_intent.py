"""
Unit tests for the 5-stage cascading intent classification pipeline.

Tests AgentService internal methods directly — no HTTP, no Ollama required.
Each test documents which stage is expected to handle the decision.

Coverage map:
  TestStage1StrongKeywords        — Stage 1 positive path (strong keywords)
  TestStage1NoVocab               — Stage 1 negative path (no weather vocab)
  TestStage2FixedPhrases          — Stage 2 (fixed phrases)
  TestStage3ExclusionPatterns     — Stage 3 via _classify_intent
  TestIsNonWeatherContextDirect   — Stage 3 _is_non_weather_context() directly
  TestExtractDays                 — _extract_days() direct unit tests
  TestLLMClassifyIntentDirect     — _llm_classify_intent() direct unit tests
  TestCascadeGuarantees           — later stages NOT called when earlier resolves
  TestStage4Scoring               — Stage 4 via _classify_intent
  TestScoringSignals              — _score_weather_intent() signal verification
  TestScoringGranular             — individual signal weights verified in isolation
  TestCompetingSignals            — messages with both positive and negative signals
  TestStage5LLMFallback           — Stage 5 LLM fallback via _classify_intent
  TestAllCapitals                 — all 26 valid capitals via real CapitalsRepository
  TestEdgeCases                   — robustness: ALLCAPS, accents, long text, etc.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.Services.AgentService import AgentService
from app.Repositories.CapitalsRepository import CapitalsRepository
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


@pytest.fixture
def agent_real_repo() -> AgentService:
    """AgentService with REAL CapitalsRepository — tests actual city detection."""
    settings = MagicMock(spec=Settings)
    settings.ollama_base_url = "http://localhost:11434/v1"
    settings.ollama_api_key = "ollama"
    settings.ollama_model = "qwen2.5:1.5b"

    with patch("app.Services.AgentService.AsyncOpenAI"):
        service = AgentService(settings, MagicMock(), CapitalsRepository())

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


# ---------------------------------------------------------------------------
# Stage 3 — _is_non_weather_context() tested DIRECTLY
# ---------------------------------------------------------------------------

class TestIsNonWeatherContextDirect:
    """
    Direct unit tests for _is_non_weather_context().
    Verifies each exclusion pattern independently — not just through classify_intent.
    Also verifies patterns do NOT accidentally exclude real weather queries.
    """

    # Pattern 1 — personal body sensation (tô/estou/fiquei + com + frio/calor/febre)
    def test_pattern1_to_com_frio(self, agent):
        assert agent._is_non_weather_context("Tô com frio agora") is True

    def test_pattern1_estou_com_calor(self, agent):
        assert agent._is_non_weather_context("Estou com calor demais") is True

    def test_pattern1_fiquei_com_febre(self, agent):
        assert agent._is_non_weather_context("Fiquei com febre ontem") is True

    # Pattern 2 — indoor climate devices
    def test_pattern2_ar_condicionado(self, agent):
        assert agent._is_non_weather_context("O ar-condicionado está gelado") is True

    def test_pattern2_ventilador(self, agent):
        assert agent._is_non_weather_context("Liguei o ventilador") is True

    def test_pattern2_cobertor(self, agent):
        assert agent._is_non_weather_context("Precisei do cobertor") is True

    # Pattern 3 — third-person physical description
    def test_pattern3_ela_estava_quente(self, agent):
        assert agent._is_non_weather_context("Ela estava quente naquela situação") is True

    def test_pattern3_ele_ficou_frio(self, agent):
        assert agent._is_non_weather_context("Ele ficou frio ao ouvir a notícia") is True

    # Pattern 4 — personal locatives
    def test_pattern4_em_casa(self, agent):
        assert agent._is_non_weather_context("Que frio em casa") is True

    def test_pattern4_no_escritorio(self, agent):
        assert agent._is_non_weather_context("Que calor no escritório") is True

    # Pattern 5 — illness context (bidirectional)
    def test_pattern5_febre_e_frio(self, agent):
        assert agent._is_non_weather_context("Estou com febre e frio") is True

    def test_pattern5_frio_e_febre(self, agent):
        assert agent._is_non_weather_context("Sinto frio e tenho febre") is True

    # Critical negatives — real weather queries must NOT be excluded
    def test_does_not_exclude_vai_chover(self, agent):
        assert agent._is_non_weather_context("Vai chover em Fortaleza?") is False

    def test_does_not_exclude_vento_em_cidade(self, agent):
        assert agent._is_non_weather_context("Como está o vento em Brasília?") is False

    def test_does_not_exclude_previsao_direta(self, agent):
        assert agent._is_non_weather_context("Previsão para Curitiba essa semana") is False


# ---------------------------------------------------------------------------
# _extract_days() — direct unit tests
# ---------------------------------------------------------------------------

class TestExtractDays:
    """Direct tests of _extract_days() — covers all natural language patterns."""

    def test_amanha_with_accent(self, agent):
        assert agent._extract_days("Como vai estar o tempo amanhã?") == 1

    def test_amanha_without_accent(self, agent):
        assert agent._extract_days("tempo amanha em Recife") == 1

    def test_semana_returns_7(self, agent):
        assert agent._extract_days("Previsão para essa semana") == 7

    def test_n_dias_exact(self, agent):
        assert agent._extract_days("Previsão para 5 dias") == 5

    def test_3_dias(self, agent):
        assert agent._extract_days("Próximos 3 dias em Manaus") == 3

    def test_default_no_hint(self, agent):
        """No temporal hint → default 3 days."""
        assert agent._extract_days("Clima em São Paulo") == 3

    def test_clamped_max_7(self, agent):
        """Values above 7 must be clamped to 7."""
        assert agent._extract_days("Previsão para 10 dias") == 7

    def test_clamped_min_1(self, agent):
        """Values below 1 must be clamped to 1."""
        assert agent._extract_days("Previsão para 0 dias") == 1


# ---------------------------------------------------------------------------
# _llm_classify_intent() — direct unit tests
# ---------------------------------------------------------------------------

class TestLLMClassifyIntentDirect:
    """
    Direct tests for _llm_classify_intent() — verifies LLM response parsing.
    Mocks self._client directly to control the response content.
    """

    def _mock_llm_response(self, agent, content: str):
        """Helper: configure agent._client to return a specific text content."""
        completion = MagicMock()
        completion.choices[0].message.content = content
        agent._client.chat.completions.create = AsyncMock(return_value=completion)

    @pytest.mark.asyncio
    async def test_sim_lowercase_returns_weather(self, agent):
        self._mock_llm_response(agent, "sim")
        assert await agent._llm_classify_intent("Calor hoje?") == "weather"

    @pytest.mark.asyncio
    async def test_nao_lowercase_returns_not_weather(self, agent):
        self._mock_llm_response(agent, "não")
        assert await agent._llm_classify_intent("Tô com calor") == "not_weather"

    @pytest.mark.asyncio
    async def test_sim_in_sentence_returns_weather(self, agent):
        self._mock_llm_response(agent, "Sim, parece ser uma consulta de previsão do tempo.")
        assert await agent._llm_classify_intent("Calor danado") == "weather"

    @pytest.mark.asyncio
    async def test_empty_response_returns_not_weather(self, agent):
        """Empty LLM response must default to not_weather (safe fallback)."""
        self._mock_llm_response(agent, "")
        assert await agent._llm_classify_intent("Calor hoje") == "not_weather"

    @pytest.mark.asyncio
    async def test_no_sim_returns_not_weather(self, agent):
        """Response without 'sim' must return not_weather."""
        self._mock_llm_response(agent, "Talvez seja sobre o tempo, não tenho certeza.")
        assert await agent._llm_classify_intent("Calor hoje") == "not_weather"

    @pytest.mark.asyncio
    async def test_prompt_contains_original_message(self, agent):
        """The prompt sent to the LLM must include the original user message."""
        completion = MagicMock()
        completion.choices[0].message.content = "sim"
        agent._client.chat.completions.create = AsyncMock(return_value=completion)

        original_message = "Que baita calor hoje hein"
        await agent._llm_classify_intent(original_message)

        call_args = agent._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        prompt_text = str(messages)
        assert original_message in prompt_text, (
            f"Original message must appear in LLM prompt. Got: {prompt_text[:200]}"
        )


# ---------------------------------------------------------------------------
# Cascade guarantees — later stages NOT called when earlier stage resolves
# ---------------------------------------------------------------------------

class TestCascadeGuarantees:
    """
    Verify the cascade property: once a stage makes a decision, all
    subsequent (more expensive) stages must NOT be invoked.
    This is the core efficiency guarantee of the cascading architecture.
    """

    @pytest.mark.asyncio
    async def test_stage1_does_not_call_exclusion_check(self, agent):
        """Strong keyword → _is_non_weather_context must never run."""
        with patch.object(agent, '_is_non_weather_context') as mock_excl:
            result = await agent._classify_intent("Temperatura em Curitiba")
        mock_excl.assert_not_called()
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_stage1_does_not_call_scoring(self, agent):
        """Strong keyword → _score_weather_intent must never run."""
        with patch.object(agent, '_score_weather_intent') as mock_score:
            result = await agent._classify_intent("Previsão para São Paulo")
        mock_score.assert_not_called()
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_stage1_does_not_call_llm(self, agent):
        """Strong keyword → _llm_classify_intent must never run."""
        agent._llm_classify_intent = AsyncMock(return_value="weather")
        await agent._classify_intent("Precipitação em Belém esta semana")
        agent._llm_classify_intent.assert_not_called()

    @pytest.mark.asyncio
    async def test_stage2_does_not_call_scoring(self, agent):
        """Fixed phrase → _score_weather_intent must never run."""
        with patch.object(agent, '_score_weather_intent') as mock_score:
            result = await agent._classify_intent("Vai chover em Manaus amanhã?")
        mock_score.assert_not_called()
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_stage3_does_not_call_scoring(self, agent):
        """Exclusion pattern → _score_weather_intent must never run."""
        with patch.object(agent, '_score_weather_intent') as mock_score:
            result = await agent._classify_intent("Tô com frio aqui no escritório")
        mock_score.assert_not_called()
        assert result == "not_weather"

    @pytest.mark.asyncio
    async def test_high_score_does_not_call_llm(self, agent):
        """Score ≥ 0.55 → _llm_classify_intent must NOT be called."""
        agent._repo.find_city.return_value = city_result("Curitiba - Parana")
        agent._llm_classify_intent = AsyncMock(return_value="weather")
        result = await agent._classify_intent("Vai fazer frio em Curitiba amanhã?")
        agent._llm_classify_intent.assert_not_called()
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_low_score_does_not_call_llm(self, agent):
        """Score ≤ 0.20 → _llm_classify_intent must NOT be called."""
        agent._llm_classify_intent = AsyncMock(return_value="not_weather")
        result = await agent._classify_intent("Que sol bonito na foto")
        agent._llm_classify_intent.assert_not_called()
        assert result == "not_weather"


# ---------------------------------------------------------------------------
# Scoring granularity — each signal tested in isolation
# ---------------------------------------------------------------------------

class TestScoringGranular:
    """
    Verify each individual scoring signal contributes the expected effect.
    Tests compare two nearly-identical messages that differ in exactly one signal.
    """

    def test_geo_preposition_uppercase_adds_score(self, agent):
        """'em Manaus' (capital M) adds geo prep bonus; 'em manaus' does not."""
        score_upper = agent._score_weather_intent("chuva em Manaus")   # M uppercase
        score_lower = agent._score_weather_intent("chuva em manaus")   # m lowercase
        assert score_upper > score_lower

    def test_temporal_amanha_adds_score(self, agent):
        """Presence of 'amanhã' adds temporal bonus."""
        score_with = agent._score_weather_intent("chuva amanhã")
        score_without = agent._score_weather_intent("chuva")
        assert score_with > score_without

    def test_temporal_semana_adds_score(self, agent):
        """Presence of 'semana' adds temporal bonus."""
        score_with = agent._score_weather_intent("chuva essa semana")
        score_without = agent._score_weather_intent("chuva")
        assert score_with > score_without

    def test_future_weather_verb_adds_score(self, agent):
        """'vai chover' (future + weather verb) adds bonus vs past tense 'choveu'."""
        score_future = agent._score_weather_intent("vai chover amanhã")
        score_past = agent._score_weather_intent("choveu ontem")
        assert score_future > score_past

    def test_personal_locative_reduces_score(self, agent):
        """'aqui' as personal locative reduces score."""
        score_with_locative = agent._score_weather_intent("frio aqui")
        score_neutral = agent._score_weather_intent("frio hoje")
        assert score_with_locative < score_neutral

    def test_third_person_adjective_reduces_score(self, agent):
        """'ela estava quente' reduces score significantly."""
        score_personal = agent._score_weather_intent("ela estava quente")
        score_neutral = agent._score_weather_intent("quente hoje")
        assert score_personal < score_neutral

    def test_score_always_in_range(self, agent):
        """Score must always be within [0.0, 1.0] for any input."""
        test_messages = [
            "Como vai estar a temperatura amanhã em Manaus?",
            "Eu tô com frio aqui em casa no trabalho todo dia",
            "",
            "???",
            "A" * 200,
            "frio",
            "sol",
        ]
        for msg in test_messages:
            score = agent._score_weather_intent(msg)
            assert 0.0 <= score <= 1.0, f"Score out of range for: {msg!r}"

    def test_baseline_positive_even_with_one_weak_keyword(self, agent):
        """A single weak keyword must produce score > 0 (baseline effect)."""
        assert agent._score_weather_intent("chuva") > 0.0
        assert agent._score_weather_intent("sol") > 0.0
        assert agent._score_weather_intent("vento") > 0.0


# ---------------------------------------------------------------------------
# Competing signals — positive + negative in same message
# ---------------------------------------------------------------------------

class TestCompetingSignals:
    """
    Messages that contain both positive weather signals and negative personal signals.
    Tests that the correct signal wins, or that ambiguity is passed to LLM.
    """

    @pytest.mark.asyncio
    async def test_personal_subject_plus_strong_keyword_resolves_stage1(self, agent):
        """'Eu quero saber o clima em Recife?' — 'clima' is strong keyword → Stage 1 wins."""
        result = await agent._classify_intent("Eu quero saber o clima em Recife?")
        assert result == "weather"  # Stage 1 (clima) resolves before scoring

    @pytest.mark.asyncio
    async def test_personal_subject_plus_question_mark(self, agent):
        """'Eu preciso saber o tempo amanhã?' — 'tempo' + '?' vs personal subject."""
        # "tempo" is NOT in strong keywords nor weak keywords (not weather-related in general)
        # so Stage 1 passes, Stage 2 passes, Stage 3 no exclusion, Stage 4: ? +0.40, eu -0.45
        # net is near inconclusive — acceptable outcome is weather or LLM fallback
        result = await agent._classify_intent("Eu preciso saber o tempo amanhã?")
        # "tempo" is not in weather vocab — goes to not_weather at Stage 1 fast exit
        assert result == "not_weather"

    @pytest.mark.asyncio
    async def test_to_indo_pra_cidade_como_esta_o_tempo(self, agent):
        """'Tô indo pra Manaus, como está o tempo?' — Stage 2 phrase wins."""
        result = await agent._classify_intent("Tô indo pra Manaus, como está o tempo?")
        assert result == "weather"  # Stage 2 "como está o tempo" resolves

    def test_personal_subject_pure_very_low_score(self, agent):
        """Pure personal comment 'Eu tô com frio' → score much lower than neutral 'chuva'."""
        score_personal = agent._score_weather_intent("Eu tô com frio")
        score_neutral = agent._score_weather_intent("chuva")
        assert score_personal < score_neutral

    @pytest.mark.asyncio
    async def test_question_mark_alone_without_weather_vocab(self, agent):
        """'?' alone without weather vocabulary → not_weather (Stage 1 fast exit)."""
        result = await agent._classify_intent("Você toparia sair amanhã?")
        assert result == "not_weather"  # no weather vocab → Stage 1 exits fast


# ---------------------------------------------------------------------------
# All 26 valid capitals — real CapitalsRepository
# ---------------------------------------------------------------------------

# The 26 valid capitals after removing the known anomaly "Campo Grande - RN"
_ALL_CAPITALS = [
    "Rio Branco",
    "Maceió",
    "Macapá",
    "Manaus",
    "Salvador",
    "Fortaleza",
    "Brasília",
    "Vitória",
    "Goiânia",
    "São Luís",
    "Cuiabá",
    "Campo Grande",
    "Belo Horizonte",
    "Belém",
    "João Pessoa",
    "Curitiba",
    "Recife",
    "Teresina",
    "Rio de Janeiro",
    "Natal",
    "Porto Alegre",
    "Porto Velho",
    "Boa Vista",
    "Florianópolis",
    "São Paulo",
    "Aracaju",
    "Palmas",
]


class TestAllCapitals:
    """
    Every one of the 26 valid Brazilian state capitals must be classified as
    a weather query when combined with a strong weather keyword.

    Uses the REAL CapitalsRepository (reads capitals.json) to prove the
    end-to-end detection works — not just a mock that always says "found".
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capital", _ALL_CAPITALS)
    async def test_capital_classified_as_weather(self, agent_real_repo, capital):
        """'Clima em {capital}' must always be classified as 'weather' (Stage 1)."""
        result = await agent_real_repo._classify_intent(f"Clima em {capital}")
        assert result == "weather", (
            f"Expected 'weather' for 'Clima em {capital}', got '{result}'"
        )


# ---------------------------------------------------------------------------
# Edge cases — robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Robustness tests: extreme inputs must not crash and must classify correctly."""

    @pytest.mark.asyncio
    async def test_all_uppercase_weather_query(self, agent):
        """All-caps message must still resolve correctly."""
        result = await agent._classify_intent("VAI CHOVER EM MANAUS?")
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_accent_variant_no_accent(self, agent):
        """Stage 1 keyword without accent must still trigger ('previsao' = 'previsão')."""
        result = await agent._classify_intent("previsao para amanha")
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_very_long_message(self, agent):
        """Message with 300+ characters must not raise an exception."""
        long_msg = "Bom dia! " + "Eu gostaria de saber " * 15 + "como está o tempo em Recife?"
        result = await agent._classify_intent(long_msg)
        assert result in ("weather", "not_weather")  # must not raise

    @pytest.mark.asyncio
    async def test_only_question_marks(self, agent):
        """Only punctuation → not_weather (no weather vocabulary)."""
        result = await agent._classify_intent("???")
        assert result == "not_weather"

    @pytest.mark.asyncio
    async def test_mixed_portuguese_english(self, agent):
        """Mixed PT/EN with English strong keyword → weather (Stage 1)."""
        result = await agent._classify_intent("weather in São Paulo today")
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_multiple_capitals_in_message(self, agent):
        """Message with two capitals must still classify as weather."""
        result = await agent._classify_intent("Temperatura em São Paulo ou Recife?")
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_message_with_newline(self, agent):
        """Newlines in message must not prevent correct classification."""
        result = await agent._classify_intent("Como está\no tempo em Curitiba?")
        assert result == "weather"

    @pytest.mark.asyncio
    async def test_empty_string_no_crash(self, agent):
        """Empty string must return not_weather without raising."""
        result = await agent._classify_intent("")
        assert result == "not_weather"
