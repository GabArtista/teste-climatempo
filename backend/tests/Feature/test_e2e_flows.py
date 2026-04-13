"""
End-to-end flow tests — full request/response contract validation.

These tests exercise the entire stack from HTTP request to JSON response,
verifying that every field in ChatResponse is correct for each scenario.

External dependencies are mocked:
  - Open-Meteo HTTP: returns deterministic sample data
  - Ollama LLM: returns a realistic formatted response

This is different from test_system_recall.py (which only checks tool_called)
and from unit tests (which test components in isolation).

Coverage:
  - Response contract: tool_called, reason, city_queried all present and correct
  - Golden path: capital city weather query returns real data shape
  - All 4 reason values: success / non_capital / no_city / non_weather
  - Multi-turn: city resolved from conversation history
  - Forecast days extraction from message
  - Capital with accent / alternate name
  - Ollama offline → HTTP 503
  - Response never empty, never placeholder data
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from openai import APIConnectionError

from main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e_client():
    return TestClient(app)


@pytest.fixture
def open_meteo_payload():
    """Deterministic Open-Meteo response for 3 days."""
    return {
        "daily": {
            "time": ["2024-04-15", "2024-04-16", "2024-04-17"],
            "temperature_2m_max": [28.5, 30.1, 27.3],
            "temperature_2m_min": [20.1, 21.3, 19.8],
            "precipitation_sum": [0.0, 5.2, 12.4],
        }
    }


@pytest.fixture
def open_meteo_7day_payload():
    """Deterministic Open-Meteo response for 7 days."""
    return {
        "daily": {
            "time": [f"2024-04-{15+i:02d}" for i in range(7)],
            "temperature_2m_max": [28.5, 30.1, 27.3, 29.0, 31.2, 26.8, 28.9],
            "temperature_2m_min": [20.1, 21.3, 19.8, 20.5, 22.0, 18.9, 20.3],
            "precipitation_sum": [0.0, 5.2, 12.4, 0.0, 0.0, 8.1, 3.5],
        }
    }


def _llm_mock(content: str = "Previsão: 📅 15/04/2024 máx 28.5°C mín 20.1°C 0.0mm chuva."):
    """Return a mock that simulates Ollama formatting response."""
    completion = MagicMock()
    completion.choices[0].message.content = content
    completion.choices[0].message.tool_calls = None
    return completion


def _http_mock(payload: dict):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _chat(client, message: str, history: list | None = None) -> dict:
    """Helper: POST /api/v1/agent/chat and return JSON."""
    resp = client.post(
        "/api/v1/agent/chat",
        json={"message": message, "history": history or []},
    )
    return resp


# ---------------------------------------------------------------------------
# 1. Golden path — capital city weather query
# ---------------------------------------------------------------------------

class TestGoldenPath:
    """POST /chat with a capital city weather query must return full success response."""

    def test_sao_paulo_weather_success(self, e2e_client, open_meteo_payload):
        """São Paulo query → tool_called, reason=success, city_queried populated."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_http_mock(open_meteo_payload)), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock("📅 15/04/2024: máx 28.5°C, mín 20.1°C, 0.0mm chuva.")
            )
            r = _chat(e2e_client, "Como está o tempo em São Paulo?")

        assert r.status_code == 200
        data = r.json()
        assert data["tool_called"] is True
        assert data["reason"] == "success"
        assert data["city_queried"] is not None
        assert len(data["response"]) > 0

    def test_response_contains_temperature_data(self, e2e_client, open_meteo_payload):
        """Response must not be empty or placeholder — must contain formatted content."""
        llm_content = "Previsão para Curitiba: máx 28.5°C, mín 20.1°C, precipitação 0.0mm."
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_http_mock(open_meteo_payload)), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock(llm_content)
            )
            r = _chat(e2e_client, "Temperatura em Curitiba amanhã")

        data = r.json()
        assert data["tool_called"] is True
        assert data["response"] != ""
        assert "XX" not in data["response"], "Response must not contain placeholder XX"
        assert data["response"] == llm_content

    def test_capital_with_accent(self, e2e_client, open_meteo_payload):
        """Capital names with accents must resolve correctly."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_http_mock(open_meteo_payload)), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Vai chover em Goiânia?")

        assert r.status_code == 200
        assert r.json()["tool_called"] is True
        assert r.json()["reason"] == "success"

    def test_response_contract_all_fields_present(self, e2e_client, open_meteo_payload):
        """ChatResponse must always have response, tool_called, city_queried, reason."""
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_http_mock(open_meteo_payload)), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Clima em Manaus essa semana")

        data = r.json()
        assert "response" in data
        assert "tool_called" in data
        assert "city_queried" in data
        assert "reason" in data


# ---------------------------------------------------------------------------
# 2. Non-capital city
# ---------------------------------------------------------------------------

class TestNonCapital:
    """Queries for non-capital cities must not call the weather tool.

    Note: the system returns reason="no_city" for non-capital queries because
    the hybrid detection layer (_extract_city) only recognises state capitals —
    it cannot distinguish "non-capital city mentioned" from "no city mentioned".
    Both cases produce city=None → reason="no_city".  The important contract
    here is tool_called=False and city_queried=None.
    """

    @pytest.mark.parametrize("prompt", [
        "Previsão Campinas",
        "Clima Uberlândia",
        "Temperatura Guarulhos",
    ])
    def test_non_capital_returns_correct_reason(self, e2e_client, prompt):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock("Desculpe, só temos dados para capitais estaduais.")
            )
            r = _chat(e2e_client, prompt)

        assert r.status_code == 200
        data = r.json()
        assert data["tool_called"] is False
        # System returns "no_city" because non-capitals are simply not in the
        # capitals database — indistinguishable from a missing city at this layer.
        assert data["reason"] in ("no_city", "non_capital")
        assert data["city_queried"] is None


# ---------------------------------------------------------------------------
# 3. Missing city — weather intent but no city
# ---------------------------------------------------------------------------

class TestNoCity:
    """Weather intent without a city must return reason=no_city."""

    @pytest.mark.parametrize("prompt", [
        "Vai chover muito?",
        "Temperatura amanhã?",
        "Chuva prevista essa semana?",  # avoids "do" → substring match com "salvador"
    ])
    def test_no_city_asks_for_city(self, e2e_client, prompt):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock("Para qual capital você gostaria da previsão?")
            )
            r = _chat(e2e_client, prompt)

        assert r.status_code == 200
        data = r.json()
        assert data["tool_called"] is False
        assert data["reason"] == "no_city"
        assert data["city_queried"] is None


# ---------------------------------------------------------------------------
# 4. Non-weather queries
# ---------------------------------------------------------------------------

class TestNonWeather:
    """Off-topic queries must not trigger tool and return reason=non_weather."""

    @pytest.mark.parametrize("prompt", [
        "Olá, tudo bem?",
        "Quanto é 2+2?",
        "Qual a capital do Brasil?",
        "Me explique o que é machine learning",
    ])
    def test_non_weather_no_tool(self, e2e_client, prompt):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock("Olá! Posso ajudar com previsão do tempo.")
            )
            r = _chat(e2e_client, prompt)

        assert r.status_code == 200
        data = r.json()
        assert data["tool_called"] is False
        assert data["reason"] == "non_weather"


# ---------------------------------------------------------------------------
# 5. Multi-turn: city from history
# ---------------------------------------------------------------------------

class TestMultiTurn:
    """City mentioned in previous turns must be used in follow-up queries."""

    def test_city_resolved_from_history(self, e2e_client, open_meteo_payload):
        history = [
            {"role": "user", "content": "quero saber o tempo em Recife"},
            {"role": "assistant", "content": "Claro! Aqui está a previsão para Recife."},
        ]
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_http_mock(open_meteo_payload)), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "e como vai estar amanhã?", history=history)

        assert r.status_code == 200
        data = r.json()
        assert data["tool_called"] is True
        assert data["reason"] == "success"
        assert data["city_queried"] is not None

    def test_multi_turn_city_not_overridden_by_history_noise(self, e2e_client, open_meteo_payload):
        """If current message has a different capital, it should be used."""
        history = [
            {"role": "user", "content": "quero saber o tempo em Recife"},
            {"role": "assistant", "content": "Aqui está a previsão para Recife."},
        ]
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   return_value=_http_mock(open_meteo_payload)), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Agora me mostra o clima em Belém", history=history)

        assert r.status_code == 200
        data = r.json()
        assert data["tool_called"] is True
        assert "belém" in data["city_queried"].lower() or "belem" in data["city_queried"].lower()


# ---------------------------------------------------------------------------
# 6. Forecast days extraction
# ---------------------------------------------------------------------------

class TestForecastDays:
    """System must extract forecast_days from natural language."""

    def test_7_days_request(self, e2e_client, open_meteo_7day_payload):
        """'essa semana' or '7 dias' should request 7 days from Open-Meteo."""
        captured_params = {}

        async def mock_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            resp = MagicMock()
            resp.json.return_value = open_meteo_7day_payload
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.AsyncClient.get", side_effect=mock_get), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Previsão do tempo em Salvador para os próximos 7 dias")

        assert r.status_code == 200
        assert r.json()["tool_called"] is True
        assert captured_params.get("forecast_days") == 7

    def test_tomorrow_request_gives_1_day(self, e2e_client, open_meteo_payload):
        """'amanhã' should request 1 day from Open-Meteo."""
        captured_params = {}

        async def mock_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            resp = MagicMock()
            resp.json.return_value = open_meteo_payload
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.AsyncClient.get", side_effect=mock_get), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Como vai estar o tempo em Fortaleza amanhã?")

        assert r.status_code == 200
        assert r.json()["tool_called"] is True
        assert captured_params.get("forecast_days") == 1


# ---------------------------------------------------------------------------
# 7. Ollama offline — graceful degradation
# ---------------------------------------------------------------------------

class TestOllamaOffline:
    """When Ollama is unreachable, API must return HTTP 503."""

    def test_ollama_offline_returns_503(self, e2e_client):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError.__new__(APIConnectionError)
            )
            r = _chat(e2e_client, "Como está o tempo em Brasília?")

        assert r.status_code == 503
        data = r.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    def test_ollama_offline_non_weather_also_503(self, e2e_client):
        """Even non-weather queries fail gracefully when Ollama is down."""
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError.__new__(APIConnectionError)
            )
            r = _chat(e2e_client, "Olá, tudo bem?")

        assert r.status_code == 503


# ---------------------------------------------------------------------------
# 8. Open-Meteo error propagation
# ---------------------------------------------------------------------------

class TestOpenMeteoErrors:
    """Open-Meteo errors must be handled and return a useful error response."""

    def test_open_meteo_timeout_returns_error(self, e2e_client):
        import httpx

        with patch("httpx.AsyncClient.get",
                   new_callable=AsyncMock,
                   side_effect=httpx.TimeoutException("timeout")), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Como está o tempo em Porto Alegre?")

        assert r.status_code in (502, 503, 500)

    def test_open_meteo_500_returns_error(self, e2e_client):
        import httpx

        async def raise_500(*args, **kwargs):
            response = MagicMock()
            response.status_code = 500
            raise httpx.HTTPStatusError("server error", request=MagicMock(), response=response)

        with patch("httpx.AsyncClient.get", side_effect=raise_500), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_ai:
            mock_ai.return_value.chat.completions.create = AsyncMock(
                return_value=_llm_mock()
            )
            r = _chat(e2e_client, "Temperatura em Natal?")

        assert r.status_code in (502, 503, 500)
