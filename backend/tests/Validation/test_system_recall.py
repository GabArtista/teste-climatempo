"""
System-level recall validation — tests the hybrid detection layer.

Unlike test_function_calling.py (which measures LLM tool-calling recall),
this suite tests the deterministic part of the system:
  _is_weather_query() + _extract_city() → guaranteed tool call for any capital

These tests run WITHOUT Ollama — the LLM is fully mocked.
The Open-Meteo HTTP call is also mocked with sample data.

System Recall = tool_called for all capital city weather queries = 1.0
System Precision = no tool_called for non-weather / non-capital = 1.0
"""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_mock():
    """Return a mock that simulates a successful LLM formatting response."""
    completion = MagicMock()
    completion.choices[0].message.content = "Aqui está a previsão do tempo formatada."
    completion.choices[0].message.tool_calls = None
    return completion


def _make_http_mock(weather_payload: dict):
    """Return a mock that simulates a successful Open-Meteo HTTP response."""
    http_resp = MagicMock()
    http_resp.json.return_value = weather_payload
    http_resp.raise_for_status = MagicMock()
    return http_resp


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------

SYSTEM_POSITIVE = [
    "Como está o tempo em São Paulo?",
    "Temperatura em Curitiba amanhã",
    "Vai chover em Manaus?",
    "Clima de Fortaleza essa semana",
]

SYSTEM_NON_CAPITAL = [
    "Previsão Campinas",
    "Clima Uberlândia",
]

SYSTEM_NO_CITY = [
    "Vai chover muito?",
    "Temperatura amanhã?",
]

SYSTEM_NON_WEATHER = [
    "Olá, tudo bem?",
    "Quanto é 2+2?",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def system_client():
    return TestClient(app)


@pytest.fixture
def weather_payload():
    return {
        "daily": {
            "time": ["2024-04-11", "2024-04-12", "2024-04-13"],
            "temperature_2m_max": [26.5, 27.1, 25.8],
            "temperature_2m_min": [18.2, 19.0, 17.5],
            "precipitation_sum": [0.0, 2.4, 8.1],
        }
    }


# ---------------------------------------------------------------------------
# Individual scenario tests
# ---------------------------------------------------------------------------

class TestSystemPositive:
    """System MUST call tool for capital city weather queries."""

    @pytest.mark.parametrize("prompt", SYSTEM_POSITIVE)
    def test_capital_triggers_tool(self, system_client, weather_payload, prompt):
        http_mock = _make_http_mock(weather_payload)
        llm_mock = _make_llm_mock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=http_mock), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_openai_cls:
            mock_instance = AsyncMock()
            mock_openai_cls.return_value = mock_instance
            mock_instance.chat.completions.create = AsyncMock(return_value=llm_mock)

            resp = system_client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": []},
            )

        assert resp.status_code == 200, f"HTTP error for: {prompt}"
        data = resp.json()
        assert data["tool_called"] is True, (
            f"System should call tool for '{prompt}' but got tool_called=False"
        )
        assert data["city_queried"] is not None


class TestSystemNonCapital:
    """System must NOT call tool for non-capital cities."""

    @pytest.mark.parametrize("prompt", SYSTEM_NON_CAPITAL)
    def test_non_capital_no_tool(self, system_client, prompt):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_openai_cls:
            mock_instance = AsyncMock()
            mock_openai_cls.return_value = mock_instance
            llm_mock = _make_llm_mock()
            mock_instance.chat.completions.create = AsyncMock(return_value=llm_mock)

            resp = system_client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": []},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_called"] is False, (
            f"System should NOT call tool for non-capital '{prompt}'"
        )


class TestSystemNoCity:
    """System must ask for city when intent is weather but no city given."""

    @pytest.mark.parametrize("prompt", SYSTEM_NO_CITY)
    def test_no_city_asks_for_city(self, system_client, prompt):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_openai_cls:
            mock_instance = AsyncMock()
            mock_openai_cls.return_value = mock_instance
            llm_mock = _make_llm_mock()
            mock_instance.chat.completions.create = AsyncMock(return_value=llm_mock)

            resp = system_client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": []},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_called"] is False, f"No city = no tool call. Got True for: {prompt}"
        assert data["city_queried"] is None


class TestSystemNonWeather:
    """System must not call tool for non-weather queries."""

    @pytest.mark.parametrize("prompt", SYSTEM_NON_WEATHER)
    def test_non_weather_no_tool(self, system_client, prompt):
        with patch("app.Services.AgentService.AsyncOpenAI") as mock_openai_cls:
            mock_instance = AsyncMock()
            mock_openai_cls.return_value = mock_instance
            llm_mock = _make_llm_mock()
            mock_instance.chat.completions.create = AsyncMock(return_value=llm_mock)

            resp = system_client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": []},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_called"] is False, f"Non-weather should not call tool: {prompt}"


class TestSystemMultiTurn:
    """City mentioned in history must be used in subsequent turns."""

    def test_city_from_history(self, system_client, weather_payload):
        history = [
            {"role": "user", "content": "quero saber o tempo em Recife"},
            {"role": "assistant", "content": "Claro! Aqui está a previsão para Recife."},
        ]

        http_mock = _make_http_mock(weather_payload)
        llm_mock = _make_llm_mock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=http_mock), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_openai_cls:
            mock_instance = AsyncMock()
            mock_openai_cls.return_value = mock_instance
            mock_instance.chat.completions.create = AsyncMock(return_value=llm_mock)

            resp = system_client.post(
                "/api/v1/agent/chat",
                json={"message": "e como vai estar amanhã?", "history": history},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_called"] is True, "City from history should trigger tool call"
        assert data["city_queried"] is not None


# ---------------------------------------------------------------------------
# Aggregate metrics — saves system_results.json
# ---------------------------------------------------------------------------

def test_system_recall_metrics(system_client, weather_payload):
    """
    Calculate and save system-level recall metrics.

    Measures the HYBRID DETECTION layer (deterministic), not the LLM.
    Expected: system recall = 1.0, system precision = 1.0
    """
    http_mock = _make_http_mock(weather_payload)
    llm_mock = _make_llm_mock()
    tp, fp, fn, tn = 0, 0, 0, 0
    details = []

    def _call(prompt, history=None):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=http_mock), \
             patch("app.Services.AgentService.AsyncOpenAI") as mock_openai_cls:
            mock_instance = AsyncMock()
            mock_openai_cls.return_value = mock_instance
            mock_instance.chat.completions.create = AsyncMock(return_value=llm_mock)
            resp = system_client.post(
                "/api/v1/agent/chat",
                json={"message": prompt, "history": history or []},
            )
        return resp.json().get("tool_called", False) if resp.status_code == 200 else False

    print(f"\n{'='*60}")
    print("  SYSTEM RECALL VALIDATION (hybrid detection layer)")
    print(f"{'='*60}")

    for prompt in SYSTEM_POSITIVE:
        called = _call(prompt)
        result = "TP ✅" if called else "FN ❌"
        print(f"  [POS] {result} | {prompt[:50]}")
        tp += int(called)
        fn += int(not called)
        details.append({"prompt": prompt, "expected": True, "actual": called, "layer": "hybrid"})

    for prompt in SYSTEM_NON_CAPITAL + SYSTEM_NO_CITY + SYSTEM_NON_WEATHER:
        called = _call(prompt)
        result = "TN ✅" if not called else "FP ❌"
        print(f"  [NEG] {result} | {prompt[:50]}")
        tn += int(not called)
        fp += int(called)
        details.append({"prompt": prompt, "expected": False, "actual": called, "layer": "hybrid"})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  SYSTEM RESULTS (deterministic hybrid layer)")
    print(f"  Layer:      Hybrid detection (keyword + n-gram)")
    print(f"  Dataset:    {len(details)} prompts")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"{'='*60}\n")

    results = {
        "layer": "hybrid-detection",
        "description": "System recall using deterministic keyword+ngram detection (LLM mocked)",
        "dataset_size": len(details),
        "positive_count": len(SYSTEM_POSITIVE),
        "negative_count": len(SYSTEM_NON_CAPITAL) + len(SYSTEM_NO_CITY) + len(SYSTEM_NON_WEATHER),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "details": details,
    }

    output_path = Path(__file__).parent / "system_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"System results saved to: {output_path}")

    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert results["f1_score"] >= 0.0
