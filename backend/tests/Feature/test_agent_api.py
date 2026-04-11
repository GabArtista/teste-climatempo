"""Feature tests for agent and weather API endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.Models.ChatMessage import ChatResponse
from app.Models.WeatherForecast import WeatherResponse, DailyForecast
from datetime import date


@pytest.fixture
def mock_weather_response():
    return WeatherResponse(
        city="Sao Paulo - Sao Paulo",
        latitude=-23.548,
        longitude=-46.636,
        forecasts=[
            DailyForecast(date=date(2024, 1, 15), temp_max=28.5, temp_min=20.1, precipitation=0.0),
            DailyForecast(date=date(2024, 1, 16), temp_max=30.1, temp_min=21.3, precipitation=5.2),
            DailyForecast(date=date(2024, 1, 17), temp_max=27.3, temp_min=19.8, precipitation=12.4),
        ],
    )


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "docs" in data


def test_chat_empty_message_rejected(client):
    response = client.post("/api/v1/agent/chat", json={"message": "", "history": []})
    assert response.status_code == 422


def test_chat_blank_message_rejected(client):
    response = client.post("/api/v1/agent/chat", json={"message": "   ", "history": []})
    assert response.status_code == 422


def test_weather_city_not_found(client):
    response = client.get("/api/v1/weather/", params={"city": "Mordor", "days": 3})
    assert response.status_code == 404
    assert "error" in response.json() or "detail" in response.json()


def test_weather_invalid_days(client):
    response = client.get("/api/v1/weather/", params={"city": "São Paulo", "days": 0})
    assert response.status_code == 422


def test_weather_days_too_high(client):
    response = client.get("/api/v1/weather/", params={"city": "São Paulo", "days": 100})
    assert response.status_code == 422


def test_list_cities(client):
    response = client.get("/api/v1/weather/cities")
    assert response.status_code == 200
    data = response.json()
    assert "cities" in data
    # 26 valid capitals after removing the anomalous "Campo Grande - RN" entry
    assert data["count"] == 26


def test_data_quality_endpoint(client):
    """Data quality endpoint must report the known anomaly in capitals.json."""
    response = client.get("/api/v1/weather/data-quality")
    assert response.status_code == 200
    data = response.json()
    assert data["anomalies_found"] > 0
    assert "anomalies" in data


def test_agent_health(client):
    response = client.get("/api/v1/agent/health")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "status" in data


def test_weather_direct_success(client, mock_weather_response):
    with patch(
        "app.Services.WeatherService.WeatherService.get_forecast",
        new_callable=AsyncMock,
        return_value=mock_weather_response,
    ):
        response = client.get("/api/v1/weather/", params={"city": "São Paulo", "days": 3})

    assert response.status_code == 200
    data = response.json()
    assert "forecasts" in data
    assert len(data["forecasts"]) == 3
    assert data["forecasts"][0]["temp_max"] == pytest.approx(28.5)
