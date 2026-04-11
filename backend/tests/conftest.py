"""Shared pytest fixtures for all test suites."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Provide a FastAPI test client for the full application."""
    return TestClient(app)


@pytest.fixture
def sample_weather_payload() -> dict:
    """Sample Open-Meteo API response for São Paulo."""
    return {
        "daily": {
            "time": ["2024-01-15", "2024-01-16", "2024-01-17"],
            "temperature_2m_max": [28.5, 30.1, 27.3],
            "temperature_2m_min": [20.1, 21.3, 19.8],
            "precipitation_sum": [0.0, 5.2, 12.4],
        }
    }


@pytest.fixture
def sao_paulo_city() -> dict:
    """São Paulo city data."""
    return {
        "name": "Sao Paulo - Sao Paulo",
        "latitude": -23.548,
        "longitude": -46.636,
    }
