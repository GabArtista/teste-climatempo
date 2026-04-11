"""Unit tests for WeatherService with mocked HTTP."""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from app.Services.WeatherService import WeatherService, CityNotFoundError, WeatherAPIError
from app.Repositories.CapitalsRepository import CapitalsRepository
from config.settings import Settings


@pytest.fixture
def settings():
    return Settings(
        open_meteo_base_url="https://api.open-meteo.com/v1",
        request_timeout=10,
    )


@pytest.fixture
def repo():
    return CapitalsRepository()


@pytest.fixture
def service(settings, repo):
    return WeatherService(settings=settings, repo=repo)


@pytest.mark.asyncio
async def test_get_forecast_success(service, sample_weather_payload):
    mock_response = MagicMock()
    mock_response.json.return_value = sample_weather_payload
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await service.get_forecast(city="São Paulo", forecast_days=3)

    assert result.city is not None
    assert len(result.forecasts) == 3
    assert result.forecasts[0].temp_max == pytest.approx(28.5)
    assert result.forecasts[0].precipitation == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_city_not_found_raises(service):
    with pytest.raises(CityNotFoundError):
        await service.get_forecast(city="CidadeInexistente", forecast_days=3)


@pytest.mark.asyncio
async def test_api_timeout_raises(service):
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(WeatherAPIError, match="tempo"):
            await service.get_forecast(city="São Paulo", forecast_days=3)


@pytest.mark.asyncio
async def test_api_http_error_raises(service):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock(status_code=500)
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(WeatherAPIError):
            await service.get_forecast(city="São Paulo", forecast_days=3)
