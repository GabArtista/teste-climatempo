"""Service for fetching weather data from Open-Meteo API."""
import logging
from datetime import date as DateType

import httpx

from app.Models.WeatherForecast import DailyForecast, WeatherResponse
from app.Repositories.CapitalsRepository import CapitalsRepository
from config.settings import Settings

logger = logging.getLogger(__name__)

_DAILY_VARIABLES = "temperature_2m_max,temperature_2m_min,precipitation_sum"


class CityNotFoundError(Exception):
    """Raised when a city is not in the capitals database."""


class WeatherAPIError(Exception):
    """Raised when the Open-Meteo API returns an error."""


class WeatherService:
    """Fetches daily weather forecasts from Open-Meteo API."""

    def __init__(self, settings: Settings, repo: CapitalsRepository) -> None:
        self._settings = settings
        self._repo = repo

    async def get_forecast(self, city: str, forecast_days: int = 3) -> WeatherResponse:
        """
        Fetch weather forecast for a Brazilian capital city.

        Args:
            city: City name (will be matched against capitals database).
            forecast_days: Number of days to forecast (1–7).

        Returns:
            WeatherResponse with daily temperature and precipitation data.

        Raises:
            CityNotFoundError: If city is not found in the capitals database.
            WeatherAPIError: If Open-Meteo API returns an error.
            httpx.TimeoutException: If the request times out.
        """
        city_data = self._repo.find_city(city)
        if city_data is None:
            raise CityNotFoundError(f"Cidade '{city}' não encontrada. Use o nome de uma capital brasileira.")

        params = {
            "latitude": city_data["latitude"],
            "longitude": city_data["longitude"],
            "daily": _DAILY_VARIABLES,
            "timezone": "auto",
            "forecast_days": forecast_days,
        }

        logger.info(
            "Fetching forecast: city=%s lat=%.3f lon=%.3f days=%d",
            city_data["name"], city_data["latitude"], city_data["longitude"], forecast_days,
        )

        async with httpx.AsyncClient(timeout=self._settings.request_timeout) as client:
            try:
                response = await client.get(
                    f"{self._settings.open_meteo_base_url}/forecast",
                    params=params,
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                logger.error("Open-Meteo API timeout: %s", exc)
                raise WeatherAPIError("Serviço de previsão do tempo não respondeu a tempo.") from exc
            except httpx.HTTPStatusError as exc:
                logger.error("Open-Meteo API error %d: %s", exc.response.status_code, exc)
                raise WeatherAPIError(f"Erro na API de previsão: {exc.response.status_code}") from exc

        data = response.json()
        return self._parse_response(city_data["name"], city_data, data)

    def _parse_response(
        self,
        city_name: str,
        city_data: dict,
        data: dict,
    ) -> WeatherResponse:
        """Parse Open-Meteo JSON response into WeatherResponse model."""
        daily = data.get("daily", {})
        dates: list[str] = daily.get("time", [])
        temp_max: list[float] = daily.get("temperature_2m_max", [])
        temp_min: list[float] = daily.get("temperature_2m_min", [])
        precipitation: list[float] = daily.get("precipitation_sum", [])

        forecasts = [
            DailyForecast(
                date=DateType.fromisoformat(dates[i]),
                temp_max=temp_max[i] if i < len(temp_max) else 0.0,
                temp_min=temp_min[i] if i < len(temp_min) else 0.0,
                precipitation=precipitation[i] if i < len(precipitation) else 0.0,
            )
            for i in range(len(dates))
        ]

        return WeatherResponse(
            city=city_name,
            latitude=city_data["latitude"],
            longitude=city_data["longitude"],
            forecasts=forecasts,
        )
