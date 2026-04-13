"""HTTP controller for direct weather forecast endpoints."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.Models.WeatherForecast import WeatherResponse
from app.Providers.ServiceProvider import get_weather_service
from app.Services.WeatherService import CityNotFoundError, WeatherAPIError, WeatherService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/weather", tags=["Weather"])


@router.get(
    "/",
    response_model=WeatherResponse,
    summary="Get weather forecast directly",
    description=(
        "Fetch weather forecast for a Brazilian state capital without going through "
        "the LLM agent. Useful for testing the Open-Meteo integration directly."
    ),
)
async def get_weather(
    city: Annotated[str, Query(min_length=1, description="Brazilian state capital name")],
    days: Annotated[int, Query(ge=1, le=7, description="Number of forecast days")] = 3,
    weather_service: WeatherService = Depends(get_weather_service),
) -> WeatherResponse:
    """
    Return daily weather forecast for the given city.

    Uses Open-Meteo API with temperature_2m_max, temperature_2m_min,
    and precipitation_sum variables.
    """
    logger.info("Direct weather request: city=%s, days=%d", city, days)
    try:
        return await weather_service.get_forecast(city=city, forecast_days=days)
    except CityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except WeatherAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in weather endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch weather data.",
        ) from exc


@router.get(
    "/cities",
    summary="List available cities",
    description="Returns all Brazilian state capitals available for weather queries.",
)
async def list_cities(
    weather_service: WeatherService = Depends(get_weather_service),
) -> dict:
    """Return list of all supported Brazilian state capitals."""
    cities = weather_service.get_repository().list_cities()
    return {"cities": cities, "count": len(cities)}


@router.get(
    "/data-quality",
    summary="Data quality report for capitals database",
    description=(
        "Returns anomalies detected in the capitals.json source file. "
        "Example: 'Campo Grande - Rio Grande do Norte' is incorrect — "
        "the capital of RN is Natal."
    ),
)
async def data_quality(
    weather_service: WeatherService = Depends(get_weather_service),
) -> dict:
    """
    Expose data integrity issues found in the capitals database.

    The source capitals.json contains at least one known anomaly:
    'Campo Grande - Rio Grande do Norte' is an incorrect entry.
    This endpoint documents the issue for transparency.
    """
    anomalies = weather_service.get_repository().get_anomalies()
    return {
        "anomalies_found": len(anomalies),
        "anomalies": anomalies,
        "note": (
            "Known incorrect entry 'Campo Grande - Rio Grande do Norte' "
            "has been removed. Queries for 'Campo Grande' resolve to "
            "Campo Grande - Mato Grosso do Sul (correct capital)."
        ),
    }
