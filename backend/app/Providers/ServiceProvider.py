"""Dependency injection providers for FastAPI."""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.Repositories.CapitalsRepository import CapitalsRepository
from app.Services.AgentService import AgentService
from app.Services.WeatherService import WeatherService
from config.settings import Settings, get_settings


@lru_cache
def get_capitals_repository() -> CapitalsRepository:
    """Provide singleton CapitalsRepository."""
    return CapitalsRepository()


def get_weather_service(
    settings: Annotated[Settings, Depends(get_settings)],
    repo: Annotated[CapitalsRepository, Depends(get_capitals_repository)],
) -> WeatherService:
    """Provide WeatherService with injected dependencies."""
    return WeatherService(settings=settings, repo=repo)


def get_agent_service(
    settings: Annotated[Settings, Depends(get_settings)],
    weather_service: Annotated[WeatherService, Depends(get_weather_service)],
    repo: Annotated[CapitalsRepository, Depends(get_capitals_repository)],
) -> AgentService:
    """Provide AgentService with injected dependencies."""
    return AgentService(settings=settings, weather_service=weather_service, repo=repo)
