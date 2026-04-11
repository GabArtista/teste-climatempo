"""API v1 route aggregator."""
from fastapi import APIRouter

from app.Http.Controllers.AgentController import router as agent_router
from app.Http.Controllers.WeatherController import router as weather_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(agent_router)
api_router.include_router(weather_router)
