"""
Weather LLM Agent — FastAPI Application Entry Point.

Integrates a local LLM (Ollama / qwen2.5:1.5b) with the Open-Meteo API
via OpenAI-compatible function calling.

Run:
    uvicorn main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.routes import api_router
from app.Http.Middleware.ErrorHandler import ErrorHandlerMiddleware
from app.Http.Middleware.LoggingMiddleware import LoggingMiddleware
from config.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Ollama model: %s @ %s", settings.ollama_model, settings.ollama_base_url)
    logger.info("Open-Meteo API: %s", settings.open_meteo_base_url)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware (order matters — outermost first)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router)


@app.get("/", tags=["Root"], summary="API root")
async def root() -> dict:
    """Return API info and available endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/agent/health",
    }
