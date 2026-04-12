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
from config.settings import get_settings, Settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def _auto_select_model(settings: Settings) -> None:
    """Query Ollama and auto-select the best available model from priority list."""
    from openai import AsyncOpenAI
    from openai import APIConnectionError as OllamaConnectionError

    client = AsyncOpenAI(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
    )
    try:
        models_response = await client.models.list()
        available_ids = {m.id for m in models_response.data}
        logger.info("Ollama available models: %s", sorted(available_ids))

        for rank, candidate in enumerate(settings.model_priority, start=1):
            matched = any(
                m == candidate or m.startswith(candidate.split(":")[0] + ":" + candidate.split(":")[1])
                for m in available_ids
            )
            if matched:
                if settings.ollama_model != candidate:
                    logger.info(
                        "Auto-selected model: %s (priority rank %d, was: %s)",
                        candidate, rank, settings.ollama_model,
                    )
                    settings.ollama_model = candidate
                else:
                    logger.info("Model confirmed: %s (priority rank %d)", candidate, rank)
                return

        logger.warning(
            "No priority model found in Ollama. Available: %s. Using configured: %s",
            sorted(available_ids), settings.ollama_model,
        )
    except OllamaConnectionError:
        logger.warning(
            "Ollama not reachable at startup — skipping auto-selection. "
            "Using configured model: %s", settings.ollama_model,
        )
    except Exception as exc:
        logger.warning("Auto-model selection failed (%s). Using: %s", exc, settings.ollama_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    await _auto_select_model(settings)
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
