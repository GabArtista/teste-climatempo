"""HTTP controller for agent (chat) endpoints."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.Http.Requests.ChatRequest import ChatRequestSchema
from app.Models.ChatMessage import ChatResponse
from app.Providers.ServiceProvider import get_agent_service
from app.Services.AgentService import AgentService, OllamaUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the weather agent",
    description=(
        "Send a message to the LLM agent. The agent will call the weather tool "
        "when the user asks about forecast, temperature, or rain in a Brazilian city."
    ),
)
async def chat(
    request: ChatRequestSchema,
    agent: Annotated[AgentService, Depends(get_agent_service)],
) -> ChatResponse:
    """
    Process a user message through the agentic loop.

    The agent decides autonomously whether to call the weather tool.
    Supports multi-turn conversations via the `history` field.
    """
    logger.info("Chat request: message=%r", request.message[:50])
    try:
        return await agent.chat(
            message=request.message,
            history=request.history,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in chat endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error. Please try again.",
        ) from exc


@router.get(
    "/health",
    summary="Check agent and Ollama status",
    description="Returns whether the Ollama model is running and available.",
)
async def health(
    agent: Annotated[AgentService, Depends(get_agent_service)],
) -> dict:
    """Return health status of the LLM agent and Ollama backend."""
    return await agent.check_health()
