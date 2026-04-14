"""
Controller HTTP para os endpoints do agente conversacional.

Expõe dois endpoints sob o prefixo `/api/v1/agent`:
    - POST /chat   — envia uma mensagem ao agente LLM e recebe a resposta em texto.
    - GET  /health — verifica se o Ollama está acessível e qual modelo está ativo.

A injeção de dependências é feita via `Annotated[AgentService, Depends(get_agent_service)]`,
seguindo o padrão do FastAPI. Mapeamento de exceções de domínio para status HTTP:
    - `OllamaUnavailableError` → 503 Service Unavailable
    - Demais exceções            → 500 Internal Server Error
"""
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
    [Bloco 1 — Humanizado]
    Recebe a mensagem do usuário e a envia ao agente LLM, que decide de forma
    autônoma se precisa consultar a API de previsão do tempo. Suporta conversas
    multi-turno: o histórico de mensagens anteriores pode ser enviado para que o
    agente mantenha o contexto da conversa.

    [Bloco 2 — Técnico]
    Recebe `ChatRequestSchema` no corpo da requisição (campos: `message: str`,
    `history: list[ChatMessage]`). Delega ao `AgentService.chat()`, que executa o
    loop agêntico com o Ollama. Retorna `ChatResponse` com campos `response` (texto),
    `tool_called` (bool), `city_queried` (str | None) e `reason` (enum de motivo).
    Mapeamento de exceções:
        - `OllamaUnavailableError` → HTTP 503 (Ollama offline ou modelo não carregado)
        - `Exception` genérica     → HTTP 500 (erro inesperado, logado com detalhes)

    Args:
        request: Corpo da requisição com a mensagem do usuário e histórico opcional.
        agent: Instância do AgentService injetada pelo FastAPI.

    Returns:
        ChatResponse com a resposta do agente e metadados sobre a tool call.

    Raises:
        HTTPException 503: Quando o Ollama não está acessível.
        HTTPException 500: Para qualquer outro erro inesperado.
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
    """
    [Bloco 1 — Humanizado]
    Verifica se o agente e o Ollama estão funcionando corretamente. Útil para
    monitoramento, para o frontend mostrar um indicador de status e para confirmar
    que o modelo selecionado no startup está de fato respondendo.

    [Bloco 2 — Técnico]
    Delega ao `AgentService.check_health()`, que tenta conectar ao Ollama e retorna
    um dicionário com informações como `status`, `model` e `ollama_reachable`.
    Não lança exceções diretamente — o próprio `check_health` trata falhas de conexão
    e reflete o estado no campo `status` do retorno.

    Args:
        agent: Instância do AgentService injetada pelo FastAPI.

    Returns:
        Dicionário com o status de saúde do agente e do backend Ollama.
    """
    return await agent.check_health()
