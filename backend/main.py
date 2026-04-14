"""
Weather LLM Agent — Ponto de entrada da aplicação FastAPI.

Este módulo cria a instância `app`, registra os middlewares, inclui as rotas e
define o ciclo de vida da aplicação (startup/shutdown). É o arquivo que o
servidor ASGI (uvicorn) carrega diretamente.

Serviços dependentes:
    - Ollama: LLM local com suporte a function calling (padrão OpenAI-compatible).
    - Open-Meteo: API gratuita de previsão do tempo (sem autenticação).

Como executar:
    uvicorn main:app --reload --port 8000

Documentação interativa:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

Ordem dos middlewares (outermost = adicionado por último com `add_middleware`):
    1. ErrorHandlerMiddleware — captura qualquer exceção não tratada (outermost)
    2. LoggingMiddleware       — loga método, path, status e duração
    3. CORSMiddleware          — trata preflight e cabeçalhos CORS (innermost)
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
    """
    [Bloco 1 — Humanizado]
    Consulta o Ollama no startup e escolhe automaticamente o melhor modelo
    disponível segundo uma lista de preferências. Assim, a aplicação se adapta ao
    ambiente sem exigir configuração manual do `.env` quando o modelo preferido
    já está instalado.

    [Bloco 2 — Técnico]
    Chama `GET /v1/models` (endpoint OpenAI-compatible do Ollama) para obter os IDs
    dos modelos instalados. Itera `settings.model_priority` em ordem de preferência;
    para cada candidato verifica correspondência exata OU prefixo de versão
    (ex.: "qwen2.5:1.5b" casa com "qwen2.5:1.5b-instruct-q4"). O primeiro match
    atualiza `settings.ollama_model` diretamente no objeto singleton.
    Em caso de `APIConnectionError` (Ollama offline) ou qualquer outra exceção,
    registra um warning e mantém o modelo configurado no `.env` sem interromper
    o startup.

    Args:
        settings: Instância singleton de Settings que será mutada com o modelo
                  selecionado, se um match for encontrado.
    """
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
                # Verifica correspondência exata primeiro (ex: "qwen2.5:1.5b" == "qwen2.5:1.5b")
                # depois testa partial match por prefixo: candidate.split(":")[0] pega "qwen2.5"
                # de "qwen2.5:1.5b", permitindo que "qwen2.5:1.5b-instruct-q4_K_M" case com
                # o candidato "qwen2.5:1.5b" — útil quando o Ollama expõe variantes quantizadas.
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
        # Nenhum modelo da lista de prioridade foi encontrado: mantém o modelo do .env.
        # Retorna None implicitamente — o startup continua com o modelo configurado.
        logger.warning(
            "No priority model found in Ollama. Available: %s. Using configured: %s",
            sorted(available_ids), settings.ollama_model,
        )
    except OllamaConnectionError:
        # Ollama offline no startup: não interrompe a aplicação. O modelo do .env será
        # usado; o endpoint /health reportará o problema quando consultado.
        logger.warning(
            "Ollama not reachable at startup — skipping auto-selection. "
            "Using configured model: %s", settings.ollama_model,
        )
    except Exception as exc:
        logger.warning("Auto-model selection failed (%s). Using: %s", exc, settings.ollama_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [Bloco 1 — Humanizado]
    Gerencia o que acontece quando a aplicação sobe e quando ela encerra. No startup,
    tenta selecionar automaticamente o melhor modelo LLM disponível no Ollama e loga
    as URLs dos serviços externos. No shutdown, registra a mensagem de encerramento
    para facilitar o diagnóstico em logs de produção.

    [Bloco 2 — Técnico]
    Implementado como `@asynccontextmanager` — padrão exigido pelo FastAPI >= 0.93
    para substituir os decoradores `@app.on_event`. O bloco antes do `yield` é o
    startup; o bloco após o `yield` é o shutdown. Recebe `app: FastAPI` por contrato
    do framework, mas não é usado diretamente aqui (o estado é gerenciado via
    `settings` singleton). Chama `_auto_select_model` de forma assíncrona antes de
    liberar a aplicação para receber requisições.

    Args:
        app: Instância da aplicação FastAPI (injetada pelo framework).

    Yields:
        Controle ao framework enquanto a aplicação está em execução.
    """
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

# Middlewares são executados em ordem LIFO (Last In, First Out):
# o último adicionado com add_middleware() é o primeiro a processar a requisição.
# Ordem de execução na entrada da requisição: ErrorHandler → Logging → CORS.
# Ordem de execução na saída da resposta: CORS → Logging → ErrorHandler.
# ErrorHandler é adicionado primeiro (= mais externo) para capturar exceções de
# TODOS os outros middlewares, incluindo erros do próprio CORSMiddleware.
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    # localhost:3000 = React (Create React App); localhost:5173 = Vite (padrão dev).
    # Hardcoded para dev: em produção, deve ser lido de variável de ambiente para
    # permitir configuração por ambiente sem rebuild da imagem Docker.
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router)


@app.get("/", tags=["Root"], summary="API root")
async def root() -> dict:
    """
    [Bloco 1 — Humanizado]
    Ponto de entrada da API para quem acessa a URL raiz. Retorna metadados básicos
    da aplicação e os links mais úteis, facilitando a descoberta da documentação e
    do endpoint de health check sem precisar consultar o código.

    [Bloco 2 — Técnico]
    Endpoint GET `/` sem autenticação. Retorna um dicionário com `name`, `version`,
    `docs` (link para o Swagger UI) e `health` (link para o health check do agente).
    Os valores são lidos do singleton `settings` para garantir consistência com o
    que está configurado no ambiente.

    Returns:
        Dicionário com nome, versão e links de documentação e saúde da API.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/agent/health",
    }
