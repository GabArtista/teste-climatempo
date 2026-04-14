"""
Middleware global de tratamento de erros para respostas JSON consistentes.

Propósito de segurança:
    Qualquer exceção não capturada pelos controllers chegaria ao cliente como uma
    resposta HTTP 500 com traceback ou mensagem interna exposta — um risco de
    segurança (information disclosure). Este middleware intercepta essas exceções
    antes que cheguem ao cliente, suprime os detalhes internos na resposta e os
    registra integralmente no log do servidor.

Assimetria intencional log vs. resposta:
    - Log: stack trace completo visível para o operador/desenvolvedor.
    - Resposta HTTP: mensagem genérica sem detalhes de implementação.
"""
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    [Bloco 1 — Humanizado]
    Protege o cliente de receber mensagens de erro internas da aplicação. Age como
    uma rede de segurança: qualquer exceção que escape dos controllers é capturada
    aqui, logada para análise interna e convertida em uma resposta JSON padronizada
    e segura.

    [Bloco 2 — Técnico]
    Herda de `BaseHTTPMiddleware` (Starlette). Deve ser o middleware mais externo da
    pilha — no FastAPI, isso significa ser adicionado por último via `add_middleware`,
    pois o Starlette empilha em ordem LIFO. Intercepta todas as rotas registradas no
    app, incluindo rotas de outros middlewares internos.
    """

    async def dispatch(self, request: Request, call_next):
        """
        [Bloco 1 — Humanizado]
        Envolve o processamento de cada requisição em um bloco try/except. Se tudo
        correr bem, a resposta passa transparentemente. Se uma exceção não tratada
        ocorrer, o cliente recebe uma mensagem de erro genérica e o operador vê o
        stack trace completo nos logs — sem vazar detalhes internos ao usuário final.

        [Bloco 2 — Técnico]
        Chama `call_next(request)` dentro de um `try/except Exception`. Em caso de
        exceção, usa `logger.exception()` (que inclui o traceback completo) para
        registrar o erro no log do servidor. Retorna `JSONResponse` com status 500 e
        corpo `{"error": "Internal Server Error", "detail": "..."}`. A mensagem real
        da exceção (`str(exc)`) é deliberadamente omitida da resposta HTTP.

        Args:
            request: Objeto de requisição HTTP do Starlette/FastAPI.
            call_next: Callable assíncrono que passa a requisição para o próximo
                       middleware ou handler na cadeia.

        Returns:
            Response original em caso de sucesso, ou JSONResponse 500 em caso de
            exceção não tratada.
        """
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "detail": "An unexpected error occurred. Please try again.",
                },
            )
