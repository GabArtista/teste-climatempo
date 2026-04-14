"""
Middleware de logging de requisições HTTP.

Registra automaticamente cada requisição recebida pela aplicação com informações
suficientes para diagnóstico de performance e rastreamento de erros: método HTTP,
caminho, status code e duração em milissegundos.
"""
import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    [Bloco 1 — Humanizado]
    Adiciona uma linha de log para cada requisição HTTP processada pela aplicação.
    Com isso, é possível acompanhar o fluxo de uso, identificar endpoints lentos e
    detectar erros sem precisar instrumentar cada controller individualmente.

    [Bloco 2 — Técnico]
    Herda de `BaseHTTPMiddleware` (Starlette). Deve ser posicionado internamente em
    relação ao `ErrorHandlerMiddleware` para garantir que requisições que resultam
    em exceção também sejam logadas (o error handler captura antes de propagar).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        [Bloco 1 — Humanizado]
        Mede o tempo que a aplicação leva para processar cada requisição e registra
        esse resultado junto com o método HTTP, o caminho e o status da resposta.
        Útil para identificar endpoints lentos e monitorar o comportamento em produção.

        [Bloco 2 — Técnico]
        Utiliza `time.perf_counter()` para medir a duração — contador monotônico de
        alta resolução, imune a ajustes de relógio do sistema operacional (NTP, DST),
        ao contrário de `time.time()`. A duração é calculada em segundos e convertida
        para milissegundos multiplicando por 1000. O log segue o formato:
        `METHOD /path → STATUS_CODE (X.Xms)`.

        Args:
            request: Objeto de requisição HTTP do Starlette/FastAPI.
            call_next: Callable assíncrono que repassa a requisição ao próximo handler.

        Returns:
            Response original sem modificação — o middleware apenas observa,
            não altera o conteúdo da resposta.
        """
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
