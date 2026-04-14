"""
Controller HTTP para endpoints diretos de previsão do tempo.

Expõe três endpoints sob o prefixo `/api/v1/weather`, todos sem passar pelo agente
LLM — úteis para testes da integração Open-Meteo, listagem de cidades suportadas
e transparência sobre a qualidade dos dados:

    - GET /         — previsão do tempo para uma capital brasileira
    - GET /cities   — lista todas as capitais disponíveis
    - GET /data-quality — anomalias detectadas no arquivo capitals.json

Mapeamento de exceções de domínio para status HTTP:
    - `CityNotFoundError` → 404 Not Found
    - `WeatherAPIError`   → 502 Bad Gateway
    - Demais exceções     → 500 Internal Server Error
"""
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
    [Bloco 1 — Humanizado]
    Busca a previsão do tempo diretamente da API Open-Meteo para uma capital
    brasileira, sem envolver o agente LLM. Ideal para testar a integração com a
    API de clima de forma isolada ou para construir UIs que exibem dados brutos
    de previsão.

    [Bloco 2 — Técnico]
    Valida `city` (min_length=1) e `days` (1–7) via query parameters do FastAPI.
    Delega ao `WeatherService.get_forecast()`, que consulta o Open-Meteo com as
    variáveis `temperature_2m_max`, `temperature_2m_min` e `precipitation_sum`.
    Retorna `WeatherResponse` com lista de `DailyForecast` e metadados.
    Mapeamento de exceções:
        - `CityNotFoundError` → HTTP 404 (cidade não está no capitals.json)
        - `WeatherAPIError`   → HTTP 502 (falha na chamada ao Open-Meteo)
        - `Exception` genérica → HTTP 500 (erro inesperado)

    Args:
        city: Nome da capital brasileira a ser consultada (case-insensitive no serviço).
        days: Número de dias de previsão, entre 1 e 7. Padrão: 3.
        weather_service: Instância do WeatherService injetada pelo FastAPI.

    Returns:
        WeatherResponse com previsões diárias e metadados da cidade.

    Raises:
        HTTPException 404: Cidade não encontrada na base de capitais.
        HTTPException 502: Falha na comunicação com a API Open-Meteo.
        HTTPException 500: Erro inesperado no processamento.
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
    """
    [Bloco 1 — Humanizado]
    Lista todas as capitais brasileiras disponíveis para consulta de previsão do
    tempo. Permite que o frontend popule um seletor de cidades ou que o usuário
    descubra os nomes aceitos pelo sistema antes de fazer uma consulta.

    [Bloco 2 — Técnico]
    Chama `CapitalsRepository.list_cities()` via `weather_service.get_repository()`,
    que retorna uma lista de strings com os nomes das capitais conforme registrados
    no `capitals.json` (após remoção das anomalias conhecidas). Retorna um dicionário
    com duas chaves: `cities` (lista de nomes) e `count` (total de capitais).

    Args:
        weather_service: Instância do WeatherService injetada pelo FastAPI.

    Returns:
        Dicionário `{"cities": [...], "count": N}` com todas as capitais suportadas.
    """
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
    [Bloco 1 — Humanizado]
    Expõe os problemas de qualidade encontrados no arquivo `capitals.json` fornecido
    como base de dados do desafio. Existe para demonstrar que a aplicação detecta e
    trata ativamente as inconsistências dos dados de entrada, em vez de silenciosamente
    propagar informações incorretas. Útil para debugging e para transparência com
    revisores do código.

    [Bloco 2 — Técnico]
    Chama `CapitalsRepository.get_anomalies()`, que retorna a lista de entradas
    identificadas como incorretas durante o carregamento do `capitals.json`.
    Anomalia conhecida: "Campo Grande - Rio Grande do Norte" é um dado incorreto —
    a capital do RN é Natal, e Campo Grande é capital do Mato Grosso do Sul.
    O repositório remove a entrada incorreta e consultas por "Campo Grande" resolvem
    para o registro correto. O retorno inclui `anomalies_found` (int), `anomalies`
    (lista de strings) e `note` explicativa sobre a correção aplicada.

    Args:
        weather_service: Instância do WeatherService injetada pelo FastAPI.

    Returns:
        Dicionário com a contagem de anomalias, a lista delas e uma nota explicativa.
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
