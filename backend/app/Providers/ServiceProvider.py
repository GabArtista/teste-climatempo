"""
Provedores de injeção de dependências para o FastAPI.

Este módulo centraliza a criação e o ciclo de vida de todos os serviços da aplicação.
Cada função registrada como dependência no FastAPI (`Depends()`) é responsável por
instanciar e entregar um serviço já configurado ao controller que o solicitar.

Padrão singleton via `@lru_cache`:
    Funções decoradas com `@lru_cache` retornam sempre a mesma instância dentro de um
    processo, evitando releituras de arquivo ou reconexões desnecessárias. Isso é
    equivalente a `app.state`, mas mais simples: não exige acesso ao objeto `app` nem
    manipulação do ciclo de vida com `lifespan`.

Cadeia de dependências:
    get_settings → get_capitals_repository
    get_settings + get_capitals_repository → get_weather_service
    get_settings + get_weather_service + get_capitals_repository → get_agent_service
"""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.Repositories.CapitalsRepository import CapitalsRepository
from app.Services.AgentService import AgentService
from app.Services.WeatherService import WeatherService
from config.settings import Settings, get_settings


@lru_cache
def get_capitals_repository() -> CapitalsRepository:
    """
    [Bloco 1 — Humanizado]
    Garante que o arquivo `capitals.json` seja lido apenas uma vez durante a vida
    do processo. Toda requisição que precisar da lista de capitais recebe a mesma
    instância, sem custo de I/O repetido.

    [Bloco 2 — Técnico]
    `@lru_cache` sem argumentos armazena o resultado da primeira chamada em memória
    e o reutiliza nas chamadas seguintes. Como `CapitalsRepository.__init__` lê e
    parseia o JSON em disco, o decorator efetivamente transforma o repositório em
    singleton por processo. Thread-safe nas versões Python >= 3.2 (GIL protege o
    cache em contextos síncronos; em contextos assíncronos o event loop serializa as
    chamadas iniciais).

    Returns:
        Instância singleton de CapitalsRepository com os dados de capitais já carregados.
    """
    return CapitalsRepository()


def get_weather_service(
    settings: Annotated[Settings, Depends(get_settings)],
    repo: Annotated[CapitalsRepository, Depends(get_capitals_repository)],
) -> WeatherService:
    """
    [Bloco 1 — Humanizado]
    Fornece o cliente de previsão do tempo já configurado com a URL da API e o
    repositório de capitais. Qualquer controller que precise buscar dados climáticos
    declara esta função como dependência e recebe um serviço pronto para uso.

    [Bloco 2 — Técnico]
    Não possui `@lru_cache` porque o FastAPI já reutiliza instâncias de dependências
    dentro do mesmo escopo de requisição. Recebe `settings` (URL do Open-Meteo,
    timeout HTTP) e `repo` (dados geográficos das capitais) via injeção automática.
    Instancia `WeatherService` com esses dois contratos a cada resolução de dependência.

    Args:
        settings: Configurações globais da aplicação (URL da API, timeout, etc.).
        repo: Repositório de capitais brasileiras já carregado em memória.

    Returns:
        Instância de WeatherService pronta para executar chamadas ao Open-Meteo.
    """
    return WeatherService(settings=settings, repo=repo)


def get_agent_service(
    settings: Annotated[Settings, Depends(get_settings)],
    weather_service: Annotated[WeatherService, Depends(get_weather_service)],
    repo: Annotated[CapitalsRepository, Depends(get_capitals_repository)],
) -> AgentService:
    """
    [Bloco 1 — Humanizado]
    Monta e entrega o agente conversacional completo, que integra o LLM (via Ollama)
    com a ferramenta de previsão do tempo. É o topo da cadeia de dependências: precisa
    de configurações, do serviço de clima e do repositório de capitais para funcionar.

    [Bloco 2 — Técnico]
    Recebe as três dependências já resolvidas pelo FastAPI antes de ser chamado.
    `AgentService` usa `settings` para conectar ao Ollama (modelo, URL, API key),
    `weather_service` para executar a tool call de previsão, e `repo` para validar
    se a cidade mencionada pelo usuário é uma capital brasileira conhecida.
    Sem `@lru_cache` — o FastAPI gerencia o ciclo de vida por requisição.

    Args:
        settings: Configurações globais (modelo Ollama, URLs, timeouts).
        weather_service: Serviço de previsão do tempo já instanciado.
        repo: Repositório de capitais para validação de cidades.

    Returns:
        Instância de AgentService pronta para processar mensagens do chat.
    """
    return AgentService(settings=settings, weather_service=weather_service, repo=repo)
