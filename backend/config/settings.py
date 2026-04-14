"""
Configurações da aplicação carregadas via variáveis de ambiente.

Utiliza `pydantic-settings` (`BaseSettings`) para carregar automaticamente os
valores do arquivo `.env` na raiz do projeto ou das variáveis de ambiente do
sistema operacional. Valores não definidos no ambiente usam os padrões declarados
na classe `Settings`.

Para sobrescrever qualquer configuração, crie ou edite o arquivo `.env`:
    OLLAMA_MODEL=llama3.1:8b
    LOG_LEVEL=DEBUG
    REQUEST_TIMEOUT=60
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    [Bloco 1 — Humanizado]
    Centraliza todas as configurações da aplicação em um único lugar. Qualquer
    valor sensível ou dependente de ambiente (URLs, timeouts, modelos) é declarado
    aqui com um padrão seguro para desenvolvimento local. Para produção ou outros
    ambientes, basta ajustar o `.env` sem tocar no código.

    [Bloco 2 — Técnico]
    Herda de `pydantic_settings.BaseSettings`. O `model_config` aponta para `.env`
    com encoding UTF-8 e busca case-insensitive (OLLAMA_MODEL == ollama_model).
    `protected_namespaces=('settings_',)` evita conflito com o namespace reservado
    do Pydantic. Todas as variáveis são opcionais — têm valores padrão funcionais
    para desenvolvimento local.

    Variáveis expostas:
        ollama_base_url   : URL base da API OpenAI-compatible do Ollama.
        ollama_model      : Modelo padrão se a auto-seleção não encontrar match.
        ollama_api_key    : Placeholder exigido pelo cliente OpenAI (não validado pelo Ollama).
        model_priority    : Lista ordenada de modelos preferidos para auto-seleção no startup.
                            O qwen2.5:7b lidera por ter melhor capacidade de function calling;
                            qwen2.5:1.5b é o fallback mínimo recomendado.
        open_meteo_base_url: URL base da API Open-Meteo (sem autenticação).
        request_timeout   : Timeout em segundos para requisições HTTP externas.
        app_name          : Nome exibido no Swagger UI e nos logs.
        app_version       : Versão semântica da aplicação.
        app_description   : Descrição exibida no Swagger UI.
        log_level         : Nível de log (DEBUG, INFO, WARNING, ERROR). Padrão: INFO.
        cors_origins      : Origens permitidas pelo CORS. Apenas localhost dev por padrão
                            (porta 3000 para Create React App, 5173 para Vite).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=('settings_',),
    )

    # Ollama / LLM
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_api_key: str = "ollama"  # Required by OpenAI client, not used by Ollama
    model_priority: list[str] = [
        "qwen2.5:7b",
        "llama3.1:8b",
        "qwen2.5:3b",
        "llama3.2:3b",
        "qwen2.5:1.5b",
    ]

    # Open-Meteo
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"

    # HTTP
    request_timeout: int = 30

    # App
    app_name: str = "Weather LLM Agent"
    app_version: str = "1.0.0"
    app_description: str = (
        "LLM agent with function calling for weather forecasts via Open-Meteo API. "
        "Built for the Climatempo technical challenge."
    )
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


_settings: Settings | None = None


def get_settings() -> Settings:
    """
    [Bloco 1 — Humanizado]
    Retorna a instância única de configurações da aplicação. Na primeira chamada,
    lê o arquivo `.env` e as variáveis de ambiente. Nas chamadas seguintes, devolve
    o objeto já criado sem releitura. Funciona como ponto central de acesso às
    configurações em todo o projeto.

    [Bloco 2 — Técnico]
    Implementa o padrão singleton lazy usando a variável de módulo `_settings`.
    Thread-safe em contextos síncronos pelo GIL do Python. Em contextos assíncronos
    (event loop único do uvicorn), a primeira chamada ocorre no startup síncrono
    do módulo (`settings = get_settings()` em `main.py`), antes de qualquer
    concorrência real. Registrada também como `Depends(get_settings)` no FastAPI,
    que gerencia o cache por si mesmo via `@lru_cache` no `ServiceProvider`.

    Returns:
        Instância singleton de `Settings` com todas as configurações carregadas.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
