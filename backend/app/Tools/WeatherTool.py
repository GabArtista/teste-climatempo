"""
Definição e executor da ferramenta de previsão do tempo no formato OpenAI function calling.

[Bloco 1 — Humanizado]
Este módulo define a "ferramenta" de clima que o agente pode usar, no formato
padrão do OpenAI function calling. Também implementa a função que executa essa
ferramenta quando chamada — validando parâmetros, buscando os dados e retornando
o resultado em JSON.

[Bloco 2 — Técnico]
Dois componentes principais:

WEATHER_TOOL: dict no formato OpenAI tools spec (type="function") com schema JSON
  dos parâmetros aceitos. Usado para registrar a ferramenta no LLM e para
  documentar a interface esperada no teste técnico.

execute_weather_tool(): função assíncrona que recebe os argumentos (str JSON ou dict),
  valida, faz clamp de forecast_days para [1, 7] e chama WeatherService.get_forecast().
  Captura todas as exceções e retorna JSON de erro — nunca propaga exceções para a
  camada de chamada, garantindo resposta estruturada sempre.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Definição da ferramenta no formato OpenAI function calling
#
# Este dict é o contrato público da ferramenta: define o nome, a descrição
# (usada pelo LLM para decidir quando chamar) e o schema JSON dos parâmetros.
#
# Campos obrigatórios pelo formato OpenAI:
#   type: "function" (único tipo suportado atualmente)
#   function.name: identificador usado pelo LLM para referenciar a ferramenta
#   function.description: instrução em linguagem natural para o LLM sobre quando usar
#   function.parameters: schema JSON com propriedades e quais são obrigatórias
#
# O parâmetro forecast_days tem default=3, minimum=1, maximum=7 alinhados com
# os limites da Open-Meteo API no plano gratuito e com o clamp em execute_weather_tool().
# ---------------------------------------------------------------------------
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": (
            "Get daily weather forecast for a Brazilian city. "
            "Call this tool whenever the user asks about weather, temperature, "
            "rain, forecast, or climate for any city or location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the Brazilian city or state capital (e.g. 'São Paulo', 'Curitiba')",
                },
                "forecast_days": {
                    "type": "integer",
                    "description": "Number of days to forecast (1 to 7). Defaults to 3.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 7,
                },
            },
            "required": ["city"],
        },
    },
}


async def execute_weather_tool(
    arguments: str | dict,
    weather_service: Any,
) -> str:
    """
    Executa a ferramenta de previsão do tempo com os argumentos fornecidos.

    [Bloco 1 — Humanizado]
    É a ponte entre a definição da ferramenta e a execução real: recebe os
    parâmetros (cidade e dias), busca a previsão e devolve o resultado em JSON.
    Nunca deixa uma exceção vazar — qualquer erro vira um JSON de erro estruturado.

    [Bloco 2 — Técnico]
    Formato dual de entrada: aceita tanto str (JSON serializado, formato retornado
    pelo OpenAI quando o LLM chama a ferramenta) quanto dict (formato usado em
    testes e chamadas diretas). A distinção é feita com isinstance(arguments, str).

    Extração de parâmetros:
      - city: str obrigatório; se ausente ou vazio → retorna JSON de erro imediato
      - forecast_days: int opcional, default=3; convertido com int() para aceitar
        strings numéricas que o LLM possa retornar

    Clamp silencioso: forecast_days = max(1, min(7, forecast_days)) — valores fora
    de [1, 7] são ajustados sem erro, alinhados com os limites da Open-Meteo API.

    Tratamento de exceções: bloco try/except genérico captura qualquer exceção
    (incluindo CityNotFoundError, WeatherAPIError, erros de rede) e retorna
    json.dumps({"error": str(exc)}) — a camada de chamada sempre recebe JSON válido.

    Args:
        arguments: JSON string ou dict com chaves 'city' (obrigatório) e
                   'forecast_days' (opcional, inteiro 1-7).
        weather_service: Instância de WeatherService para buscar a previsão.

    Returns:
        JSON string com o WeatherResponse serializado em caso de sucesso,
        ou {"error": "mensagem"} em caso de falha.
    """
    try:
        # OpenAI retorna tool_call.function.arguments como string JSON serializada;
        # chamadas internas e testes unitários passam dict diretamente — ambos os
        # formatos são suportados para não forçar serialização desnecessária em testes.
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        city: str = args.get("city", "")
        forecast_days: int = int(args.get("forecast_days", 3))
        # Clamp silencioso: API aceita apenas 1-7 dias. Silêncio evita que o agente
        # entre em loop de retry tentando corrigir um parâmetro fora de range —
        # simplesmente ajusta para o valor mais próximo válido e prossegue.
        forecast_days = max(1, min(7, forecast_days))  # clamp to valid range

        if not city:
            return json.dumps({"error": "City name is required"})

        logger.info("Executing weather tool: city=%s, days=%d", city, forecast_days)
        result = await weather_service.get_forecast(city=city, forecast_days=forecast_days)
        return result.model_dump_json()

    except Exception as exc:
        # Captura Exception genérica (não tipos específicos) porque a tool deve
        # SEMPRE retornar JSON válido — nunca propagar exceção para o loop agêntico.
        # Qualquer exceção não-capturada quebraria o ciclo de resposta do agente.
        # O chamador diferencia sucesso de falha pela presença da chave "error" no JSON.
        logger.error("Weather tool execution failed: %s", exc)
        return json.dumps({"error": str(exc)})
