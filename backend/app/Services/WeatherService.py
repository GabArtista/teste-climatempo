"""
Serviço de previsão do tempo via Open-Meteo API.

[Bloco 1 — Humanizado]
Este módulo é responsável por buscar dados reais de previsão do tempo. Dado o nome
de uma capital brasileira, encontra suas coordenadas, consulta a API Open-Meteo e
devolve a previsão diária com temperaturas máxima/mínima e precipitação.

[Bloco 2 — Técnico]
Integração com Open-Meteo (https://api.open-meteo.com/v1/forecast) — API pública,
sem autenticação, com plano gratuito de até 10.000 requisições/dia. Comunicação
assíncrona via httpx.AsyncClient. Parâmetros solicitados:
  - temperature_2m_max: temperatura máxima diária a 2m de altura (°C)
  - temperature_2m_min: temperatura mínima diária a 2m de altura (°C)
  - precipitation_sum: precipitação acumulada diária (mm)
  - timezone: "auto" — Open-Meteo infere o fuso horário pelas coordenadas lat/lon

O mapeamento de erros HTTP segue a convenção da API:
  - 404 → cidade não encontrada (raro; coordenadas são do nosso banco)
  - 429 → rate limit atingido → WeatherAPIError com mensagem amigável
  - 5xx → serviço instável → WeatherAPIError com código HTTP
  - outros → WeatherAPIError genérico
"""
import logging
from datetime import date as DateType

import httpx

from app.Models.WeatherForecast import DailyForecast, WeatherResponse
from app.Repositories.CapitalsRepository import CapitalsRepository
from config.settings import Settings

logger = logging.getLogger(__name__)

_DAILY_VARIABLES = "temperature_2m_max,temperature_2m_min,precipitation_sum"


class CityNotFoundError(Exception):
    """Raised when a city is not in the capitals database."""


class WeatherAPIError(Exception):
    """Raised when the Open-Meteo API returns an error."""


class WeatherService:
    """
    Cliente HTTP assíncrono para a Open-Meteo API de previsão do tempo.

    [Bloco 1 — Humanizado]
    Encapsula toda a comunicação com a API de clima. Recebe um nome de cidade,
    busca as coordenadas no repositório, faz a requisição HTTP e devolve os dados
    estruturados. Trata erros de rede e da API de forma amigável.

    [Bloco 2 — Técnico]
    Utiliza httpx.AsyncClient com timeout configurável (settings.request_timeout).
    O cliente é instanciado por requisição (context manager dentro de get_forecast)
    — não há pool persistente, o que é adequado para a carga esperada do projeto.
    Depende do CapitalsRepository para traduzir nome de cidade em coordenadas lat/lon.
    """

    def __init__(self, settings: Settings, repo: CapitalsRepository) -> None:
        """
        Inicializa o serviço de clima com suas dependências.

        [Bloco 1 — Humanizado]
        Guarda as configurações necessárias para as requisições e o repositório
        de capitais para encontrar as coordenadas de cada cidade.

        [Bloco 2 — Técnico]
        Armazena referências a Settings (para base_url e request_timeout) e
        CapitalsRepository (para lookup lat/lon). O httpx.AsyncClient não é
        criado aqui — é instanciado sob demanda em get_forecast() como context
        manager para garantir o fechamento correto de conexões, mesmo em erros.

        Args:
            settings: Configurações da aplicação com open_meteo_base_url e request_timeout.
            repo: Repositório das capitais brasileiras para resolução de coordenadas.
        """
        self._settings = settings
        self._repo = repo

    def get_repository(self) -> CapitalsRepository:
        """Return the capitals repository used by this service."""
        return self._repo

    async def get_forecast(self, city: str, forecast_days: int = 3) -> WeatherResponse:
        """
        Busca a previsão do tempo diária para uma capital brasileira.

        [Bloco 1 — Humanizado]
        Recebe o nome de uma cidade, encontra suas coordenadas geográficas, consulta
        a API Open-Meteo e retorna a previsão estruturada com temperaturas e precipitação
        para o número de dias solicitado. Trata erros de rede e da API com mensagens
        claras em português.

        [Bloco 2 — Técnico]
        Fluxo de execução:
          1. Resolve o nome da cidade em coordenadas via CapitalsRepository.find_city().
             Se não encontrada → CityNotFoundError (não chama a API).
          2. Monta os query params: latitude, longitude, daily=_DAILY_VARIABLES,
             timezone="auto", forecast_days=N.
          3. Abre httpx.AsyncClient com timeout de settings.request_timeout e faz
             GET em {open_meteo_base_url}/forecast com raise_for_status().
          4. Em timeout → WeatherAPIError com mensagem de timeout.
             Em HTTPStatusError: 429 → mensagem de rate limit; >= 500 → indisponível
             com código; demais → genérico com código.
          5. Parseia o JSON via _parse_response() — arrays diários alinhados por índice.

        Formato de resposta da Open-Meteo (estrutura `daily`):
          {"time": ["2024-01-01", ...], "temperature_2m_max": [30.5, ...],
           "temperature_2m_min": [22.1, ...], "precipitation_sum": [0.0, ...]}
        Todos os arrays têm o mesmo tamanho (forecast_days itens).

        Args:
            city: Nome da cidade (tolerante a variações — delegado ao CapitalsRepository).
            forecast_days: Número de dias a prever, entre 1 e 7.

        Returns:
            WeatherResponse com lista de DailyForecast (data, temp_max, temp_min, precipitação).

        Raises:
            CityNotFoundError: Se a cidade não for encontrada no banco de capitais.
            WeatherAPIError: Se a API retornar erro HTTP ou timeout.
        """
        city_data = self._repo.find_city(city)
        if city_data is None:
            raise CityNotFoundError(f"Cidade '{city}' não encontrada. Use o nome de uma capital brasileira.")

        params = {
            # Open-Meteo não aceita nome de cidade — exige coordenadas numéricas.
            # Latitude e longitude são extraídas do capitals.json via CapitalsRepository.
            "latitude": city_data["latitude"],
            "longitude": city_data["longitude"],
            # Variáveis diárias solicitadas: temperatura máxima, mínima e precipitação acumulada.
            # Definidas em _DAILY_VARIABLES para evitar repetição e facilitar manutenção.
            "daily": _DAILY_VARIABLES,
            # "auto": Open-Meteo infere o fuso horário pelas coordenadas lat/lon.
            # Garante que as datas retornadas ("time") estejam no horário local da cidade,
            # não em UTC — essencial para mostrar "sexta-feira" corretamente ao usuário.
            "timezone": "auto",
            # Clampado em [1, 7] pelo caller (WeatherTool.execute_weather_tool) antes
            # de chegar aqui — este ponto assume que o valor já está no range válido.
            "forecast_days": forecast_days,
        }

        logger.info(
            "Fetching forecast: city=%s lat=%.3f lon=%.3f days=%d",
            city_data["name"], city_data["latitude"], city_data["longitude"], forecast_days,
        )

        async with httpx.AsyncClient(timeout=self._settings.request_timeout) as client:
            try:
                response = await client.get(
                    f"{self._settings.open_meteo_base_url}/forecast",
                    params=params,
                )
                # Lança httpx.HTTPStatusError para qualquer 4xx/5xx não tratado
                # explicitamente abaixo — centraliza o tratamento de erros HTTP
                # num único ponto sem precisar checar status manualmente no happy path.
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                logger.error("Open-Meteo API timeout: %s", exc)
                raise WeatherAPIError("Serviço de previsão do tempo não respondeu a tempo.") from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.error("Open-Meteo API error %d: %s", status, exc)
                if status == 429:
                    # Open-Meteo free tier: 10.000 req/dia sem autenticação.
                    # 429 indica que o limite diário foi atingido — retry imediato não adianta.
                    raise WeatherAPIError(
                        "Limite de requisições da API de previsão atingido. Tente novamente em instantes."
                    ) from exc
                elif status >= 500:
                    # Falha na infraestrutura da Open-Meteo — não é erro do cliente.
                    # Retry imediato raramente resolve 5xx; mensagem orienta o usuário
                    # a tentar novamente mais tarde.
                    raise WeatherAPIError(
                        f"Serviço de previsão do tempo temporariamente indisponível (HTTP {status})."
                    ) from exc
                else:
                    # 4xx inesperado (ex: 400 Bad Request por coordenadas inválidas,
                    # ou 404 se a API mudar de endpoint) — erro genérico com código HTTP
                    # para facilitar diagnóstico nos logs.
                    raise WeatherAPIError(
                        f"Erro na API de previsão do tempo (HTTP {status})."
                    ) from exc

        data = response.json()
        return self._parse_response(city_data["name"], city_data, data)

    def _parse_response(
        self,
        city_name: str,
        city_data: dict,
        data: dict,
    ) -> WeatherResponse:
        """
        Converte o JSON da Open-Meteo em um modelo WeatherResponse estruturado.

        [Bloco 1 — Humanizado]
        Transforma a resposta bruta da API (arrays paralelos de datas, temperaturas
        e precipitação) em uma lista de objetos DailyForecast, um por dia, mais fáceis
        de manipular e serializar.

        [Bloco 2 — Técnico]
        A Open-Meteo retorna arrays alinhados por índice dentro de `data["daily"]`:
          - time[i]: data no formato ISO 8601 ("2024-01-15")
          - temperature_2m_max[i]: temperatura máxima do dia i (float, °C)
          - temperature_2m_min[i]: temperatura mínima do dia i (float, °C)
          - precipitation_sum[i]: precipitação acumulada do dia i (float, mm)

        Itera sobre `time` como índice mestre; os demais arrays são acessados com
        guard `if i < len(array)` para segurança — fallback para 0.0 se array
        mais curto (não ocorre em condições normais, mas previne IndexError).

        Monta e retorna WeatherResponse(city, latitude, longitude, forecasts=[...]).

        Args:
            city_name: Nome original da cidade no formato "Cidade - Estado".
            city_data: Dict com 'latitude' e 'longitude' da cidade.
            data: JSON decodificado da resposta da Open-Meteo.

        Returns:
            WeatherResponse com lista de DailyForecast pronta para serialização.
        """
        daily = data.get("daily", {})
        dates: list[str] = daily.get("time", [])
        temp_max: list[float] = daily.get("temperature_2m_max", [])
        temp_min: list[float] = daily.get("temperature_2m_min", [])
        precipitation: list[float] = daily.get("precipitation_sum", [])

        # A Open-Meteo retorna arrays paralelos alinhados por posição: dates[i],
        # temp_max[i], temp_min[i] e precipitation[i] correspondem ao mesmo dia.
        # Itera sobre dates como índice mestre (o array mais confiável de estar presente).
        forecasts = [
            DailyForecast(
                date=DateType.fromisoformat(dates[i]),
                temp_max=temp_max[i] if i < len(temp_max) else 0.0,
                temp_min=temp_min[i] if i < len(temp_min) else 0.0,
                # "or 0.0" seria insuficiente aqui porque None é falsy mas 0.0 também.
                # O guard `if i < len(precipitation)` cobre o caso raro de a API retornar
                # o array de precipitação mais curto (ex: dados ainda não processados).
                # Na prática normal todos os arrays têm o mesmo comprimento (forecast_days).
                precipitation=precipitation[i] if i < len(precipitation) else 0.0,
            )
            for i in range(len(dates))
        ]

        return WeatherResponse(
            city=city_name,
            latitude=city_data["latitude"],
            longitude=city_data["longitude"],
            forecasts=forecasts,
        )
