"""
Modelos de domínio para os dados de previsão do tempo.

Define as estruturas que representam os dados climáticos retornados pela API
Open-Meteo: `DailyForecast` para um único dia e `WeatherResponse` para o conjunto
completo de dias de uma cidade.

Ambos os modelos incluem métodos `to_text()` que convertem os dados para texto
legível pelo LLM, permitindo que o agente incorpore as previsões diretamente em
suas respostas sem precisar formatar strings manualmente.
"""
from datetime import date as DateType
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class DailyForecast(BaseModel):
    """
    [Bloco 1 — Humanizado]
    Armazena os dados climáticos de um único dia: temperatura máxima, mínima e
    precipitação total. É a unidade básica de previsão — uma lista desses objetos
    compõe a previsão completa de uma cidade.

    [Bloco 2 — Técnico]
    Campos mapeados diretamente das variáveis do Open-Meteo:
        - `temperature_2m_max` → `temp_max` (°C)
        - `temperature_2m_min` → `temp_min` (°C)
        - `precipitation_sum`  → `precipitation` (mm)
    O campo `date` usa `datetime.date` (não `datetime`) — apenas ano/mês/dia,
    sem horário, correspondendo ao formato ISO 8601 retornado pela API (YYYY-MM-DD).

    Campos:
        date: Data da previsão (YYYY-MM-DD).
        temp_max: Temperatura máxima do dia em graus Celsius.
        temp_min: Temperatura mínima do dia em graus Celsius.
        precipitation: Precipitação total do dia em milímetros.
    """

    date: DateType = Field(description="Forecast date")
    temp_max: float = Field(description="Maximum temperature in °C")
    temp_min: float = Field(description="Minimum temperature in °C")
    precipitation: float = Field(description="Total precipitation in mm")

    def to_text(self) -> str:
        """
        [Bloco 1 — Humanizado]
        Converte os dados do dia em uma linha de texto legível, no formato que o
        agente LLM usará ao formular sua resposta ao usuário. Exibe a data no padrão
        brasileiro (dd/mm/aaaa) e indica "sem chuva" quando a precipitação é zero.

        [Bloco 2 — Técnico]
        Formata a data com `strftime('%d/%m/%Y')`. A precipitação é exibida com uma
        casa decimal apenas quando `self.precipitation > 0`; caso contrário, exibe
        "sem chuva" para melhor leitura humana e do LLM. Retorna uma string única
        sem quebra de linha.

        Returns:
            String formatada com data, temperatura máxima, mínima e precipitação.
            Exemplo: "📅 15/04/2025: máx 28.5°C, mín 18.2°C, 12.3mm chuva"
        """
        rain = f"{self.precipitation:.1f}mm chuva" if self.precipitation > 0 else "sem chuva"
        return (
            f"📅 {self.date.strftime('%d/%m/%Y')}: "
            f"máx {self.temp_max:.1f}°C, mín {self.temp_min:.1f}°C, {rain}"
        )


class WeatherResponse(BaseModel):
    """
    [Bloco 1 — Humanizado]
    Representa a previsão do tempo completa para uma cidade, com todos os dias
    solicitados e metadados de localização. É o objeto retornado ao frontend pelo
    endpoint de clima e passado ao agente LLM como resultado da tool call.

    [Bloco 2 — Técnico]
    Agrega uma lista de `DailyForecast` e os metadados da cidade (nome, coordenadas).
    O campo `generated_at` é preenchido automaticamente com `datetime.now(timezone.utc)`
    no momento da criação, garantindo rastreabilidade temporal sem depender do
    cliente. Usado como resposta do endpoint `GET /api/v1/weather/` e como payload
    intermediário no fluxo agêntico.

    Campos:
        city: Nome da capital consultada.
        latitude: Latitude geográfica da cidade.
        longitude: Longitude geográfica da cidade.
        forecasts: Lista de previsões diárias ordenadas por data crescente.
        generated_at: Timestamp UTC de geração da resposta.
    """

    city: str = Field(description="City name")
    latitude: float = Field(description="City latitude")
    longitude: float = Field(description="City longitude")
    forecasts: list[DailyForecast] = Field(description="Daily forecasts")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of API response",
    )

    def to_text(self) -> str:
        """
        [Bloco 1 — Humanizado]
        Converte toda a previsão da cidade em um bloco de texto formatado, pronto
        para ser incluído no contexto do LLM. O agente usa esse texto para formular
        sua resposta em linguagem natural ao usuário.

        [Bloco 2 — Técnico]
        Monta uma lista de strings: a primeira linha é o cabeçalho com o nome da
        cidade; as linhas seguintes são os textos de cada `DailyForecast.to_text()`
        com indentação de dois espaços. As linhas são unidas com `"\n".join()`.
        O resultado é uma string multi-linha sem trailing newline.

        Returns:
            String multi-linha com cabeçalho e uma linha por dia de previsão.
            Exemplo:
                "Previsão do tempo para São Paulo:
                  📅 15/04/2025: máx 25.0°C, mín 15.0°C, sem chuva
                  📅 16/04/2025: máx 22.0°C, mín 14.0°C, 5.2mm chuva"
        """
        lines = [f"🌤️ Previsão do tempo para {self.city}:\n"]
        for forecast in self.forecasts:
            lines.append(f"  {forecast.to_text()}")
        return "\n".join(lines)
