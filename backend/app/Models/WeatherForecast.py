"""Domain models for weather forecast data."""
from datetime import date as DateType
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class DailyForecast(BaseModel):
    """Weather data for a single day."""

    date: DateType = Field(description="Forecast date")
    temp_max: float = Field(description="Maximum temperature in °C")
    temp_min: float = Field(description="Minimum temperature in °C")
    precipitation: float = Field(description="Total precipitation in mm")

    def to_text(self) -> str:
        """Return human-readable forecast line."""
        rain = f"{self.precipitation:.1f}mm chuva" if self.precipitation > 0 else "sem chuva"
        return (
            f"📅 {self.date.strftime('%d/%m/%Y')}: "
            f"máx {self.temp_max:.1f}°C, mín {self.temp_min:.1f}°C, {rain}"
        )


class WeatherResponse(BaseModel):
    """Complete weather forecast for a city."""

    city: str = Field(description="City name")
    latitude: float = Field(description="City latitude")
    longitude: float = Field(description="City longitude")
    forecasts: list[DailyForecast] = Field(description="Daily forecasts")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of API response",
    )

    def to_text(self) -> str:
        """Return full formatted forecast as readable text."""
        lines = [f"🌤️ Previsão do tempo para {self.city}:\n"]
        for forecast in self.forecasts:
            lines.append(f"  {forecast.to_text()}")
        return "\n".join(lines)
