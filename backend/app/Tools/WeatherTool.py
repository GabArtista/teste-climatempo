"""OpenAI-compatible weather tool definition and executor."""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# OpenAI-compatible tool definition (required by challenge spec)
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
    Execute the weather tool with given arguments.

    Args:
        arguments: JSON string or dict with 'city' and optional 'forecast_days'.
        weather_service: WeatherService instance to fetch forecast data.

    Returns:
        JSON string with weather data or error message.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        city: str = args.get("city", "")
        forecast_days: int = int(args.get("forecast_days", 3))
        forecast_days = max(1, min(7, forecast_days))  # clamp to valid range

        if not city:
            return json.dumps({"error": "City name is required"})

        logger.info("Executing weather tool: city=%s, days=%d", city, forecast_days)
        result = await weather_service.get_forecast(city=city, forecast_days=forecast_days)
        return result.model_dump_json()

    except Exception as exc:
        logger.error("Weather tool execution failed: %s", exc)
        return json.dumps({"error": str(exc)})
