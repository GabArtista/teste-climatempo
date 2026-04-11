"""Unit tests for WeatherTool OpenAI format compliance."""
from app.Tools.WeatherTool import WEATHER_TOOL


def test_tool_has_type_function():
    assert WEATHER_TOOL["type"] == "function"


def test_tool_has_function_key():
    assert "function" in WEATHER_TOOL


def test_tool_has_name():
    assert WEATHER_TOOL["function"]["name"] == "get_weather_forecast"


def test_tool_has_description():
    desc = WEATHER_TOOL["function"]["description"]
    assert isinstance(desc, str) and len(desc) > 10


def test_tool_parameters_are_object():
    params = WEATHER_TOOL["function"]["parameters"]
    assert params["type"] == "object"


def test_tool_city_is_required():
    required = WEATHER_TOOL["function"]["parameters"]["required"]
    assert "city" in required


def test_tool_city_property_exists():
    props = WEATHER_TOOL["function"]["parameters"]["properties"]
    assert "city" in props
    assert props["city"]["type"] == "string"


def test_tool_forecast_days_has_default():
    props = WEATHER_TOOL["function"]["parameters"]["properties"]
    assert "forecast_days" in props
    assert props["forecast_days"]["default"] == 3
