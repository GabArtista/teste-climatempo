"""HTTP request/response schemas for the chat endpoint."""
from pydantic import BaseModel, Field, field_validator

from app.Models.ChatMessage import ChatMessage


class ChatRequestSchema(BaseModel):
    """Validated incoming chat request."""

    message: str = Field(
        min_length=1,
        max_length=1000,
        description="User's message to the agent",
        examples=["Como está o tempo em São Paulo nos próximos 3 dias?"],
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns for multi-turn support",
    )

    @field_validator("message")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        """Remove surrounding whitespace and reject blank messages."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message must not be blank")
        return stripped


class WeatherQueryParams(BaseModel):
    """Query parameters for the direct weather endpoint."""

    city: str = Field(
        min_length=1,
        description="Brazilian state capital name",
        examples=["São Paulo", "Curitiba"],
    )
    days: int = Field(
        default=3,
        ge=1,
        le=7,
        description="Number of forecast days (1–7)",
    )
