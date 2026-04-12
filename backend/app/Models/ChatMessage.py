"""Domain models for chat interactions."""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Allowed roles in a chat conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    tool_call_id: str | None = Field(default=None, description="ID for tool result messages")
    name: str | None = Field(default=None, description="Tool name for tool result messages")


class ChatRequest(BaseModel):
    """Incoming chat request from the user."""

    message: str = Field(min_length=1, max_length=1000, description="User message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be blank")
        return v.strip()


class ChatResponse(BaseModel):
    """Response from the agent."""

    response: str = Field(description="Agent's text response")
    tool_called: bool = Field(description="Whether the weather tool was invoked")
    city_queried: str | None = Field(
        default=None,
        description="City name passed to weather tool, if called",
    )
    reason: Literal["success", "no_city", "non_capital", "non_weather"] | None = Field(
        default=None,
        description=(
            "Why the tool was or was not called: "
            "'success' = tool called and returned data, "
            "'no_city' = weather intent but no city mentioned, "
            "'non_capital' = city mentioned but not a Brazilian state capital, "
            "'non_weather' = query not related to weather"
        ),
    )
