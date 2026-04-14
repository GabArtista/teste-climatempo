"""
Modelos de domínio para as interações de chat do agente conversacional.

Define as estruturas de dados que trafegam entre o frontend, os controllers e o
AgentService: a mensagem individual (`ChatMessage`), a requisição de chat
(`ChatRequest`) e a resposta do agente (`ChatResponse`).

Todos os modelos são Pydantic BaseModel — validação automática na desserialização
e serialização automática nas respostas HTTP do FastAPI.
"""
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
    """
    [Bloco 1 — Humanizado]
    Representa uma única mensagem dentro de uma conversa, seja ela do usuário,
    do assistente, do sistema ou o resultado de uma tool call. É a unidade básica
    do histórico de conversas enviado ao LLM.

    [Bloco 2 — Técnico]
    Segue o schema de mensagens da API OpenAI (compatível com Ollama). O campo
    `role` é restrito ao enum `MessageRole` (user, assistant, tool, system).
    `tool_call_id` e `name` são opcionais e usados exclusivamente em mensagens
    de resultado de tool call (role == "tool"), onde `tool_call_id` vincula o
    resultado à chamada original e `name` identifica qual ferramenta foi executada.

    Campos:
        role: Papel do autor da mensagem. Valores: user, assistant, tool, system.
        content: Conteúdo textual da mensagem.
        tool_call_id: ID da tool call correspondente (apenas para role='tool').
        name: Nome da ferramenta executada (apenas para role='tool').
    """

    role: MessageRole
    content: str
    tool_call_id: str | None = Field(default=None, description="ID for tool result messages")
    name: str | None = Field(default=None, description="Tool name for tool result messages")


class ChatRequest(BaseModel):
    """
    [Bloco 1 — Humanizado]
    Representa a requisição enviada pelo frontend ao endpoint de chat. Contém a
    mensagem atual do usuário e, opcionalmente, o histórico de mensagens anteriores
    para que o agente mantenha o contexto da conversa.

    [Bloco 2 — Técnico]
    Desserializado automaticamente pelo FastAPI a partir do corpo JSON da requisição
    POST /api/v1/agent/chat. O campo `message` é validado com `min_length=1` e
    `max_length=1000` pelo Field, e o validator `message_not_blank` rejeita strings
    compostas apenas de espaços/tabs/newlines, retornando HTTP 422 com detalhes da
    validação. O `history` padrão é lista vazia (conversa sem contexto anterior).

    Campos:
        message: Mensagem atual do usuário (1–1000 caracteres, não pode ser em branco).
        history: Histórico de mensagens anteriores para contexto multi-turno.
    """

    message: str = Field(min_length=1, max_length=1000, description="User message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        """
        [Bloco 1 — Humanizado]
        Garante que a mensagem do usuário tenha conteúdo real, não apenas espaços
        em branco. Também remove espaços extras no início e no fim antes de
        processar a mensagem.

        [Bloco 2 — Técnico]
        Validator Pydantic v2 (`@field_validator`). Executado após a validação de
        tipo e comprimento do `Field`. Chama `v.strip()` e rejeita o valor se o
        resultado for falsy. Retorna a string com `.strip()` aplicado, garantindo
        que o serviço nunca receba mensagem com whitespace desnecessário.

        Args:
            v: Valor do campo `message` após validação de tipo.

        Returns:
            String sem espaços extras no início e no fim.

        Raises:
            ValueError: Se a mensagem contiver apenas espaços em branco.
        """
        if not v.strip():
            raise ValueError("Message cannot be blank")
        return v.strip()


class ChatResponse(BaseModel):
    """
    [Bloco 1 — Humanizado]
    Resposta completa do agente ao usuário: inclui o texto da resposta e metadados
    que indicam se o agente consultou a API de clima, qual cidade foi consultada e
    por que a tool call foi ou não executada. O campo `reason` permite que o frontend
    diferencie cenários de sucesso e tipos de falha sem parsear o texto da resposta.

    [Bloco 2 — Técnico]
    Serializado automaticamente pelo FastAPI como JSON na resposta HTTP. O campo
    `reason` é um Literal union com quatro valores possíveis:
        - "success"     : tool de clima chamada com sucesso e dados retornados.
        - "no_city"     : intenção de clima detectada, mas nenhuma cidade mencionada.
        - "non_capital" : cidade mencionada, mas não é capital de estado brasileiro.
        - "non_weather" : mensagem não relacionada a previsão do tempo.
    `reason` pode ser `None` em casos de erro não mapeado. `city_queried` é `None`
    quando a tool não foi invocada.

    Campos:
        response: Texto da resposta gerada pelo agente.
        tool_called: Indica se a ferramenta de previsão do tempo foi invocada.
        city_queried: Nome da cidade passada à ferramenta, se houver.
        reason: Motivo do resultado — ver valores possíveis acima.
    """

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
