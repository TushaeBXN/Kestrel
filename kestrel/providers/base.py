"""Neutral message and completion types. Every provider adapter speaks this."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generator, Literal, Union


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class TextChunk:
    text: str


@dataclass
class ThinkingChunk:
    text: str


@dataclass
class TurnDone:
    message: Message
    input_tokens: int = 0
    output_tokens: int = 0


StreamEvent = Union[TextChunk, ThinkingChunk, TurnDone]


class Provider:
    name: str = "base"

    def stream(
        self,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[Tool],
        *,
        temperature: float = 0.2,
    ) -> Generator[StreamEvent, None, None]:
        raise NotImplementedError
