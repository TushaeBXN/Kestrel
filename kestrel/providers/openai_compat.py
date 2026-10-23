"""OpenAI-compatible adapter. Covers OpenAI, Groq, Together, Qwen, Kimi,
Zhipu, DeepSeek, Ollama, LM Studio, vLLM, and any HF-hosted endpoint that
speaks the OpenAI chat-completions API."""
from __future__ import annotations
import json
from typing import Generator

from openai import OpenAI
from .base import (
    Message, Tool, ToolCall, Provider, TextChunk, TurnDone, StreamEvent,
)


def _messages_to_openai(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": m.content,
            })
    return out


def _tools_to_openai(tools: list[Tool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client = OpenAI(api_key=api_key or "kestrel", base_url=base_url)

    def stream(
        self,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[Tool],
        *,
        temperature: float = 0.2,
    ) -> Generator[StreamEvent, None, None]:
        oai_messages = [{"role": "system", "content": system}] + _messages_to_openai(messages)

        kwargs: dict = {
            "model": model or self.model,
            "messages": oai_messages,
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        text = ""
        tool_buf: dict[int, dict] = {}
        in_tok = out_tok = 0

        for chunk in self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                if getattr(chunk, "usage", None):
                    in_tok = chunk.usage.prompt_tokens or in_tok
                    out_tok = chunk.usage.completion_tokens or out_tok
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                text += delta.content
                yield TextChunk(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_buf:
                        tool_buf[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_buf[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_buf[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_buf[idx]["args"] += tc.function.arguments

            if getattr(chunk, "usage", None):
                in_tok = chunk.usage.prompt_tokens or in_tok
                out_tok = chunk.usage.completion_tokens or out_tok

        tool_calls = []
        for idx in sorted(tool_buf):
            v = tool_buf[idx]
            try:
                parsed = json.loads(v["args"]) if v["args"] else {}
            except json.JSONDecodeError:
                parsed = {"_raw": v["args"]}
            tool_calls.append(ToolCall(id=v["id"] or f"call_{idx}", name=v["name"], input=parsed))

        yield TurnDone(
            message=Message(role="assistant", content=text, tool_calls=tool_calls),
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
