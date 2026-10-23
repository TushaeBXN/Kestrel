"""The Kestrel agent loop. Plans, calls tools, iterates until done."""
from __future__ import annotations
import json
import threading
from typing import Callable

from .providers.base import Message, TextChunk, TurnDone
from .tools import TOOLS, EXPERT_TOOL, NEEDS_APPROVAL, ToolRunner
from .workspace import Workspace

SYSTEM_PROMPT = """You are Kestrel, a coding agent operating in a project workspace (a git repository).

You have tools to read, write and edit files, list and search files, run shell
commands, and inspect/commit/push with git. Use them to accomplish the user's goal.

Approach:
1. If the goal is ambiguous, inspect the codebase first (list_files, read_file, grep).
2. Plan the minimal change that achieves the goal.
3. Use write_file to create new files, edit_file to modify existing ones.
4. Prefer small, verifiable edits over large rewrites. Run tests with run_shell when they exist.
5. You are a small, fast model. If a task is genuinely hard and you have a tool called ask_expert, use it sparingly.
6. Only commit or push when the user asks you to. Write clear commit messages.
7. When done, state briefly what changed.

Be precise. Don't invent file paths — check first with list_files or read_file.
"""

MAX_TURNS = 30
INSTRUCTION_FILES = ("KESTREL.md", "AGENTS.md")


def build_system(ws: Workspace) -> str:
    extra = ""
    for name in INSTRUCTION_FILES:
        p = ws.root / name
        if p.is_file():
            extra += f"\n\nProject instructions ({name}):\n{p.read_text(encoding='utf-8', errors='replace')[:8000]}"
    return SYSTEM_PROMPT + extra


def run(
    goal: str,
    model: str,
    provider,
    emit: Callable[[dict], None],
    ws: Workspace,
    history: list[Message] | None = None,
    approve: Callable[[object, str], bool] | None = None,
    cancel: threading.Event | None = None,
    expert: Callable[[str], str] | None = None,
) -> None:
    """Run one user turn. `history` is mutated in place so chats can continue."""
    messages = history if history is not None else []
    messages.append(Message(role="user", content=goal))
    runner = ToolRunner(ws, expert)
    tools = TOOLS + ([EXPERT_TOOL] if expert else [])
    system = build_system(ws)

    for turn in range(MAX_TURNS):
        if cancel and cancel.is_set():
            emit({"type": "error", "message": "stopped"})
            return
        emit({"type": "turn", "n": turn + 1})

        assistant_msg: Message | None = None
        for ev in provider.stream(model=model, system=system, messages=messages, tools=tools):
            if isinstance(ev, TextChunk):
                emit({"type": "text", "text": ev.text})
            elif isinstance(ev, TurnDone):
                assistant_msg = ev.message
                if ev.input_tokens or ev.output_tokens:
                    emit({"type": "usage", "input": ev.input_tokens, "output": ev.output_tokens})

        if assistant_msg is None:
            emit({"type": "error", "message": "provider returned no message"})
            return

        messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            emit({"type": "done", "success": True})
            return

        for tc in assistant_msg.tool_calls:
            emit({"type": "tool_call", "id": tc.id, "name": tc.name, "input": tc.input})
            if tc.name in NEEDS_APPROVAL and approve is not None:
                detail = runner.describe(tc.name, tc.input)
                emit({"type": "approval", "id": tc.id, "name": tc.name, "detail": detail})
                if not approve(tc, detail):
                    result = {"error": "the user denied this action"}
                else:
                    result = runner.dispatch(tc.name, tc.input)
            else:
                result = runner.dispatch(tc.name, tc.input)
            emit({"type": "tool_result", "id": tc.id, "name": tc.name, "result": result})
            messages.append(Message(
                role="tool",
                content=json.dumps(result, ensure_ascii=False)[:30000],
                tool_call_id=tc.id,
                name=tc.name,
            ))

    emit({"type": "error", "message": f"hit MAX_TURNS ({MAX_TURNS}) without finishing"})
