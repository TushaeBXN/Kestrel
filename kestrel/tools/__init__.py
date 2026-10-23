"""Kestrel's tool set. Every tool is sandboxed to one Workspace."""
from __future__ import annotations
import re, subprocess
from typing import Any

from ..providers.base import Tool
from ..workspace import Workspace, GitError

# Tools that change the outside world or run arbitrary code: the loop asks the
# user before running these (unless auto-approve is on).
NEEDS_APPROVAL = {"run_shell", "git_commit", "git_push", "ask_expert"}

MAX_OUT = 12000


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TOOLS: list[Tool] = [
    Tool("read_file", "Read the full contents of a text file. Paths are relative to the workspace root.",
         _obj({"path": {"type": "string"}}, ["path"])),
    Tool("write_file", "Create or overwrite a file with the given content.",
         _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])),
    Tool("edit_file", "Replace the first occurrence of `old` with `new` in a file. `old` must match exactly.",
         _obj({"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
              ["path", "old", "new"])),
    Tool("list_files", "List files in a directory (recursive, up to 500 entries).",
         _obj({"path": {"type": "string", "default": "."}})),
    Tool("grep", "Search file contents with a regex. Returns matching lines as file:line.",
         _obj({"pattern": {"type": "string"}, "path": {"type": "string", "default": "."},
               "max": {"type": "integer", "default": 50}}, ["pattern"])),
    Tool("run_shell", "Run a shell command in the workspace root (60s timeout). Use for tests, builds, installs.",
         _obj({"command": {"type": "string"}}, ["command"])),
    Tool("git_status", "Show the current branch and changed files.", _obj({})),
    Tool("git_diff", "Show the uncommitted diff.", _obj({})),
    Tool("git_commit", "Stage all changes and commit with the given message.",
         _obj({"message": {"type": "string"}}, ["message"])),
    Tool("git_push", "Push the current branch to origin.", _obj({})),
]

# Only offered to the model when an expert profile is configured. It costs
# money, so the model is told to use it sparingly.
EXPERT_TOOL = Tool(
    "ask_expert",
    "Ask a stronger (paid) model for help. Use ONLY when stuck after real attempts, for a hard design "
    "decision, or to review a risky change. Include all needed context in `question`; the expert cannot "
    "see the workspace.",
    _obj({"question": {"type": "string"}}, ["question"]),
)


class ToolRunner:
    def __init__(self, ws: Workspace, expert=None):
        self.ws = ws
        self.expert = expert
        self._dispatch = {
            "read_file": self._read_file, "write_file": self._write_file,
            "edit_file": self._edit_file, "list_files": self._list_files,
            "grep": self._grep, "run_shell": self._run_shell,
            "git_status": self._git_status, "git_diff": self._git_diff,
            "git_commit": self._git_commit, "git_push": self._git_push,
            "ask_expert": self._ask_expert,
        }

    def describe(self, name: str, args: dict[str, Any]) -> str:
        """Human-readable summary of a call, shown in approval prompts."""
        if name == "run_shell":
            return str(args.get("command", ""))
        if name == "ask_expert":
            return f"consult the paid expert model:\n{str(args.get('question', ''))[:600]}"
        if name == "git_commit":
            return f"commit all changes: {args.get('message', '')}"
        if name == "git_push":
            return f"push {self.ws.branch()} to origin ({self.ws.remote_url() or 'no remote'})"
        return name

    def dispatch(self, name: str, args: dict[str, Any]) -> dict:
        fn = self._dispatch.get(name)
        if not fn:
            return {"error": f"unknown tool: {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            return {"error": str(e)}

    # ---- file tools --------------------------------------------------
    def _read_file(self, path: str) -> dict:
        text = self.ws.safe(path).read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": text[:MAX_OUT * 3], "truncated": len(text) > MAX_OUT * 3}

    def _write_file(self, path: str, content: str) -> dict:
        p = self.ws.safe(path)
        if ".git" in p.relative_to(self.ws.root).parts:
            return {"error": "refusing to write inside .git"}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    def _edit_file(self, path: str, old: str, new: str) -> dict:
        p = self.ws.safe(path)
        text = p.read_text(encoding="utf-8")
        if old not in text:
            return {"error": "old string not found", "path": path}
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"path": path, "replaced": True}

    def _list_files(self, path: str = ".") -> dict:
        base = self.ws.safe(path)
        out: list[str] = []
        for p in sorted(self.ws.iter_files(base)):
            out.append(str(p.relative_to(self.ws.root)))
            if len(out) >= 500:
                break
        return {"files": out, "truncated": len(out) >= 500}

    def _grep(self, pattern: str, path: str = ".", max: int = 50) -> dict:
        base = self.ws.safe(path)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {"error": f"bad regex: {e}"}
        files = [base] if base.is_file() else self.ws.iter_files(base)
        hits: list[str] = []
        for p in files:
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{p.relative_to(self.ws.root)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= max:
                            return {"matches": hits, "truncated": True}
            except (UnicodeDecodeError, OSError):
                continue
        return {"matches": hits, "truncated": False}

    # ---- shell -------------------------------------------------------
    def _run_shell(self, command: str) -> dict:
        try:
            r = subprocess.run(command, shell=True, cwd=self.ws.root, capture_output=True,
                               text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return {"error": "command timed out after 60s"}
        return {"exit_code": r.returncode, "stdout": r.stdout[-MAX_OUT:], "stderr": r.stderr[-MAX_OUT:]}

    # ---- git ---------------------------------------------------------
    def _git_status(self) -> dict:
        return self.ws.status()

    def _git_diff(self) -> dict:
        return {"diff": self.ws.diff(MAX_OUT * 2)}

    def _git_commit(self, message: str) -> dict:
        try:
            return {"commit": self.ws.commit(message)}
        except GitError as e:
            return {"error": str(e)}

    def _git_push(self) -> dict:
        try:
            return {"result": self.ws.push()}
        except GitError as e:
            return {"error": str(e)}

    def _ask_expert(self, question: str) -> dict:
        if not self.expert:
            return {"error": "no expert model configured"}
        return {"answer": self.expert(question)}
