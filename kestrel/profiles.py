"""Model profiles: named endpoints so you can switch between local models
(Qwen3.5-4B, LFM2.5, Amy, Anthos, Kerrigan-Fantasma, ...) and a paid expert.

Config lives in $KESTREL_HOME/models.json (default ~/.kestrel/models.json):

{
  "default": "local",          # profile used for normal work (cheap/local)
  "expert":  "paid",           # profile used by ask_expert and diff review
  "profiles": {
    "local": {"base_url": "http://localhost:8080/v1", "model": "qwen3.5-4b-agentic-coding"},
    "amy":   {"base_url": "http://localhost:8081/v1", "model": "amy"},
    "paid":  {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-5-5",
              "api_key_env": "ANTHROPIC_API_KEY"}
  }
}

Any server that speaks the OpenAI chat-completions API works as a profile.
"""
from __future__ import annotations
import json, os
from .providers import resolve
from .providers.base import Provider, Message, TextChunk, TurnDone
from .workspace import kestrel_home

EXAMPLE = {
    "default": "local",
    "expert": "paid",
    "profiles": {
        "local": {"base_url": "http://localhost:8080/v1", "model": "qwen3.5-4b-agentic-coding"},
        "paid": {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-5-5",
                 "api_key_env": "ANTHROPIC_API_KEY"},
    },
}


def path():
    return kestrel_home() / "models.json"


def load() -> dict:
    p = path()
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("profiles", {})
        return data
    # No file: fall back to env vars so the original quick-start still works.
    env_model = os.environ.get("KESTREL_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    return {"default": "env", "expert": None,
            "profiles": {"env": {"base_url": os.environ.get("KESTREL_BASE_URL"), "model": env_model}}}


def write_example() -> str:
    p = path()
    if p.exists():
        return f"{p} already exists"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(EXAMPLE, indent=2) + "\n", encoding="utf-8")
    return f"wrote {p}"


def get(name: str | None, cfg: dict | None = None) -> tuple[str, Provider]:
    """Return (model_id, provider) for a profile name (None = default)."""
    cfg = cfg or load()
    name = name or cfg.get("default")
    prof = cfg["profiles"].get(name)
    if not prof:
        raise KeyError(f"unknown model profile: {name}")
    key = os.environ.get(prof["api_key_env"]) if prof.get("api_key_env") else prof.get("api_key")
    return prof["model"], resolve(prof["model"], base_url=prof.get("base_url"), api_key=key)


def expert_fn(cfg: dict | None = None):
    """Build the callable behind the ask_expert tool, or None if unconfigured."""
    cfg = cfg or load()
    name = cfg.get("expert")
    if not name or name not in cfg["profiles"]:
        return None

    def ask(prompt: str) -> str:
        model, provider = get(name, cfg)
        system = ("You are a senior engineer consulted by a smaller coding agent. "
                  "Answer precisely and concisely; give concrete code or steps.")
        text = ""
        for ev in provider.stream(model=model, system=system,
                                  messages=[Message(role="user", content=prompt)], tools=[]):
            if isinstance(ev, TextChunk):
                text += ev.text
        return text

    return ask
