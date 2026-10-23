"""Provider registry and factory."""
from __future__ import annotations
import os
from .base import Provider
from .openai_compat import OpenAICompatProvider

_KNOWN: dict[str, str] = {
    "openai":   "https://api.openai.com/v1",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai/",
    "kimi":     "https://api.moonshot.cn/v1",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu":    "https://open.bigmodel.cn/api/paas/v4/",
    "deepseek": "https://api.deepseek.com/v1",
    "groq":     "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "hf":       "https://router.huggingface.co/v1",
    "ollama":   "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
}

_KEY_ENV: dict[str, str] = {
    "openai":   "OPENAI_API_KEY",
    "gemini":   "GEMINI_API_KEY",
    "kimi":     "MOONSHOT_API_KEY",
    "qwen":     "DASHSCOPE_API_KEY",
    "zhipu":    "ZHIPU_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq":     "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "hf":       "HF_TOKEN",
}

_PREFIXES = [
    ("gpt-",      "openai"),
    ("o1",        "openai"),
    ("o3",        "openai"),
    ("gemini-",   "gemini"),
    ("moonshot-", "kimi"),
    ("kimi-",     "kimi"),
    ("qwen",      "qwen"),
    ("qwq-",      "qwen"),
    ("glm-",      "zhipu"),
    ("deepseek-", "deepseek"),
]


def detect_provider(model: str) -> str:
    head = model.split("/", 1)[0].lower()
    if "/" in model and head in _KNOWN:
        return head
    lower = model.lower()
    for prefix, name in _PREFIXES:
        if lower.startswith(prefix):
            return name
    return "openai"


def bare_model(model: str) -> str:
    """Strip a leading `provider/` only when it names a known provider.
    HF-style ids like `Qwen/Qwen2.5-Coder-7B-Instruct` are left intact."""
    head = model.split("/", 1)[0].lower()
    return model.split("/", 1)[1] if "/" in model and head in _KNOWN else model


def resolve(model: str, base_url: str | None = None, api_key: str | None = None) -> Provider:
    custom_url = base_url or os.environ.get("KESTREL_BASE_URL")
    if custom_url:
        return OpenAICompatProvider(
            base_url=custom_url,
            api_key=api_key or os.environ.get("KESTREL_API_KEY", "kestrel"),
            model=bare_model(model),
        )

    provider_name = detect_provider(model)
    url = _KNOWN.get(provider_name)
    if not url:
        raise RuntimeError(
            f"Unknown provider '{provider_name}' for model '{model}'. "
            f"Set KESTREL_BASE_URL to a custom endpoint."
        )

    key_env = _KEY_ENV.get(provider_name, "OPENAI_API_KEY")
    return OpenAICompatProvider(
        base_url=url,
        api_key=api_key or os.environ.get(key_env, ""),
        model=bare_model(model),
    )
