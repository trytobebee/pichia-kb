"""LLM provider abstraction.

Public API:
    from kb_core.llm import get_llm, LLMBackend
    llm = get_llm("deepseek-chat")
    text = llm.chat([{"role": "user", "content": "hi"}])

Model id conventions:
    "gemini-*"        → Gemini       (uses GEMINI_API_KEY)
    "deepseek-*"      → DeepSeek     (uses DEEPSEEK_API_KEY)
    "qwen*"           → 阿里灵积      (uses DASHSCOPE_API_KEY)
    "doubao-*"        → 火山引擎       (uses ARK_API_KEY)
    "gpt-*" / "o1-*"  → OpenAI direct (uses OPENAI_API_KEY)

You can force a provider by prefix: "openai/gpt-4o-mini",
"qwen/qwen-vl-max", "doubao/doubao-1.5-vision-pro".
"""

from __future__ import annotations

from .base import LLMBackend
from .gemini import GeminiBackend
from .openai_compat import OpenAIBackend


_OAI_COMPAT_FORCED = {"deepseek", "openai", "qwen", "doubao"}
_OAI_COMPAT_PREFIXES = ("deepseek", "qwen", "doubao", "gpt", "o1", "o3", "o4")


def _split_provider_prefix(model: str) -> tuple[str | None, str]:
    if "/" in model:
        provider, _, name = model.partition("/")
        return provider.lower(), name
    return None, model


def get_llm(model: str, **kwargs) -> LLMBackend:
    """Return an LLMBackend for the given model id.

    Extra kwargs are forwarded to the backend constructor (api_key,
    base_url, request_timeout_*, etc.).
    """
    forced_provider, name = _split_provider_prefix(model)

    if forced_provider == "gemini" or name.startswith("gemini"):
        return GeminiBackend(name, **kwargs)
    if forced_provider in _OAI_COMPAT_FORCED or name.startswith(_OAI_COMPAT_PREFIXES):
        return OpenAIBackend(name, provider=forced_provider, **kwargs)

    # Default: assume OpenAI-compatible (caller can override base_url)
    return OpenAIBackend(name, provider=forced_provider or "openai", **kwargs)


__all__ = ["LLMBackend", "GeminiBackend", "OpenAIBackend", "get_llm"]
