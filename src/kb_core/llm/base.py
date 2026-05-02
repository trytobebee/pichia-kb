"""Backend-agnostic LLM client interface.

Two implementations live in this package: GeminiBackend and OpenAIBackend
(works for OpenAI direct, DeepSeek, and any OpenAI-compatible endpoint).

The factory in __init__.py picks one based on the model id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class LLMBackend(ABC):
    """Minimal capability surface every provider must implement.

    `messages` is a list of {"role": "user" | "assistant", "content": str}.
    `system` is the system prompt (string).
    Returned text is the model's reply (no role wrapper).
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        ...

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        ...

    @abstractmethod
    def chat_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """One-shot prompt → JSON response, parsed into a dict.

        Raises ValueError if the response can't be parsed.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """'gemini' | 'openai' | 'deepseek' | etc."""
        ...
