"""OpenAI-compatible backend.

Works for any service that implements the OpenAI Chat Completions API
shape: OpenAI direct, DeepSeek, Mistral, Together, OpenRouter, vLLM, etc.

The provider is detected from the model id (or explicit base_url):
  - 'deepseek-*' or model_id with provider='deepseek' → api.deepseek.com
  - 'gpt-*' / 'o*'                                    → api.openai.com (default)
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from openai import OpenAI

from .base import LLMBackend


# (env_var_name, base_url) pairs
_PROVIDER_DEFAULTS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "openai":   ("OPENAI_API_KEY",   "https://api.openai.com/v1"),
}


def _detect_provider(model: str) -> str:
    if model.startswith("deepseek"):
        return "deepseek"
    return "openai"


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout_s: int = 600,
    ) -> None:
        self._model = model
        self._provider = provider or _detect_provider(model)
        env_var, default_base = _PROVIDER_DEFAULTS.get(
            self._provider, _PROVIDER_DEFAULTS["openai"],
        )
        self.client = OpenAI(
            api_key=api_key or os.environ[env_var],
            base_url=base_url or default_base,
            timeout=request_timeout_s,
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def _messages(
        self,
        messages: list[dict],
        system: str | None,
    ) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(messages)
        return out

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=self._messages(messages, system),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self._model,
            messages=self._messages(messages, system),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for event in stream:
            choice = event.choices[0] if event.choices else None
            if choice is None:
                continue
            delta = getattr(choice.delta, "content", None)
            if delta:
                yield delta

    def chat_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        # response_format: {type: json_object} is the OpenAI/DeepSeek
        # equivalent of Gemini's response_mime_type='application/json'.
        # The system prompt MUST mention "json" or DeepSeek rejects.
        sys = (system or "").rstrip()
        if "json" not in sys.lower():
            sys = f"{sys}\n\nReturn ONLY valid JSON.".strip()
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=self._messages([{"role": "user", "content": prompt}], sys),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
