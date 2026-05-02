"""Gemini backend (google.genai SDK)."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from google import genai
from google.genai import types

from .base import LLMBackend


def _to_gemini_messages(messages: list[dict]) -> list[types.Content]:
    """Convert {role, content} dicts to Gemini Content objects.

    Maps OpenAI-style roles ('user' | 'assistant') to Gemini ('user' | 'model').
    """
    out: list[types.Content] = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        out.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return out


class GeminiBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        request_timeout_ms: int = 600_000,
    ) -> None:
        self._model = model
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=request_timeout_ms),
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "gemini"

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = self.client.models.generate_content(
            model=self._model,
            contents=_to_gemini_messages(messages),
            config=cfg,
        )
        return response.text or ""

    def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        for chunk in self.client.models.generate_content_stream(
            model=self._model,
            contents=_to_gemini_messages(messages),
            config=cfg,
        ):
            if chunk.text:
                yield chunk.text

    def chat_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=cfg,
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
