"""OpenAI-compatible backend.

Works for any service that implements the OpenAI Chat Completions API
shape: OpenAI direct, DeepSeek, Mistral, Together, OpenRouter, vLLM, etc.

The provider is detected from the model id (or explicit base_url):
  - 'deepseek-*' or model_id with provider='deepseek' → api.deepseek.com
  - 'gpt-*' / 'o*'                                    → api.openai.com (default)
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

from .base import LLMBackend


# (env_var_name, base_url) pairs. Add new providers here.
_PROVIDER_DEFAULTS = {
    "deepseek": ("DEEPSEEK_API_KEY",  "https://api.deepseek.com/v1"),
    "openai":   ("OPENAI_API_KEY",    "https://api.openai.com/v1"),
    # 阿里灵积(Dashscope)兼容模式 — covers Qwen / Qwen-VL family.
    "qwen":     ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    # 火山引擎(Volcengine ARK)— covers Doubao chat + vision.
    "doubao":   ("ARK_API_KEY",       "https://ark.cn-beijing.volces.com/api/v3"),
}


def _detect_provider(model: str) -> str:
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("qwen") or model.startswith("qwen2") or model.startswith("qwen3"):
        return "qwen"
    if model.startswith("doubao"):
        return "doubao"
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
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=self._messages(messages, system),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        merged_extra = {**self._default_extra_body(), **(extra_body or {})}
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def stream_chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=self._messages(messages, system),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        merged_extra = {**self._default_extra_body(), **(extra_body or {})}
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        stream = self.client.chat.completions.create(**kwargs)
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
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # response_format: {type: json_object} is the OpenAI/DeepSeek
        # equivalent of Gemini's response_mime_type='application/json'.
        # The system prompt MUST mention "json" or DeepSeek rejects.
        sys = (system or "").rstrip()
        if "json" not in sys.lower():
            sys = f"{sys}\n\nReturn ONLY valid JSON.".strip()
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=self._messages([{"role": "user", "content": prompt}], sys),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        merged_extra = {**self._default_extra_body(), **(extra_body or {})}
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        resp = self.client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)

    def _default_extra_body(self) -> dict[str, Any]:
        """Provider-specific defaults applied to every call unless caller
        overrides via the explicit `extra_body` argument.

        Currently only qwen — qwen3.6 reasoning models default to
        thinking-on, which makes simple structured-extraction calls 10x
        slower with no quality gain. We turn it off so the framework's
        extractor / vision calls finish quickly. Users who want reasoning
        (e.g. for cross-paper synthesis) can pass enable_thinking=True
        explicitly via the `extra_body` arg.
        """
        if self._provider == "qwen":
            return {"enable_thinking": False}
        return {}

    def chat_vision_json(
        self,
        prompt: str,
        image_path: Path,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an image + text prompt to a vision-capable OpenAI-compatible
        model. Used for Qwen-VL, Doubao-vision, GPT-4o, etc.

        Note: not all providers under this backend support vision. DeepSeek
        as of writing does not (server returns "[Unsupported Image]").
        Caller should pick a vision model id (qwen-vl-*, doubao-*-vision-*,
        gpt-4o, ...).
        """
        mime, _ = mimetypes.guess_type(str(image_path))
        mime = mime or "image/png"
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        sys = (system or "").rstrip()
        if "json" not in sys.lower():
            sys = f"{sys}\n\nReturn ONLY valid JSON.".strip()
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]
        merged_extra = {**self._default_extra_body(), **(extra_body or {})}
        # Some OAI-compat servers (Qwen) don't accept response_format on
        # vision endpoints. Try with json_object first, fall back without.
        common_kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=self._messages([{"role": "user", "content": user_content}], sys),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if merged_extra:
            common_kwargs["extra_body"] = merged_extra
        try:
            resp = self.client.chat.completions.create(
                **common_kwargs, response_format={"type": "json_object"},
            )
        except Exception:
            resp = self.client.chat.completions.create(**common_kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
