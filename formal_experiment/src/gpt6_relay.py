"""Reusable GPT-6 relay adapter for this workspace.

This module deliberately lives beside :mod:`deepseek_flash` and uses the same
workspace credential file, ``.env.llm``. Add ``GPT6_RELAY_API_KEY`` to
that file; the key is never embedded in this module.

Typical use::

    from gpt6_relay import GPT6Relay

    with GPT6Relay.from_env_file(".env.llm") as gpt6:
        result = gpt6.ask("You are concise.", "Summarize this text.")
        print(result.content)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from deepseek_flash import load_env_file


DEFAULT_BASE_URL = "https://sub.ai8888.shop/v1"
DEFAULT_MODEL = "gpt-6"


@dataclass(frozen=True)
class GPT6Result:
    """Visible completion data returned by :class:`GPT6Relay`."""

    content: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    model: str


class GPT6Relay:
    """Small OpenAI-compatible Chat Completions client for the GPT-6 relay."""

    def __init__(self, *, api_key: str, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, timeout_seconds: float = 180.0,
                 client: Any | None = None) -> None:
        if not api_key:
            raise ValueError("A GPT-6 relay API key is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._http_client: Any | None = None
        if client is None:
            import httpx
            from openai import OpenAI

            self._http_client = httpx.Client(timeout=timeout_seconds, trust_env=False)
            self._client = OpenAI(api_key=api_key, base_url=self.base_url, http_client=self._http_client)
        else:
            self._client = client

    @classmethod
    def from_env_file(cls, path: str | Path = ".env.llm", **kwargs: Any) -> "GPT6Relay":
        """Create a client from ``GPT6_RELAY_API_KEY`` in the shared env file."""
        load_env_file(path)
        api_key = os.environ.get("GPT6_RELAY_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Credential file must contain GPT6_RELAY_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("GPT6_RELAY_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("GPT6_RELAY_MODEL", DEFAULT_MODEL),
            timeout_seconds=float(os.environ.get("GPT6_RELAY_TIMEOUT", "180")),
            **kwargs,
        )

    def ask(self, system: str, user: str, *, max_output_tokens: int = 16_384,
            temperature: float | None = None, json_object: bool = False,
            **extra: Any) -> GPT6Result:
        """Send one system/user prompt pair through the relay."""
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            json_object=json_object,
            **extra,
        )

    def complete(self, messages: Sequence[Mapping[str, Any]], *,
                 max_output_tokens: int = 16_384, temperature: float | None = None,
                 json_object: bool = False, **extra: Any) -> GPT6Result:
        """Complete OpenAI-format messages and return the visible answer."""
        if not messages:
            raise ValueError("messages must not be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_output_tokens,
            **extra,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if json_object:
            request["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**request)
        if not getattr(response, "choices", None):
            raise RuntimeError("GPT-6 relay returned no choices")
        choice = response.choices[0]
        content = getattr(choice.message, "content", None) or ""
        if not content.strip():
            raise RuntimeError(
                "GPT-6 relay returned empty final content "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r})"
            )
        usage = getattr(response, "usage", None)
        return GPT6Result(
            content=content,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            model=getattr(response, "model", None) or self.model,
        )

    def ask_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Request one JSON object and decode the returned content."""
        return json.loads(self.ask(system, user, json_object=True, **kwargs).content)

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> "GPT6Relay":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
