"""Reusable DeepSeek Flash adapter for this workspace.

Typical use::

    from deepseek_flash import DeepSeekFlash

    with DeepSeekFlash.from_env_file(".env.llm") as deepseek:
        result = deepseek.ask("You are concise.", "Summarize this text.")
        print(result.content)

The adapter keeps authentication, thinking-mode request shape, response parsing,
and usage extraction in one place.  It never writes chain-of-thought to disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 65_536
_EFFORTS = frozenset({"low", "high", "max"})


@dataclass(frozen=True)
class DeepSeekResult:
    """Final response data returned by :class:`DeepSeekFlash`."""

    content: str
    reasoning_content: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    model: str

    @property
    def reasoning_content_present(self) -> bool:
        return bool(self.reasoning_content)


def load_env_file(path: str | Path) -> None:
    """Load simple ``KEY=VALUE`` credentials, accepting UTF-8 BOM files."""
    env_path = Path(path)
    if not env_path.is_file():
        raise FileNotFoundError(f"DeepSeek credential file not found: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ[key] = value.strip().strip('"').strip("'")


class DeepSeekFlash:
    """A small interface for DeepSeek Flash chat completions.

    ``ask`` is suitable for normal text prompts. ``complete`` accepts an
    OpenAI-format message sequence when a caller needs multi-turn context.
    Thinking is enabled by default; its tokens and visible answer tokens share
    ``max_output_tokens``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("A DeepSeek API key is required")
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
    def from_env_file(cls, path: str | Path = ".env.llm", **kwargs: Any) -> "DeepSeekFlash":
        """Build an adapter from ``OPENAI_API_KEY`` or ``DEEPSEEK_API_KEY``."""
        load_env_file(path)
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Credential file must contain DEEPSEEK_API_KEY or OPENAI_API_KEY")
        return cls(api_key=api_key, **kwargs)

    def ask(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = True,
        reasoning_effort: str = "low",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        json_object: bool = False,
        top_p: float = 1.0,
        temperature: float | None = None,
    ) -> DeepSeekResult:
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            json_object=json_object,
            top_p=top_p,
            temperature=temperature,
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        thinking: bool = True,
        reasoning_effort: str = "low",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        json_object: bool = False,
        top_p: float = 1.0,
        temperature: float | None = None,
    ) -> DeepSeekResult:
        """Execute one completion and return final content plus diagnostics."""
        if not messages:
            raise ValueError("messages must not be empty")
        if reasoning_effort not in _EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(_EFFORTS)}")
        if not 1 <= max_output_tokens <= 393_216:
            raise ValueError("max_output_tokens must be between 1 and 393216")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "extra_body": {
                "thinking": {"type": "enabled" if thinking else "disabled"},
                "reasoning_effort": reasoning_effort if thinking else "none",
            },
        }
        if json_object:
            request["response_format"] = {"type": "json_object"}
        if temperature is not None:
            request["temperature"] = temperature

        response = self._client.chat.completions.create(**request)
        if not getattr(response, "choices", None):
            raise RuntimeError("DeepSeek returned no choices")
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None) or ""
        if not content.strip():
            raise RuntimeError(f"DeepSeek returned empty final content (finish_reason={getattr(choice, 'finish_reason', None)!r})")
        usage = getattr(response, "usage", None)
        reasoning = getattr(message, "reasoning_content", None)
        return DeepSeekResult(
            content=content,
            reasoning_content=reasoning or None,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            model=getattr(response, "model", None) or self.model,
        )

    def ask_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Run a JSON-object completion and decode its final content."""
        result = self.ask(system, user, json_object=True, **kwargs)
        return json.loads(result.content)

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> "DeepSeekFlash":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
