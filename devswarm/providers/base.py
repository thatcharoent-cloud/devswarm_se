"""
devswarm.providers.base — Normalized provider interface and shared data structures.

Enables the swarm to interact with diverse LLM providers (Ollama, OpenAI, Anthropic,
DeepSeek, Gemini) via a unified async interface. Loops, agents, and orchestrators
interact strictly with these normalized contracts.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    MOCK = "mock"


# Standard Exceptions
class ProviderError(Exception):
    """Base exception for provider communication errors."""

    def __init__(self, message: str, provider: str = "", model: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code


class ProviderAuthError(ProviderError):
    """Raised when authentication fails (missing or invalid API key)."""


class ProviderRateLimitError(ProviderError):
    """Raised when a 429 Rate Limit is encountered."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, status_code=429)
        self.retry_after_s = retry_after_s


class ProviderConnectionError(ProviderError):
    """Raised on network / DNS / connection failure."""


class ProviderTimeoutError(ProviderError):
    """Raised when a request exceeds configured timeout."""


class ProviderContextLengthError(ProviderError):
    """Raised when the prompt exceeds the model's context window."""


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available or not found."""


# Shared Data Contracts
@dataclass
class ChatMessage:
    """Normalized chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # Populated when role == "tool"
    name: str | None = None          # Optional tool name or author

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        raw_tools = data.get("tool_calls", [])
        tool_calls = [
            tc if isinstance(tc, ToolCall) else ToolCall.from_dict(tc)
            for tc in raw_tools
        ]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class ToolDefinition:
    """Normalized tool specification exposed to the model."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to standard OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to standard Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """Normalized tool call returned by any model."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        fn = data.get("function", {})
        if isinstance(fn, dict) and "name" in fn:
            name = fn.get("name", "")
            args = fn.get("arguments", {})
        else:
            name = data.get("name", "")
            args = data.get("arguments", {})

        call_id = data.get("id") or data.get("tool_call_id", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        return cls(
            id=call_id,
            name=name,
            arguments=args if isinstance(args, dict) else {},
        )


@dataclass
class TokenUsage:
    """Normalized token accounting."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class ModelPricing:
    """Per-million tokens pricing in USD."""
    input_per_million: float = 0.0
    output_per_million: float = 0.0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost given token counts."""
        cost_in = (input_tokens / 1_000_000.0) * self.input_per_million
        cost_out = (output_tokens / 1_000_000.0) * self.output_per_million
        return round(cost_in + cost_out, 6)


@dataclass
class ProviderCapabilities:
    """Known feature capabilities for a specific model."""
    tools: bool = True
    json_schema: bool = True
    thinking: bool = False
    web_search_native: bool = False
    context_window: int = 16384


@dataclass
class ProviderResponse:
    """Normalized response payload returned by any provider."""
    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    thinking: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0
    raw: Any = None

    def parsed_json(self) -> dict[str, Any]:
        """Parse text content as JSON tolerantly, stripping markdown blocks, think tags, and preamble."""
        text = self.content.strip()
        if not text:
            return {}

        # 1. Direct parse attempt
        try:
            parsed = json.loads(text, strict=False)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except Exception:
            pass

        # 2. Strip think tags (<think>...</think>)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # 3. Search for markdown code block ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                parsed = json.loads(candidate, strict=False)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
            except Exception:
                try:
                    from devswarm.agents.parser import sanitize_dirty_json
                    parsed = json.loads(sanitize_dirty_json(candidate), strict=False)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        return parsed[0]
                except Exception:
                    pass

        # 4. Search for outermost JSON object { ... }
        obj_match = re.search(r"(\{[\s\S]*\})", cleaned)
        if obj_match:
            candidate = obj_match.group(1).strip()
            try:
                parsed = json.loads(candidate, strict=False)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    from devswarm.agents.parser import sanitize_dirty_json
                    parsed = json.loads(sanitize_dirty_json(candidate), strict=False)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        # 5. Fallback: sanitize original text
        try:
            from devswarm.agents.parser import sanitize_dirty_json
            parsed = json.loads(sanitize_dirty_json(text), strict=False)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except Exception:
            pass

        # 6. Fallback: repair truncated JSON
        try:
            from devswarm.agents.parser import repair_truncated_json
            repaired = repair_truncated_json(cleaned)
            parsed = json.loads(repaired, strict=False)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except Exception:
            pass

        try:
            return json.loads(text, strict=False)
        except Exception:
            from devswarm.agents.parser import repair_truncated_json
            return json.loads(repair_truncated_json(text), strict=False)


class BaseProvider(ABC):
    """Abstract interface that all provider adapters must implement."""

    def __init__(
        self,
        name: str,
        base_url: str = "",
        api_key: str | None = None,
        default_timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_timeout_s = default_timeout_s
        self.max_retries = max_retries

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        thinking: bool = False,
        timeout_s: float | None = None,
    ) -> ProviderResponse:
        """Execute a chat completion with optional tools or structured output."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return available model IDs from provider's catalog endpoint."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Quick ping or connectivity check returning True if reachable."""
        ...

