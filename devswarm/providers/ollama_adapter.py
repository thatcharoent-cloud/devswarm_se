"""
devswarm.providers.ollama_adapter — Adapter for local Ollama models.

Wraps OllamaClient to conform to the normalized BaseProvider interface.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from devswarm.agents.parser import extract_tool_calls
from devswarm.llm.ollama_client import OllamaClient, OllamaError, OllamaConnectionError
from devswarm.providers.base import (
    BaseProvider,
    ChatMessage,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderResponse,
    ProviderTimeoutError,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseProvider):
    """Adapter for running local models via Ollama."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        client: OllamaClient | None = None,
        default_timeout_s: float = 120.0,
    ) -> None:
        super().__init__(
            name="ollama",
            base_url=base_url,
            api_key=None,
            default_timeout_s=default_timeout_s,
        )
        self.client = client or OllamaClient(base_url=base_url)

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
        """Call Ollama /api/chat with normalized inputs and outputs."""
        start_t = time.perf_counter()

        # Convert ChatMessage list to Ollama dict list
        raw_messages: list[dict[str, Any]] = []
        for msg in messages:
            m_dict: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m_dict["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            raw_messages.append(m_dict)

        # Convert ToolDefinition list to Ollama tools
        raw_tools: list[dict[str, Any]] | None = None
        if tools:
            raw_tools = [t.to_openai_dict() for t in tools]

        # Options
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        try:
            resp = await self.client.chat(
                model=model,
                messages=raw_messages,
                tools=raw_tools,
                format=response_format,
                thinking=thinking,
                options=options,
            )
        except OllamaConnectionError as exc:
            raise ProviderConnectionError(str(exc), provider="ollama", model=model) from exc
        except OllamaError as exc:
            err_msg = str(exc)
            if "not found" in err_msg.lower():
                raise ModelNotFoundError(err_msg, provider="ollama", model=model) from exc
            raise ProviderError(err_msg, provider="ollama", model=model) from exc

        duration = time.perf_counter() - start_t

        # Convert tool calls to normalized ToolCall
        normalized_tools: list[ToolCall] = []
        parsed_tools = extract_tool_calls(
            response_tool_calls=resp.tool_calls,
            content=resp.content,
        )
        for i, pt in enumerate(parsed_tools):
            normalized_tools.append(
                ToolCall(
                    id=pt.tool_call_id or f"call_{i}",
                    name=pt.name,
                    arguments=pt.arguments if isinstance(pt.arguments, dict) else {},
                )
            )

        if not normalized_tools and resp.tool_calls:
            for i, tc in enumerate(resp.tool_calls):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                normalized_tools.append(
                    ToolCall(
                        id=tc.get("id", f"call_{i}") if isinstance(tc, dict) else f"call_{i}",
                        name=fn.get("name", "") if isinstance(fn, dict) else "",
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        usage = TokenUsage(
            input_tokens=resp.prompt_eval_count,
            output_tokens=resp.eval_count,
            total_tokens=resp.prompt_eval_count + resp.eval_count,
        )

        return ProviderResponse(
            content=resp.content,
            model=model,
            tool_calls=normalized_tools,
            usage=usage,
            thinking=resp.thinking,
            cost_usd=0.0,
            latency_s=round(duration, 3),
            raw=resp,
        )

    async def list_models(self) -> list[str]:
        """Fetch installed tags from Ollama."""
        try:
            if hasattr(self.client, "list_models"):
                return await self.client.list_models()
            elif hasattr(self.client, "tags"):
                models_info = await self.client.tags()
                return [m.get("name", "") for m in models_info if m.get("name")]
            return []
        except Exception as exc:
            logger.warning("Failed to list models from Ollama: %s", exc)
            return []

    async def health_check(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            if hasattr(self.client, "get_version"):
                ver = await self.client.get_version()
            elif hasattr(self.client, "version"):
                ver = await self.client.version()
            else:
                ver = None
            return bool(ver)
        except Exception:
            return False

