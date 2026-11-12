"""
devswarm.providers.anthropic_adapter — Native Messages API adapter for Anthropic Claude.

Targets:
  - https://api.anthropic.com/v1/messages
  - https://api.anthropic.com/v1/models
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

import httpx

from devswarm.providers.base import (
    BaseProvider,
    ChatMessage,
    ModelNotFoundError,
    ModelPricing,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderTimeoutError,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseProvider):
    """Adapter for Anthropic Claude Messages API using httpx."""

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        default_timeout_s: float = 60.0,
        max_retries: int = 3,
        pricing: ModelPricing | None = None,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        super().__init__(
            name="anthropic",
            base_url=base_url,
            api_key=api_key,
            default_timeout_s=default_timeout_s,
            max_retries=max_retries,
        )
        self.pricing = pricing or ModelPricing()
        self.anthropic_version = anthropic_version

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        """Separate top-level system prompt and format Anthropic message blocks."""
        system_prompts: list[str] = []
        anthropic_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_prompts.append(msg.content)
                continue

            if msg.role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                anthropic_msgs.append({"role": "assistant", "content": content_blocks or msg.content})

            elif msg.role == "tool":
                # Anthropic tool results are user-role messages with type: tool_result
                tool_res_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "unknown",
                    "content": msg.content,
                }
                anthropic_msgs.append({"role": "user", "content": [tool_res_block]})

            else:
                # User message
                anthropic_msgs.append({"role": "user", "content": msg.content})

        # Anthropic requires alternating user/assistant messages. Merge adjacent identical roles.
        merged_msgs: list[dict[str, Any]] = []
        for m in anthropic_msgs:
            if merged_msgs and merged_msgs[-1]["role"] == m["role"]:
                prev = merged_msgs[-1]
                prev_c = prev["content"] if isinstance(prev["content"], list) else [{"type": "text", "text": prev["content"]}]
                curr_c = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
                prev["content"] = prev_c + curr_c
            else:
                merged_msgs.append(m)

        system_str = "\n\n".join(system_prompts)
        return system_str, merged_msgs

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
        """Execute chat completion with Anthropic Claude."""
        timeout = timeout_s or self.default_timeout_s
        headers = self._build_headers()
        system_str, anthropic_msgs = self._convert_messages(messages)

        # Anthropic requires max_tokens
        req_max_tokens = max_tokens or 4096

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": req_max_tokens,
            "temperature": temperature,
        }

        if system_str:
            payload["system"] = system_str

        if tools:
            payload["tools"] = [t.to_anthropic_dict() for t in tools]

        if thinking:
            # Extended thinking requires temperature=1.0 and minimum budget_tokens (1024)
            budget_tokens = min(2048, req_max_tokens - 1000)
            if budget_tokens >= 1024:
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
                payload["temperature"] = 1.0

        if response_format:
            # Instruct schema enforcement in system prompt or tool
            schema_str = json.dumps(response_format)
            if "system" in payload:
                payload["system"] += f"\n\nOUTPUT FORMAT: You MUST reply with valid JSON conforming to schema:\n{schema_str}"
            else:
                payload["system"] = f"OUTPUT FORMAT: You MUST reply with valid JSON conforming to schema:\n{schema_str}"

        url = f"{self.base_url}/messages"

        last_exc: Exception | None = None
        attempt = 0
        backoff = 1.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            while attempt <= self.max_retries:
                attempt += 1
                start_t = time.perf_counter()
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    duration = time.perf_counter() - start_t

                    if resp.status_code == 200:
                        data = resp.json()
                        return self._parse_success_response(data, model, duration)

                    status = resp.status_code
                    body_text = resp.text

                    if status in (401, 403):
                        raise ProviderAuthError(
                            f"Anthropic Auth Error ({status}): {body_text}",
                            provider=self.name,
                            model=model,
                            status_code=status,
                        )

                    if status == 404:
                        raise ModelNotFoundError(
                            f"Anthropic Model Not Found ({status}): {body_text}",
                            provider=self.name,
                            model=model,
                            status_code=404,
                        )

                    if status == 400 and "prompt is too long" in body_text.lower():
                        raise ProviderContextLengthError(
                            f"Context length exceeded: {body_text}",
                            provider=self.name,
                            model=model,
                            status_code=400,
                        )

                    # 429 or 529 (overloaded)
                    if status in (429, 529):
                        retry_after_str = resp.headers.get("retry-after")
                        retry_after_s: float | None = None
                        if retry_after_str:
                            try:
                                retry_after_s = float(retry_after_str)
                            except ValueError:
                                pass

                        if attempt > self.max_retries:
                            raise ProviderRateLimitError(
                                f"Anthropic rate limit/overloaded ({status}): {body_text}",
                                provider=self.name,
                                model=model,
                                retry_after_s=retry_after_s,
                            )
                        sleep_time = retry_after_s if (retry_after_s and retry_after_s > 0) else backoff
                        logger.warning(
                            "Anthropic rate-limited (%d). Retrying in %.2fs (attempt %d/%d)...",
                            status, sleep_time, attempt, self.max_retries,
                        )
                        await asyncio.sleep(sleep_time + random.uniform(0.1, 0.5))
                        backoff *= 2.0
                        continue

                    if status >= 500:
                        if attempt > self.max_retries:
                            raise ProviderError(
                                f"Anthropic server error ({status}): {body_text}",
                                provider=self.name,
                                model=model,
                                status_code=status,
                            )
                        logger.warning("Anthropic server error %d. Retrying in %.2fs...", status, backoff)
                        await asyncio.sleep(backoff + random.uniform(0.1, 0.5))
                        backoff *= 2.0
                        continue

                    raise ProviderError(
                        f"Anthropic request error ({status}): {body_text}",
                        provider=self.name,
                        model=model,
                        status_code=status,
                    )

                except (httpx.ConnectError, httpx.NetworkError) as exc:
                    last_exc = exc
                    if attempt > self.max_retries:
                        raise ProviderConnectionError(
                            f"Connection error to Anthropic: {exc}",
                            provider=self.name,
                            model=model,
                        ) from exc
                    logger.warning("Connection failure to Anthropic. Retrying in %.2fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    if attempt > self.max_retries:
                        raise ProviderTimeoutError(
                            f"Timeout reaching Anthropic after {timeout}s",
                            provider=self.name,
                            model=model,
                        ) from exc
                    logger.warning("Timeout from Anthropic. Retrying in %.2fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

        raise ProviderError(f"Request failed after retries: {last_exc}", provider=self.name, model=model)

    def _parse_success_response(self, data: dict[str, Any], model: str, duration: float) -> ProviderResponse:
        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        normalized_tools: list[ToolCall] = []

        for block in content_blocks:
            b_type = block.get("type")
            if b_type == "text":
                text_parts.append(block.get("text", ""))
            elif b_type == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif b_type == "tool_use":
                normalized_tools.append(
                    ToolCall(
                        id=block.get("id", f"call_{len(normalized_tools)}"),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        raw_usage = data.get("usage", {})
        input_tokens = raw_usage.get("input_tokens", 0)
        output_tokens = raw_usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        cost_usd = self.pricing.calculate_cost(input_tokens, output_tokens)

        return ProviderResponse(
            content="\n".join(text_parts),
            model=data.get("model", model),
            tool_calls=normalized_tools,
            usage=usage,
            thinking="\n".join(thinking_parts),
            cost_usd=cost_usd,
            latency_s=round(duration, 3),
            raw=data,
        )

    async def list_models(self) -> list[str]:
        """Fetch model list from /models."""
        headers = self._build_headers()
        url = f"{self.base_url}/models"
        async with httpx.AsyncClient(timeout=self.default_timeout_s) as client:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("data", [])
                    return [m.get("id", "") for m in models_list if m.get("id")]
                return []
            except Exception as exc:
                logger.warning("Failed to list models from Anthropic: %s", exc)
                return []

    async def health_check(self) -> bool:
        """Check reachability."""
        try:
            models = await self.list_models()
            return len(models) >= 0
        except Exception:
            return False

