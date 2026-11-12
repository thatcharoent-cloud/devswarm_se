"""
devswarm.providers.openai_adapter — Universal OpenAI-compatible provider adapter.

Covers:
  - OpenAI (https://api.openai.com/v1)
  - DeepSeek (https://api.deepseek.com)
  - Gemini OpenAI-compatible endpoint (https://generativelanguage.googleapis.com/v1beta/openai/)
  - Any local or remote OpenAI-compatible gateway (e.g. vLLM, LiteLLM, LM Studio)
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


class OpenAICompatibleAdapter(BaseProvider):
    """Universal adapter for OpenAI-compatible HTTP APIs using httpx."""

    def __init__(
        self,
        name: str = "openai_compatible",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        default_timeout_s: float = 60.0,
        max_retries: int = 3,
        pricing: ModelPricing | None = None,
        custom_headers: dict[str, str] | None = None,
        models_catalog: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            default_timeout_s=default_timeout_s,
            max_retries=max_retries,
        )
        self.pricing = pricing or ModelPricing()
        self.custom_headers = custom_headers or {}
        self.models_catalog = models_catalog or {}

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.custom_headers)
        return headers

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        for msg in messages:
            item: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.role == "assistant" and msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            elif msg.role == "tool":
                if msg.tool_call_id:
                    item["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    item["name"] = msg.name
            raw.append(item)
        return raw

    def _convert_response_format(self, response_format: dict[str, Any] | None) -> dict[str, Any] | None:
        if not response_format:
            return None
        # If caller already wrapped in {"type": "json_schema", ...}
        if "type" in response_format and response_format["type"] in ("json_schema", "json_object"):
            return response_format
        # If naked schema provided:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": response_format,
            },
        }

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
        """Execute chat completion with retry and backoff."""
        timeout = timeout_s or self.default_timeout_s
        headers = self._build_headers()
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
        }

        # Resolve model temperature constraints (e.g. Kimi-K3, OpenAI o1/o3)
        m_lower = model.lower()
        model_info = self.models_catalog.get(model, {})
        caps = model_info.get("capabilities", {})
        params = model_info.get("parameters", {})

        is_fixed_reasoning = (
            caps.get("supports_temperature") is False
            or "kimi-k3" in m_lower
            or "kimi_k3" in m_lower
            or "o1-" in m_lower
            or m_lower == "o1"
            or "o3-" in m_lower
            or m_lower == "o3"
        )

        if "temperature" in params:
            payload["temperature"] = params["temperature"]
        elif is_fixed_reasoning:
            # For Kimi-K3 models, temperature must be 1.0 if sent, or omitted
            if "kimi-k3" in m_lower or "kimi_k3" in m_lower:
                payload["temperature"] = 1.0
            # For o1/o3, omit temperature completely
        else:
            payload["temperature"] = temperature

        if "top_p" in params:
            payload["top_p"] = params["top_p"]

        if tools:
            payload["tools"] = [t.to_openai_dict() for t in tools]
            payload["tool_choice"] = "auto"

        rf = self._convert_response_format(response_format)
        if rf:
            payload["response_format"] = rf

        if max_tokens:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/chat/completions"

        last_exc: Exception | None = None
        attempt = 0
        backoff = 1.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            max_loop_attempts = max(self.max_retries, 6)
            while attempt <= max_loop_attempts:
                attempt += 1
                start_t = time.perf_counter()
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    duration = time.perf_counter() - start_t

                    if resp.status_code == 200:
                        data = resp.json()
                        return self._parse_success_response(data, model, duration)

                    # Handle HTTP Errors
                    status = resp.status_code
                    body_text = resp.text

                    if status in (401, 403):
                        raise ProviderAuthError(
                            f"Authentication error ({status}): {body_text}",
                            provider=self.name,
                            model=model,
                            status_code=status,
                        )

                    if status == 404:
                        raise ModelNotFoundError(
                            f"Model not found ({status}): {body_text}",
                            provider=self.name,
                            model=model,
                            status_code=404,
                        )

                    if status == 400 and "context_length" in body_text.lower():
                        raise ProviderContextLengthError(
                            f"Context length exceeded: {body_text}",
                            provider=self.name,
                            model=model,
                            status_code=400,
                        )

                    if status == 400 and ("response_format" in body_text.lower() or "schema" in body_text.lower()):
                        curr_rf = payload.get("response_format")
                        if isinstance(curr_rf, dict) and curr_rf.get("type") == "json_schema":
                            logger.info(
                                "Provider %s rejected json_schema; retrying with type='json_object'",
                                self.name,
                            )
                            payload["response_format"] = {"type": "json_object"}
                            continue

                    if status == 429:
                        # Rate limit: inspect Retry-After
                        retry_after_str = resp.headers.get("retry-after")
                        retry_after_s: float | None = None
                        if retry_after_str:
                            try:
                                retry_after_s = float(retry_after_str)
                            except ValueError:
                                pass

                        max_rate_limit_retries = max(self.max_retries, 6)
                        if attempt > max_rate_limit_retries:
                            raise ProviderRateLimitError(
                                f"Rate limit exceeded (429): {body_text}",
                                provider=self.name,
                                model=model,
                                retry_after_s=retry_after_s,
                            )
                        # For TPM rate limits on cloud providers (e.g. SiliconFlow), wait at least 5s
                        base_wait = max(backoff, 5.0)
                        sleep_time = retry_after_s if (retry_after_s and retry_after_s > 0) else base_wait
                        logger.warning(
                            "Provider %s rate-limited (429). Retrying in %.2fs (attempt %d/%d)...",
                            self.name, sleep_time, attempt, max_rate_limit_retries,
                        )
                        await asyncio.sleep(sleep_time + random.uniform(0.5, 1.5))
                        backoff = max(backoff * 2.0, 10.0)
                        continue

                    # Server error 5xx: retryable
                    if status >= 500:
                        if attempt > self.max_retries:
                            raise ProviderError(
                                f"Provider server error ({status}): {body_text}",
                                provider=self.name,
                                model=model,
                                status_code=status,
                            )
                        logger.warning(
                            "Provider %s returned status %d. Retrying in %.2fs (attempt %d/%d)...",
                            self.name, status, backoff, attempt, self.max_retries,
                        )
                        await asyncio.sleep(backoff + random.uniform(0.1, 0.5))
                        backoff *= 2.0
                        continue

                    # Unhandled client error
                    raise ProviderError(
                        f"Provider request error ({status}): {body_text}",
                        provider=self.name,
                        model=model,
                        status_code=status,
                    )

                except (httpx.ConnectError, httpx.NetworkError) as exc:
                    last_exc = exc
                    if attempt > self.max_retries:
                        raise ProviderConnectionError(
                            f"Connection error to {self.base_url}: {exc}",
                            provider=self.name,
                            model=model,
                        ) from exc
                    logger.warning("Connection failure to %s. Retrying in %.2fs...", self.name, backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    if attempt > self.max_retries:
                        raise ProviderTimeoutError(
                            f"Timeout after {timeout}s reaching {self.base_url}",
                            provider=self.name,
                            model=model,
                        ) from exc
                    logger.warning("Timeout from %s. Retrying in %.2fs...", self.name, backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

        raise ProviderError(f"Request failed after retries: {last_exc}", provider=self.name, model=model)

    def _parse_success_response(self, data: dict[str, Any], model: str, duration: float) -> ProviderResponse:
        choices = data.get("choices", [])
        if not choices:
            raise ProviderError("Empty choices returned from provider", provider=self.name, model=model)

        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or ""

        # Extract tool calls
        normalized_tools: list[ToolCall] = []
        raw_tools = msg.get("tool_calls", [])
        for i, tc in enumerate(raw_tools):
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            if isinstance(args_str, dict):
                args = args_str
            else:
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"raw": args_str}

            normalized_tools.append(
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        # Usage
        raw_usage = data.get("usage", {})
        prompt_tokens = raw_usage.get("prompt_tokens", 0)
        completion_tokens = raw_usage.get("completion_tokens", 0)
        total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)
        usage = TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        cost_usd = self.pricing.calculate_cost(prompt_tokens, completion_tokens)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            tool_calls=normalized_tools,
            usage=usage,
            thinking=thinking,
            cost_usd=cost_usd,
            latency_s=round(duration, 3),
            raw=data,
        )

    async def list_models(self) -> list[str]:
        """Fetch model IDs from /models."""
        headers = self._build_headers()
        url = f"{self.base_url}/models"
        async with httpx.AsyncClient(timeout=self.default_timeout_s) as client:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("data", [])
                    return [m.get("id", "") for m in models_list if m.get("id")]
                logger.warning("GET /models returned status %d from %s", resp.status_code, self.name)
                return []
            except Exception as exc:
                logger.warning("Failed to fetch models from %s: %s", self.name, exc)
                return []

    async def health_check(self) -> bool:
        """Ping /models or return True if list succeeds."""
        try:
            models = await self.list_models()
            return len(models) >= 0  # 200 OK
        except Exception:
            return False

