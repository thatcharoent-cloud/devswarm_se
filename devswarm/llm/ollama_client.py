"""
devswarm.llm.ollama_client — Native client for Ollama's local chat API.

Communicates with Ollama (http://127.0.0.1:11434) via httpx using streaming.
Features:
  - Native options (num_ctx, temperature, num_predict).
  - keep_alive handling to keep the model loaded in GPU memory.
  - Native tools calling support.
  - Native structured output (JSON schema format).
  - Thinking toggle (think: true/false).
  - Streaming with stall detection (time between chunks) and generous model load timeout.
  - Evaluation statistics: tokens/sec, time to first token, load time, prompt tokens.
  - Retry on connection errors only with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LOAD_TIMEOUT_S = float(os.getenv("OLLAMA_LOAD_TIMEOUT_S", "180.0"))
DEFAULT_STALL_TIMEOUT_S = float(os.getenv("OLLAMA_STALL_TIMEOUT_S", "60.0"))
DEFAULT_CONNECT_TIMEOUT_S = 10.0
MAX_CONNECT_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_FACTOR = 2.0


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama daemon cannot be reached."""


class OllamaAPIError(OllamaError):
    """Raised when Ollama returns a non-200 HTTP response."""

    def __init__(self, status_code: int, message: str, details: Any = None) -> None:
        super().__init__(f"Ollama API HTTP {status_code}: {message}")
        self.status_code = status_code
        self.details = details


class OllamaStallError(OllamaError):
    """Raised when streaming from Ollama stalls without generating tokens."""


@dataclass
class OllamaChatResponse:
    """Parsed output and generation metrics from an Ollama chat call."""

    model: str
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_eval_count: int = 0
    prompt_eval_duration_s: float = 0.0
    eval_count: int = 0
    eval_duration_s: float = 0.0
    load_duration_s: float = 0.0
    total_duration_s: float = 0.0
    time_to_first_token_s: float = 0.0
    tokens_per_sec: float = 0.0
    done: bool = True
    raw_final_chunk: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_eval_count + self.eval_count

    def parsed_json(self) -> Any:
        """Parse content as JSON if structured output was requested."""
        if not self.content.strip():
            raise ValueError("Response content is empty; cannot parse JSON")
        return json.loads(self.content)


class OllamaClient:
    """Asynchronous client interacting with the local Ollama API."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        load_timeout_s: float = DEFAULT_LOAD_TIMEOUT_S,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.load_timeout_s = load_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.transport = transport
        self._lock = asyncio.Lock()

    def _build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        format: dict[str, Any] | str | None = None,
        thinking: bool = False,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = -1,
    ) -> dict[str, Any]:
        """Construct the request body for /api/chat."""
        # Ollama API expects integer for indefinite (-1) or immediate unload (0).
        # String without units (e.g. "-1") causes Go duration parse error "missing unit in duration".
        normalized_keep_alive: int | str = keep_alive
        if isinstance(keep_alive, str):
            try:
                normalized_keep_alive = int(keep_alive)
            except ValueError:
                normalized_keep_alive = keep_alive

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": normalized_keep_alive,
        }

        # Thinking toggle: Ollama native parameter
        payload["think"] = bool(thinking)

        if tools:
            payload["tools"] = tools

        if format is not None:
            payload["format"] = format

        if options:
            payload["options"] = options

        return payload

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        format: dict[str, Any] | str | None = None,
        thinking: bool = False,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = -1,
        max_retries: int = MAX_CONNECT_RETRIES,
    ) -> OllamaChatResponse:
        """Send a chat request to Ollama with streaming, stall detection, and retry.

        Retries ONLY on connection errors (e.g. daemon temporarily busy/unreachable).
        Fails immediately on 4xx/5xx responses or validation errors.
        """
        payload = self._build_payload(
            model=model,
            messages=messages,
            tools=tools,
            format=format,
            thinking=thinking,
            options=options,
            keep_alive=keep_alive,
        )

        async with self._lock:
            backoff = INITIAL_BACKOFF_S
            last_conn_exc: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await self._stream_chat(payload, model)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
                    last_conn_exc = exc
                    logger.warning(
                        "Ollama connection error on attempt %d/%d: %s",
                        attempt, max_retries, exc
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(backoff + random.uniform(0, 0.25))
                        backoff *= BACKOFF_FACTOR
                    continue
                except OllamaError:
                    # Do not retry semantic or API errors
                    raise

            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {self.base_url} after {max_retries} attempts: {last_conn_exc}"
            ) from last_conn_exc

    async def unload_model(self, model: str) -> None:
        """Unload a model from memory via keep_alive=0 to clear KV cache and free memory."""
        logger.info("Unloading model '%s' from memory via keep_alive=0", model)
        async with self._lock:
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.connect_timeout_s, transport=self.transport) as client:
                    await client.post(
                        "/api/chat",
                        json={"model": model, "messages": [], "keep_alive": 0},
                    )
            except Exception as exc:
                logger.debug("Unload request returned: %s", exc)

    async def _stream_chat(
        self,
        payload: dict[str, Any],
        model: str,
    ) -> OllamaChatResponse:
        """Execute the streaming request with stall detection and metrics recording."""
        start_time = time.monotonic()
        first_token_time: float | None = None

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        final_chunk: dict[str, Any] = {}

        # Custom timeout: generous load timeout for the connection and first byte
        timeout = httpx.Timeout(
            connect=self.connect_timeout_s,
            read=None,  # Handled explicitly via stall detection
            write=15.0,
            pool=15.0,
        )

        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=self.transport) as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise OllamaAPIError(
                        status_code=response.status_code,
                        message=body.decode(errors="replace"),
                    )

                lines_iter = response.aiter_lines()
                # Before receiving the first chunk, allow up to load_timeout_s (model loading)
                current_timeout = self.load_timeout_s

                while True:
                    try:
                        line = await asyncio.wait_for(
                            lines_iter.__anext__(),
                            timeout=current_timeout,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        is_initial = first_token_time is None
                        stage = "loading model" if is_initial else "generating output"
                        raise OllamaStallError(
                            f"Ollama stream stalled during {stage} (no response for {current_timeout:.1f}s)"
                        )

                    # Once the first chunk arrives, enforce the tighter stall timeout
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                        current_timeout = self.stall_timeout_s

                    line_str = line.strip()
                    if not line_str:
                        continue

                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as exc:
                        logger.debug("Skipping unparseable chunk line: %s (%s)", line_str, exc)
                        continue

                    msg = chunk.get("message", {})
                    # Accumulate content
                    chunk_content = msg.get("content", "")
                    if chunk_content:
                        content_parts.append(chunk_content)

                    # Accumulate thinking
                    chunk_thinking = msg.get("thinking", "")
                    if chunk_thinking:
                        thinking_parts.append(chunk_thinking)

                    # Accumulate tool calls
                    chunk_tool_calls = msg.get("tool_calls", [])
                    if chunk_tool_calls:
                        tool_calls.extend(chunk_tool_calls)

                    if chunk.get("done", False):
                        final_chunk = chunk
                        break

        total_duration_wall = time.monotonic() - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_duration_wall

        # Extract timing and token counts from final chunk (nanoseconds in Ollama API)
        prompt_eval_count = int(final_chunk.get("prompt_eval_count", 0))
        eval_count = int(final_chunk.get("eval_count", 0))

        load_duration_ns = final_chunk.get("load_duration", 0)
        total_duration_ns = final_chunk.get("total_duration", 0)
        prompt_eval_duration_ns = final_chunk.get("prompt_eval_duration", 0)
        eval_duration_ns = final_chunk.get("eval_duration", 0)

        load_duration_s = load_duration_ns / 1e9 if load_duration_ns else 0.0
        total_duration_s = total_duration_ns / 1e9 if total_duration_ns else total_duration_wall
        prompt_eval_duration_s = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else 0.0
        eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns else 0.0

        if eval_duration_s > 0 and eval_count > 0:
            tokens_per_sec = eval_count / eval_duration_s
        elif total_duration_wall > 0 and eval_count > 0:
            tokens_per_sec = eval_count / total_duration_wall
        else:
            tokens_per_sec = 0.0

        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts)

        return OllamaChatResponse(
            model=model,
            content=full_content,
            thinking=full_thinking,
            tool_calls=tool_calls,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration_s=prompt_eval_duration_s,
            eval_count=eval_count,
            eval_duration_s=eval_duration_s,
            load_duration_s=load_duration_s,
            total_duration_s=total_duration_s,
            time_to_first_token_s=ttft,
            tokens_per_sec=tokens_per_sec,
            done=bool(final_chunk.get("done", True)),
            raw_final_chunk=final_chunk,
        )

    async def get_version(self) -> str:
        """Fetch Ollama daemon version."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.connect_timeout_s, transport=self.transport) as client:
            resp = await client.get("/api/version")
            if resp.status_code != 200:
                raise OllamaAPIError(resp.status_code, resp.text)
            return resp.json().get("version", "unknown")

    async def list_models(self) -> list[str]:
        """Fetch list of locally installed model tags."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.connect_timeout_s, transport=self.transport) as client:
            resp = await client.get("/api/tags")
            if resp.status_code != 200:
                raise OllamaAPIError(resp.status_code, resp.text)
            models = resp.json().get("models", [])
            tags: list[str] = []
            for m in models:
                tag = m.get("name") or m.get("model")
                if tag:
                    tags.append(tag)
            return tags

    tags = list_models
    version = get_version

    async def ps(self) -> list[dict[str, Any]]:
        """Fetch loaded models and their processor allocation (VRAM / GPU / CPU)."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.connect_timeout_s, transport=self.transport) as client:
            resp = await client.get("/api/ps")
            if resp.status_code != 200:
                raise OllamaAPIError(resp.status_code, resp.text)
            return resp.json().get("models", [])
