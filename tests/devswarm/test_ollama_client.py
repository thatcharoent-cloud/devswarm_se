"""
tests/devswarm/test_ollama_client.py — Unit tests for OllamaClient streaming, parsing, timeouts, and retries.
"""

import asyncio
import json
import pytest
import httpx

from devswarm.llm.ollama_client import (
    OllamaClient,
    OllamaChatResponse,
    OllamaAPIError,
    OllamaStallError,
    OllamaConnectionError,
)


class MockAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay_between: float = 0.0) -> None:
        self.chunks = chunks
        self.delay_between = delay_between

    async def __aiter__(self):
        for c in self.chunks:
            if self.delay_between > 0:
                await asyncio.sleep(self.delay_between)
            yield c


@pytest.mark.asyncio
async def test_streaming_and_metrics():
    """Test full streaming response assembly and metrics calculation."""
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Hello ", "thinking": "Let's greet."}}).encode("utf-8") + b"\n",
        json.dumps({"message": {"role": "assistant", "content": "world!"}}).encode("utf-8") + b"\n",
        json.dumps({
            "done": True,
            "eval_count": 20,
            "eval_duration": 500_000_000,           # 0.5s -> 40 tok/s
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100_000_000,    # 0.1s
            "load_duration": 200_000_000,           # 0.2s
            "total_duration": 800_000_000,          # 0.8s
        }).encode("utf-8") + b"\n",
    ]

    class CustomTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, stream=MockAsyncByteStream(lines), request=request)

    client = OllamaClient(transport=CustomTransport())

    resp = await client.chat(
        model="qwen2.5-coder:7b",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert resp.content == "Hello world!"
    assert resp.thinking == "Let's greet."
    assert resp.eval_count == 20
    assert resp.prompt_eval_count == 10
    assert resp.load_duration_s == 0.2
    assert abs(resp.tokens_per_sec - 40.0) < 0.01
    assert resp.done is True


@pytest.mark.asyncio
async def test_tool_calls_parsing():
    """Test parsing of tool call chunks from Ollama."""
    tool_call_obj = {
        "function": {
            "name": "read_file",
            "arguments": {"path": "main.py", "start_line": 1, "end_line": 20},
        }
    }
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call_obj]}}).encode("utf-8") + b"\n",
        json.dumps({"done": True, "eval_count": 15, "eval_duration": 300_000_000}).encode("utf-8") + b"\n",
    ]

    class ToolTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, stream=MockAsyncByteStream(lines), request=request)

    client = OllamaClient(transport=ToolTransport())
    resp = await client.chat(
        model="qwen2.5-coder:7b",
        messages=[{"role": "user", "content": "Read lines 1-20 of main.py"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "read_file"
    assert resp.tool_calls[0]["function"]["arguments"]["path"] == "main.py"


@pytest.mark.asyncio
async def test_structured_output_json_parsing():
    """Test JSON structured output validation and parsed_json helper."""
    payload_content = json.dumps({"status": "ready", "score": 98})
    lines = [
        json.dumps({"message": {"role": "assistant", "content": payload_content}}).encode("utf-8") + b"\n",
        json.dumps({"done": True, "eval_count": 10, "eval_duration": 200_000_000}).encode("utf-8") + b"\n",
    ]

    class JSONTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, stream=MockAsyncByteStream(lines), request=request)

    client = OllamaClient(transport=JSONTransport())
    resp = await client.chat(
        model="qwen2.5-coder:7b",
        messages=[{"role": "user", "content": "Send status JSON"}],
        format={"type": "object", "properties": {"status": {"type": "string"}, "score": {"type": "integer"}}},
    )

    parsed = resp.parsed_json()
    assert parsed["status"] == "ready"
    assert parsed["score"] == 98


@pytest.mark.asyncio
async def test_stall_detection_raises_stall_error():
    """Test that stream stall (delay between tokens exceeding stall timeout) raises OllamaStallError."""
    lines = [
        json.dumps({"message": {"content": "First token"}}).encode("utf-8") + b"\n",
        json.dumps({"message": {"content": "Second token"}}).encode("utf-8") + b"\n",
    ]

    class StallingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, stream=MockAsyncByteStream(lines, delay_between=0.15), request=request)

    client = OllamaClient(stall_timeout_s=0.05, load_timeout_s=0.5, transport=StallingTransport())

    with pytest.raises(OllamaStallError) as exc_info:
        await client.chat(model="qwen2.5-coder:7b", messages=[{"role": "user", "content": "test"}])

    assert "stalled" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_retry_on_connection_error_only():
    """Test that connection error retries with backoff and succeeds on retry."""
    attempts = 0

    lines = [
        json.dumps({"message": {"content": "Recovered!"}}).encode("utf-8") + b"\n",
        json.dumps({"done": True, "eval_count": 2}).encode("utf-8") + b"\n",
    ]

    class FailThenSucceedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("Connection refused by localhost:11434")

            return httpx.Response(200, stream=MockAsyncByteStream(lines), request=request)

    client = OllamaClient(transport=FailThenSucceedTransport())
    resp = await client.chat(model="qwen2.5-coder:7b", messages=[{"role": "user", "content": "retry test"}])

    assert resp.content == "Recovered!"
    assert attempts == 2


@pytest.mark.asyncio
async def test_no_retry_on_4xx_or_5xx_api_error():
    """Test that HTTP error (e.g. 404 model not found) does NOT retry and fails immediately."""
    attempts = 0

    class ErrorTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(404, text="model 'missing-model' not found", request=request)

    client = OllamaClient(transport=ErrorTransport())
    with pytest.raises(OllamaAPIError) as exc_info:
        await client.chat(model="missing-model", messages=[{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code == 404
    assert attempts == 1  # No retries on 404
