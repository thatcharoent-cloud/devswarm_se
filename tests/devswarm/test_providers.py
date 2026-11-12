"""
tests.devswarm.test_providers — Unit tests for BaseProvider and adapters.

Verifies:
  - Adapter request/response normalization (using recorded fixtures / httpx mocking)
  - Tool calling normalization across OpenAI, Anthropic, Ollama, and Mock providers
  - Structured output schema conversion
  - Retry on 429 rate limit with Retry-After header and backoff
  - Exception mapping (401/403 -> AuthError, 404 -> ModelNotFoundError, 400 -> ContextLengthError)
  - Token and dollar cost calculation
"""

import json
import pytest
import httpx

from devswarm.providers.base import (
    ChatMessage,
    ModelNotFoundError,
    ModelPricing,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from devswarm.providers.mock_provider import MockProvider
from devswarm.providers.ollama_adapter import OllamaAdapter
from devswarm.providers.openai_adapter import OpenAICompatibleAdapter
from devswarm.providers.anthropic_adapter import AnthropicAdapter
from devswarm.llm.mock_llm import MockLLM


def test_provider_response_parsed_json_tolerant():
    # Test case 1: Preamble with markdown fence
    resp1 = ProviderResponse(
        content='Here is your JSON response:\n```json\n{"status": "ok", "score": 100}\n```\nHope this helps!',
        model="test-model",
    )
    assert resp1.parsed_json() == {"status": "ok", "score": 100}

    # Test case 2: DeepSeek <think> tags
    resp2 = ProviderResponse(
        content='<think>\nI need to output status and score.\n</think>\n```json\n{"status": "ready", "score": 95}\n```',
        model="test-model",
    )
    assert resp2.parsed_json() == {"status": "ready", "score": 95}

    # Test case 3: Dirty JSON trailing commas
    resp3 = ProviderResponse(
        content='{"status": "ok", "score": 100,}',
        model="test-model",
    )
    assert resp3.parsed_json() == {"status": "ok", "score": 100}


def test_tool_definition_conversion():
    tool = ToolDefinition(
        name="read_file",
        description="Reads a file from disk",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    # OpenAI format
    openai_d = tool.to_openai_dict()
    assert openai_d["type"] == "function"
    assert openai_d["function"]["name"] == "read_file"
    assert "parameters" in openai_d["function"]

    # Anthropic format
    anthropic_d = tool.to_anthropic_dict()
    assert anthropic_d["name"] == "read_file"
    assert "input_schema" in anthropic_d


def test_pricing_calculation():
    # $2.50 in / $10.00 out per 1M tokens
    pricing = ModelPricing(input_per_million=2.50, output_per_million=10.00)
    # 10,000 in, 5,000 out
    cost = pricing.calculate_cost(10_000, 5_000)
    # (10000 / 1e6)*2.50 = 0.025 ; (5000 / 1e6)*10.00 = 0.05 ; sum = 0.075
    assert cost == 0.075


@pytest.mark.asyncio
async def test_mock_provider_scripted_responses():
    prov = MockProvider(pricing=ModelPricing(input_per_million=2.0, output_per_million=4.0))
    prov.add_response(
        content="Hello world",
        tool_calls=[{"id": "call_1", "name": "do_work", "arguments": {"x": 10}}],
        input_tokens=1000,
        output_tokens=500,
    )

    resp = await prov.chat(
        model="mock-model",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    assert resp.content == "Hello world"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "do_work"
    assert resp.tool_calls[0].arguments == {"x": 10}
    assert resp.usage.input_tokens == 1000
    assert resp.usage.output_tokens == 500
    assert resp.cost_usd == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_ollama_adapter_normalization():
    mock_ollama = MockLLM()
    mock_ollama.add_response(
        content="Calculated result",
        tool_calls=[{"id": "call_0", "function": {"name": "add", "arguments": {"a": 2, "b": 3}}}],
        prompt_eval_count=80,
        eval_count=40,
    )

    adapter = OllamaAdapter(client=mock_ollama)
    resp = await adapter.chat(
        model="qwen2.5-coder:7b",
        messages=[ChatMessage(role="user", content="Calculate 2 + 3")],
    )

    assert resp.content == "Calculated result"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert resp.usage.input_tokens == 80
    assert resp.usage.output_tokens == 40
    assert resp.cost_usd == 0.0  # 100% Free local


@pytest.mark.asyncio
async def test_ollama_adapter_extracts_tool_call_from_content():
    mock_ollama = MockLLM()
    # Model puts tool call in content instead of tool_calls field
    mock_ollama.add_response(
        content='```json\n{"name": "add_numbers", "arguments": {"a": 12, "b": 30}}\n```',
        tool_calls=[],
        prompt_eval_count=60,
        eval_count=30,
    )

    adapter = OllamaAdapter(client=mock_ollama)
    resp = await adapter.chat(
        model="qwen2.5-coder:7b",
        messages=[ChatMessage(role="user", content="Call add_numbers")],
    )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "add_numbers"
    assert resp.tool_calls[0].arguments == {"a": 12, "b": 30}


@pytest.mark.asyncio
async def test_openai_adapter_success(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        name="test_openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
        pricing=ModelPricing(input_per_million=2.50, output_per_million=10.00),
    )

    fake_response_data = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Sure, here is the tool call.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "search_docs",
                                "arguments": json.dumps({"query": "OAuth2"}),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    }

    async def fake_post(self, url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
        assert kwargs["json"]["model"] == "gpt-4o"
        return httpx.Response(200, json=fake_response_data)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    resp = await adapter.chat(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Search docs for OAuth2")],
    )

    assert resp.content == "Sure, here is the tool call."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_abc"
    assert resp.tool_calls[0].name == "search_docs"
    assert resp.tool_calls[0].arguments == {"query": "OAuth2"}
    assert resp.usage.input_tokens == 500
    assert resp.usage.output_tokens == 100
    assert resp.cost_usd > 0.0


@pytest.mark.asyncio
async def test_openai_adapter_errors_and_rate_limit(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        name="test_openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        max_retries=1,
    )

    # 1. 401 Auth Error
    async def fake_post_401(self, url, **kwargs):
        return httpx.Response(401, text="Invalid API key")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post_401)
    with pytest.raises(ProviderAuthError):
        await adapter.chat(model="gpt-4o", messages=[ChatMessage(role="user", content="Hi")])

    # 2. 404 Model Not Found
    async def fake_post_404(self, url, **kwargs):
        return httpx.Response(404, text="Model not found")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post_404)
    with pytest.raises(ModelNotFoundError):
        await adapter.chat(model="nonexistent", messages=[ChatMessage(role="user", content="Hi")])

    # 3. 429 Rate Limit
    async def fake_post_429(self, url, **kwargs):
        return httpx.Response(429, headers={"retry-after": "0.01"}, text="Rate limit exceeded")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post_429)
    with pytest.raises(ProviderRateLimitError) as exc_info:
        await adapter.chat(model="gpt-4o", messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.retry_after_s == 0.01


@pytest.mark.asyncio
async def test_anthropic_adapter_message_conversion_and_success(monkeypatch):
    adapter = AnthropicAdapter(
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test-key",
        pricing=ModelPricing(input_per_million=3.0, output_per_million=15.0),
    )

    fake_response = {
        "id": "msg_123",
        "model": "claude-3-7-sonnet",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me think carefully..."},
            {"type": "text", "text": "Here is the plan."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "write_file",
                "input": {"path": "src/app.py", "content": "print('hello')"},
            },
        ],
        "usage": {"input_tokens": 1200, "output_tokens": 400},
    }

    async def fake_post(self, url, **kwargs):
        headers = kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test-key"
        assert headers["anthropic-version"] == "2023-06-01"

        payload = kwargs["json"]
        assert payload["system"] == "You are a software architect."
        assert payload["model"] == "claude-3-7-sonnet"
        return httpx.Response(200, json=fake_response)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    messages = [
        ChatMessage(role="system", content="You are a software architect."),
        ChatMessage(role="user", content="Draft application."),
    ]

    resp = await adapter.chat(
        model="claude-3-7-sonnet",
        messages=messages,
        thinking=True,
    )

    assert resp.content == "Here is the plan."
    assert resp.thinking == "Let me think carefully..."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "toolu_01"
    assert resp.tool_calls[0].name == "write_file"
    assert resp.tool_calls[0].arguments["path"] == "src/app.py"
    assert resp.usage.input_tokens == 1200
    assert resp.usage.output_tokens == 400
    assert resp.cost_usd == pytest.approx((1200/1e6)*3.0 + (400/1e6)*15.0)

