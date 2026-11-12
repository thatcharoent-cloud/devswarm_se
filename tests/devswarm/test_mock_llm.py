"""
tests/devswarm/test_mock_llm.py — Unit tests verifying MockLLM functionality.
"""

import json
import pytest
from devswarm.llm.mock_llm import MockLLM
from devswarm.llm.ollama_client import OllamaStallError


@pytest.mark.asyncio
async def test_mock_llm_scripted_response():
    mock = MockLLM()
    mock.add_response(content="Hello world", eval_count=20, tokens_per_sec=30.0)

    resp = await mock.chat(model="mock-model", messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "Hello world"
    assert resp.eval_count == 20
    assert abs(resp.tokens_per_sec - 30.0) < 0.01
    assert len(mock.calls) == 1
    assert mock.calls[0].model == "mock-model"


@pytest.mark.asyncio
async def test_mock_llm_scripted_tool_call():
    mock = MockLLM()
    tool_call = {
        "function": {
            "name": "add_numbers",
            "arguments": {"a": 10, "b": 20},
        }
    }
    mock.add_response(content="", tool_calls=[tool_call])

    resp = await mock.chat(model="mock-model", messages=[{"role": "user", "content": "add 10 and 20"}])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "add_numbers"
    assert resp.tool_calls[0]["function"]["arguments"]["a"] == 10


@pytest.mark.asyncio
async def test_mock_llm_scripted_failure():
    mock = MockLLM()
    mock.add_failure(OllamaStallError("Simulated stream stall"))

    with pytest.raises(OllamaStallError):
        await mock.chat(model="mock-model", messages=[{"role": "user", "content": "stall test"}])


@pytest.mark.asyncio
async def test_mock_llm_default_format_json():
    mock = MockLLM()
    resp = await mock.chat(
        model="mock-model",
        messages=[{"role": "user", "content": "format test"}],
        format={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    parsed = resp.parsed_json()
    assert "status" in parsed or "result" in parsed

