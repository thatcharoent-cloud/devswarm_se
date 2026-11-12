"""
tests/devswarm/test_bench.py — Unit tests for model benchmark report generation.
"""

import json
import pytest
from devswarm.llm.bench import ModelBenchmark, ModelBenchmarkResult
from devswarm.llm.mock_llm import MockLLM


@pytest.mark.asyncio
async def test_bench_report_generation(tmp_path):
    report_file = tmp_path / "bench_report.md"
    mock_llm = MockLLM()

    # Configure a responder that answers tool calls and code tasks properly
    def responder(**kwargs):
        tools = kwargs.get("tools")
        fmt = kwargs.get("format")
        msg = kwargs.get("messages", [{}])[-1].get("content", "")

        if tools:
            return mock_llm._default_responder.__class__(
                model=kwargs.get("model", "mock"),
                content="",
                tool_calls=[{
                    "function": {
                        "name": "edit_file",
                        "arguments": {"path": "src/app.py", "old_str": "foo", "new_str": "bar"},
                    }
                }],
                tokens_per_sec=38.0,
                eval_count=20,
            ) if hasattr(mock_llm._default_responder, "__class__") else None

        return None

    # We can also populate scripted responses directly:
    # 2 candidates in active tier, each runs: 5 tool calls, 3 JSON calls, 1 code task = 9 calls per candidate
    for _ in range(2):
        # 5 tool calls
        for _ in range(5):
            mock_llm.add_response(
                content="",
                tool_calls=[{"function": {"name": "edit_file", "arguments": {"path": "src/app.py", "old_str": "foo", "new_str": "bar"}}}],
                tokens_per_sec=36.0,
            )
        # 3 JSON calls
        for _ in range(3):
            mock_llm.add_response(
                content=json.dumps({"decision": "approve", "confidence": 0.95}),
                tokens_per_sec=40.0,
            )
        # 1 Code task
        mock_llm.add_response(
            content="```python\ndef safe_divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n```",
            tokens_per_sec=32.0,
        )

    bench = ModelBenchmark(
        client=mock_llm,
        config_path="config/models.yaml",
        report_path=report_file,
    )

    results = await bench.run()
    assert len(results) > 0
    assert report_file.exists()

    report_text = report_file.read_text(encoding="utf-8")
    assert "Swarm Local Model Benchmark Report" in report_text
    assert "Tool Call Rate" in report_text
    assert "Recommendation" in report_text

