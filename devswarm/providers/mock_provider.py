"""
devswarm.providers.mock_provider — Scriptable Mock Provider for test execution.

Matches the BaseProvider interface without making external network calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from devswarm.providers.base import (
    BaseProvider,
    ChatMessage,
    ModelPricing,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


@dataclass
class MockProviderCall:
    """Record of a call to MockProvider."""
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None
    response_format: dict[str, Any] | None
    temperature: float
    max_tokens: int | None
    thinking: bool


class MockProvider(BaseProvider):
    """Scriptable mock provider implementing BaseProvider."""

    def __init__(
        self,
        name: str = "mock",
        base_url: str = "mock://api",
        models: list[str] | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        super().__init__(name=name, base_url=base_url)
        self.available_models = models or ["mock-cloud-model", "mock-mid-model", "mock-local-model"]
        self.pricing = pricing or ModelPricing(input_per_million=2.5, output_per_million=10.0)
        self.calls: list[MockProviderCall] = []
        self._scripted_responses: list[Any] = []
        self._default_responder: Callable[..., ProviderResponse] | None = None

    def set_scripted_responses(self, responses: list[Any]) -> None:
        """Set a queue of ProviderResponse objects or Exceptions."""
        self._scripted_responses = list(responses)

    def add_response(
        self,
        content: str | ProviderResponse = "",
        tool_calls: list[ToolCall] | list[dict[str, Any]] | None = None,
        thinking: str = "",
        input_tokens: int = 100,
        output_tokens: int = 50,
        model: str = "mock-model",
    ) -> None:
        """Convenience method to enqueue a successful response."""
        if isinstance(content, ProviderResponse):
            self._scripted_responses.append(content)
            return

        normalized_tools: list[ToolCall] = []
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, ToolCall):
                    normalized_tools.append(tc)
                elif isinstance(tc, dict):
                    normalized_tools.append(ToolCall.from_dict(tc))

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        cost_usd = self.pricing.calculate_cost(input_tokens, output_tokens)

        resp = ProviderResponse(
            content=content,
            model=model,
            tool_calls=normalized_tools,
            usage=usage,
            thinking=thinking,
            cost_usd=cost_usd,
            latency_s=0.01,
        )
        self._scripted_responses.append(resp)

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
        """Process chat call from queue or default."""
        self.calls.append(
            MockProviderCall(
                model=model,
                messages=messages,
                tools=tools,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
            )
        )

        if self._scripted_responses:
            item = self._scripted_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, ProviderResponse):
                return item
            if isinstance(item, str):
                return ProviderResponse(
                    content=item,
                    model=model,
                    usage=TokenUsage(50, 50, 100),
                    cost_usd=self.pricing.calculate_cost(50, 50),
                    latency_s=0.01,
                )

        if self._default_responder:
            return self._default_responder(model=model, messages=messages, tools=tools)

        if response_format:
            import json
            schema_str = str(response_format)
            if "PMSpecification" in schema_str or "problem_statement" in schema_str:
                mock_pm = {
                    "problem_statement": "Implement requested feature and requirements.",
                    "user_stories": ["As a developer, I want clean modular implementation."],
                    "acceptance_criteria": [
                        {
                            "id": "AC-1",
                            "description": "Core function produces expected outcome without errors",
                            "verification_method": "pytest automated unit test",
                            "observable_result": "Success outcome observed",
                        }
                    ],
                    "non_goals": ["Out of scope features"],
                    "assumptions": ["Standard Python environment"],
                    "open_questions": [],
                    "milestones": [{"title": "M1", "target": "Deliver feature"}],
                    "risks": ["Minimal risk"],
                }
                return ProviderResponse(
                    content=json.dumps(mock_pm),
                    model=model,
                    usage=TokenUsage(100, 50, 150),
                    cost_usd=self.pricing.calculate_cost(100, 50),
                    latency_s=0.01,
                )
            if "ArchitectureDesign" in schema_str or "stack_decision" in schema_str:
                mock_arch = {
                    "stack_decision": "Standard Python 3",
                    "stack_justification": "Robust, standard, portable.",
                    "sources_cited": [],
                    "folder_structure": ["src/", "tests/"],
                    "data_models": [],
                    "components": ["Core Engine"],
                    "contracts": [
                        {
                            "path": "calc.py",
                            "content": "def multiply(a: int, b: int) -> int:\n    raise NotImplementedError(\"Not implemented\")\n",
                            "description": "Multiply contract stub",
                        }
                    ],
                    "adrs": [
                        {
                            "id": "ADR-001",
                            "title": "Use Python",
                            "status": "Accepted",
                            "context": "Feature requirements",
                            "decision": "Python standard library",
                            "consequences": "Clean integration",
                            "alternatives_considered": [],
                        }
                    ],
                    "security_considerations": ["Local sandbox execution"],
                    "implementer_constraints": ["Small atomic modules", "Type annotations"],
                }
                return ProviderResponse(
                    content=json.dumps(mock_arch),
                    model=model,
                    usage=TokenUsage(100, 50, 150),
                    cost_usd=self.pricing.calculate_cost(100, 50),
                    latency_s=0.01,
                )
            if "QATestSuite" in schema_str or "ac_mapping" in schema_str or "test_files" in schema_str:
                mock_qa = {
                    "test_files": [
                        {
                            "path": "tests/test_feature.py",
                            "content": "def test_feature():\n    assert True\n",
                        }
                    ],
                    "ac_mapping": {
                        "AC-1": ["test_feature"],
                    },
                }
                return ProviderResponse(
                    content=json.dumps(mock_qa),
                    model=model,
                    usage=TokenUsage(100, 50, 150),
                    cost_usd=self.pricing.calculate_cost(100, 50),
                    latency_s=0.01,
                )

        # Default fallback response
        return ProviderResponse(
            content="Mock response",
            model=model,
            usage=TokenUsage(100, 50, 150),
            cost_usd=self.pricing.calculate_cost(100, 50),
            latency_s=0.01,
        )

    async def list_models(self) -> list[str]:
        return list(self.available_models)

    async def health_check(self) -> bool:
        return True

