"""
devswarm.llm.mock_llm — Scripted MockLLM implementing the OllamaClient interface.

Used across test suites to verify:
  - Tool calling loops
  - Structured output parsing and schema validation
  - Stall detection and timeout handling
  - Retry logic on connection errors
  - Doctor diagnostics
  - Benchmark workflows
No real Ollama calls or network access required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from devswarm.llm.ollama_client import (
    OllamaChatResponse,
    OllamaError,
    OllamaAPIError,
    OllamaStallError,
    OllamaConnectionError,
)


@dataclass
class MockCallRecord:
    """Record of a call made to MockLLM."""
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    format: dict[str, Any] | str | None
    thinking: bool
    options: dict[str, Any] | None
    keep_alive: str | int


class MockLLM:
    """Scriptable mock client matching the OllamaClient contract."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        version: str = "0.5.12",
        installed_models: list[str] | None = None,
        loaded_models: list[dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.installed_models = installed_models or ["qwen2.5-coder:7b", "qwen3.5:9b"]
        self.loaded_models = loaded_models or [
            {
                "name": "qwen2.5-coder:7b",
                "model": "qwen2.5-coder:7b",
                "size": 4700000000,
                "size_vram": 4700000000,
                "processor": "100% GPU",
            }
        ]

        self._scripted_responses: list[Any] = []
        self._default_responder: Callable[..., OllamaChatResponse] | None = None
        self.calls: list[MockCallRecord] = []

    def set_scripted_responses(self, responses: list[Any]) -> None:
        """Set a queue of scripted responses, exceptions, or payloads."""
        self._scripted_responses = list(responses)

    def add_response(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        thinking: str = "",
        tokens_per_sec: float = 35.0,
        eval_count: int = 120,
        prompt_eval_count: int = 50,
        load_duration_s: float = 0.05,
    ) -> None:
        """Helper to append a mock chat response."""
        resp = OllamaChatResponse(
            model="mock-model",
            content=content,
            thinking=thinking,
            tool_calls=tool_calls or [],
            tokens_per_sec=tokens_per_sec,
            eval_count=eval_count,
            prompt_eval_count=prompt_eval_count,
            load_duration_s=load_duration_s,
            eval_duration_s=(eval_count / tokens_per_sec) if tokens_per_sec > 0 else 0.0,
            total_duration_s=1.0,
            time_to_first_token_s=0.1,
            done=True,
        )
        self._scripted_responses.append(resp)

    def add_failure(self, exc: Exception) -> None:
        """Helper to append an exception to raise on next call."""
        self._scripted_responses.append(exc)

    def set_default_responder(self, fn: Callable[..., OllamaChatResponse]) -> None:
        """Set a fallback responder when scripted queue is empty."""
        self._default_responder = fn

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        format: dict[str, Any] | str | None = None,
        thinking: bool = False,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "-1",
        max_retries: int = 3,
    ) -> OllamaChatResponse:
        """Simulate a chat call."""
        self.calls.append(
            MockCallRecord(
                model=model,
                messages=messages,
                tools=tools,
                format=format,
                thinking=thinking,
                options=options,
                keep_alive=keep_alive,
            )
        )

        if self._scripted_responses:
            item = self._scripted_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, OllamaChatResponse):
                item.model = model
                return item
            if isinstance(item, str):
                return OllamaChatResponse(
                    model=model,
                    content=item,
                    tokens_per_sec=30.0,
                    eval_count=len(item.split()),
                    prompt_eval_count=len(str(messages).split()),
                )
            if isinstance(item, dict):
                content = item.get("content", "")
                if not content and format:
                    content = json.dumps(item)
                return OllamaChatResponse(
                    model=model,
                    content=content,
                    tool_calls=item.get("tool_calls", []),
                    thinking=item.get("thinking", ""),
                    tokens_per_sec=float(item.get("tokens_per_sec", 30.0)),
                    eval_count=int(item.get("eval_count", 50)),
                    prompt_eval_count=int(item.get("prompt_eval_count", 20)),
                )

        if self._default_responder:
            return self._default_responder(
                model=model,
                messages=messages,
                tools=tools,
                format=format,
                thinking=thinking,
                options=options,
            )

        # Basic default response if nothing is configured
        last_msg = messages[-1].get("content", "") if messages else ""
        default_content = "OK"
        tool_calls = []

        if tools:
            tool_names = {t.get("function", {}).get("name") for t in tools}
            has_tool_result = any(m.get("role") == "tool" for m in messages)

            if "finish_task" in tool_names and has_tool_result:
                tool_calls.append({
                    "function": {
                        "name": "finish_task",
                        "arguments": {
                            "summary": "Implementation completed and verified.",
                            "files_modified": ["calc.py"],
                        },
                    }
                })
                default_content = ""
            elif "read_file" in tool_names and not has_tool_result:
                tool_calls.append({
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "calc.py"},
                    }
                })
                default_content = ""
            elif "finish_task" in tool_names:
                tool_calls.append({
                    "function": {
                        "name": "finish_task",
                        "arguments": {
                            "summary": "Task verified directly.",
                            "files_modified": [],
                        },
                    }
                })
                default_content = ""
            else:
                for t in tools:
                    fn = t.get("function", {})
                    name = fn.get("name")
                    if name == "add_numbers":
                        tool_calls.append({
                            "function": {
                                "name": "add_numbers",
                                "arguments": {"a": 12, "b": 30},
                            }
                        })
                        default_content = ""
                        break
                    elif name == "edit_file":
                        tool_calls.append({
                            "function": {
                                "name": "edit_file",
                                "arguments": {"path": "src/app.py", "old_str": "foo", "new_str": "bar"},
                            }
                        })
                        default_content = ""
                        break

        if format and isinstance(format, dict):
            props = format.get("properties", {})
            if "verdict" in props:
                default_content = json.dumps({
                    "verdict": "APPROVE",
                    "summary": "Implementation is clean, well-tested, and meets acceptance criteria.",
                    "strengths": [
                        "Deterministic quality gates passed",
                        "Narrow write scopes respected",
                        "High-risk files avoided",
                    ],
                    "concerns": [],
                    "security_assessment": "Zero secrets detected; no dangerous functions.",
                    "confidence_score": 0.95,
                })
            elif "status" in props and "code" in props:
                default_content = json.dumps({"status": "ready", "code": 200})
            elif "decision" in props and "confidence" in props:
                default_content = json.dumps({"decision": "approve", "confidence": 0.95})
            elif "tasks" in props:
                default_content = json.dumps({
                    "goal": "Implement requested engineering changes",
                    "assumptions": ["Decompose into atomic tasks", "All tests must pass"],
                    "tasks": [
                        {
                            "id": "task-1",
                            "title": "Implement feature in backend",
                            "role": "backend",
                            "description": "Implement core logic and methods in target file.",
                            "acceptance_criteria": [
                                "Implementation completed with no syntax errors",
                                "Conforms to 16 GB profile constraints",
                            ],
                            "depends_on": [],
                            "write_scope": ["calc.py"],
                            "read_scope": ["*"],
                            "risk_level": "low",
                            "minutes_budget": 5.0,
                        },
                        {
                            "id": "task-2",
                            "title": "Add and verify unit tests",
                            "role": "tester",
                            "description": "Write deterministic unit tests covering the new feature.",
                            "acceptance_criteria": [
                                "Unit tests pass deterministically",
                                "Zero regressions on existing tests",
                            ],
                            "depends_on": ["task-1"],
                            "write_scope": ["test_calc.py"],
                            "read_scope": ["*"],
                            "risk_level": "low",
                            "minutes_budget": 5.0,
                        },
                    ],
                })
            else:
                default_content = json.dumps({"status": "pass", "result": "ok"})
        elif format == "json":
            default_content = json.dumps({"result": "ok"})
        elif "SWARM_SECRET_KEY_789" in last_msg:
            default_content = "The secret key is SWARM_SECRET_KEY_789."
        elif "safe_divide" in last_msg:
            default_content = "```python\ndef safe_divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n```"

        return OllamaChatResponse(
            model=model,
            content=default_content,
            tool_calls=tool_calls,
            tokens_per_sec=40.0,
            eval_count=20,
            prompt_eval_count=15,
            load_duration_s=0.01,
            time_to_first_token_s=0.05,
            total_duration_s=0.5,
        )

    async def get_version(self) -> str:
        return self.version

    async def list_models(self) -> list[str]:
        return list(self.installed_models)

    async def ps(self) -> list[dict[str, Any]]:
        return list(self.loaded_models)

    async def unload_model(self, model: str) -> None:
        """Simulate model unloading via keep_alive=0."""
        self.loaded_models = [m for m in self.loaded_models if m.get("name") != model and m.get("model") != model]
        self.calls.append(
            MockCallRecord(
                model=model,
                messages=[],
                tools=None,
                format=None,
                thinking=False,
                options=None,
                keep_alive=0,
            )
        )
