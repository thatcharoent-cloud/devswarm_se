"""
devswarm.providers.doctor — Unified diagnostic suite for cloud and local providers.

Checks for each configured provider and model:
  1. API Key present in environment
  2. Model available (via list endpoint)
  3. Tool-calling capability smoke test
  4. Structured JSON schema output smoke test
  5. Pricing definition known
  6. Round-trip latency
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from devswarm.llm.doctor import Doctor as OllamaDoctor
from devswarm.providers.base import (
    BaseProvider,
    ChatMessage,
    ModelPricing,
    ToolCall,
    ToolDefinition,
)
from devswarm.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class ProviderCheckResult:
    provider: str
    model: str
    key_present: bool
    model_exists: bool
    tool_call_ok: bool
    schema_ok: bool
    pricing_known: bool
    latency_ms: float
    status: str  # PASS, WARN, FAIL
    details: str = ""


class ProvidersDoctor:
    """Runs diagnostics across configured hybrid swarm providers."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self.registry = registry or ProviderRegistry()

    async def check_provider_model(
        self,
        provider_name: str,
        model_name: str,
        skip_live_calls: bool = False,
    ) -> ProviderCheckResult:
        """Run full battery of diagnostics for a provider and model."""
        cfg = self.registry.providers_config.get(provider_name, {})
        key_env = cfg.get("api_key_env", "")
        api_key = os.getenv(key_env) if key_env else None

        key_present = bool(api_key) or provider_name == "ollama" or skip_live_calls or self.registry.mock_mode

        if not key_present:
            return ProviderCheckResult(
                provider=provider_name,
                model=model_name,
                key_present=False,
                model_exists=False,
                tool_call_ok=False,
                schema_ok=False,
                pricing_known=False,
                latency_ms=0.0,
                status="FAIL",
                details=f"Missing API key env var: {key_env}",
            )

        # 1. Model Catalog & Pricing check
        m_catalog = self.registry.models_catalog.get(model_name, {})
        pricing_known = "pricing" in m_catalog

        if skip_live_calls:
            return ProviderCheckResult(
                provider=provider_name,
                model=model_name,
                key_present=key_present,
                model_exists=True,
                tool_call_ok=True,
                schema_ok=True,
                pricing_known=pricing_known,
                latency_ms=10.0,
                status="PASS" if pricing_known else "WARN",
                details="Dry run / mocked validation",
            )

        provider = self.registry.get_provider(provider_name)

        # 2. Check model exists via list endpoint
        model_exists = False
        try:
            available = await provider.list_models()
            model_exists = any(model_name in m for m in available) if available else True
        except Exception as exc:
            logger.debug("Model list check failed for %s: %s", provider_name, exc)
            model_exists = False

        # 3. Tool call smoke test
        tool_call_ok = False
        schema_ok = False
        latency_ms = 0.0

        smoke_tool = ToolDefinition(
            name="add_numbers",
            description="Add two numbers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

        smoke_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "ready"]},
                "score": {"type": "integer"},
            },
            "required": ["status", "score"],
        }

        try:
            start_t = time.perf_counter()
            tool_resp = await provider.chat(
                model=model_name,
                messages=[ChatMessage(role="user", content="Call add_numbers to add 12 and 30.")],
                tools=[smoke_tool],
                temperature=0.0,
                max_tokens=256,
            )
            latency_ms = round((time.perf_counter() - start_t) * 1000.0, 1)
            tool_call_ok = bool(tool_resp.tool_calls)
            if not tool_call_ok:
                tool_resp2 = await provider.chat(
                    model=model_name,
                    messages=[ChatMessage(role="user", content="Call the add_numbers tool with a=5 and b=7.")],
                    tools=[smoke_tool],
                    temperature=0.0,
                    max_tokens=256,
                )
                tool_call_ok = bool(tool_resp2.tool_calls)
        except Exception as exc:
            logger.debug("Tool call smoke test failed for %s: %s", provider_name, exc)
            tool_call_ok = False

        # 4. JSON Schema output test
        try:
            schema_resp = await provider.chat(
                model=model_name,
                messages=[
                    ChatMessage(role="system", content="You must respond strictly with a valid JSON object. Do not include markdown fences or any explanation."),
                    ChatMessage(role="user", content='Output JSON with status="ok" and score=100.'),
                ],
                response_format=smoke_schema,
                temperature=0.0,
                max_tokens=256,
            )
            parsed = schema_resp.parsed_json()
            schema_ok = parsed.get("status") in ("ok", "ready") or parsed.get("score") == 100
            if not schema_ok:
                schema_resp2 = await provider.chat(
                    model=model_name,
                    messages=[
                        ChatMessage(role="user", content='Output JSON object: {"status": "ok", "score": 100}'),
                    ],
                    response_format=smoke_schema,
                    temperature=0.0,
                    max_tokens=256,
                )
                parsed2 = schema_resp2.parsed_json()
                schema_ok = parsed2.get("status") in ("ok", "ready") or parsed2.get("score") == 100
        except Exception as exc:
            logger.debug("JSON schema smoke test failed for %s: %s", provider_name, exc)
            schema_ok = False

        # Compute status
        if key_present and tool_call_ok and schema_ok and pricing_known:
            status = "PASS"
            details = "All capabilities verified"
        elif not key_present or not tool_call_ok:
            status = "FAIL"
            details = "Tool calling failed or key missing"
        else:
            status = "WARN"
            details = "Missing pricing or non-strict schema"

        return ProviderCheckResult(
            provider=provider_name,
            model=model_name,
            key_present=key_present,
            model_exists=model_exists,
            tool_call_ok=tool_call_ok,
            schema_ok=schema_ok,
            pricing_known=pricing_known,
            latency_ms=latency_ms,
            status=status,
            details=details,
        )

    async def run_diagnostics(self, skip_live_calls: bool = False) -> str:
        """Run diagnostics for all active roles and format markdown report."""
        lines = [
            "# 🩺 Swarm Providers Diagnostic Report",
            "",
            "| Provider | Role | Model | API Key | Exists | Tool Call | JSON Schema | Pricing | Latency | Status |",
            "|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        results: list[ProviderCheckResult] = []

        # Iterate active/hybrid roles
        roles_to_check = ["pm", "architect", "qa", "backend", "tester", "reviewer"]
        for role_name in roles_to_check:
            r_cfg = self.registry.roles_config.get(role_name, {})
            prov_name = r_cfg.get("provider", "ollama")
            model_name = r_cfg.get("model", "")
            if model_name == "<set model id>" or not model_name:
                model_name = "qwen2.5-coder:7b" if prov_name == "ollama" else "mock-model"

            res = await self.check_provider_model(prov_name, model_name, skip_live_calls=skip_live_calls)
            results.append(res)

            lines.append(
                f"| `{prov_name}` | `{role_name}` | `{model_name}` | "
                f"{'✅' if res.key_present else '❌'} | "
                f"{'✅' if res.model_exists else '⚠️'} | "
                f"{'✅' if res.tool_call_ok else '❌'} | "
                f"{'✅' if res.schema_ok else '❌'} | "
                f"{'✅' if res.pricing_known else '⚠️'} | "
                f"{res.latency_ms:.0f}ms | "
                f"**{res.status}** |"
            )

        # Check local Ollama health if configured
        if "ollama" in self.registry.providers_config and not skip_live_calls:
            lines.append("")
            lines.append("## Local Ollama Diagnostic")
            backend_cfg = self.registry.roles_config.get("backend", {})
            ollama_model = backend_cfg.get("model") or os.getenv("MODEL") or "qwen2.5-coder:7b"
            ollama_doc = OllamaDoctor(model=ollama_model)
            try:
                report = await ollama_doc.run_diagnostics()
                lines.append(f"- Status: **{report.overall_status}** (Model: `{ollama_model}`)")
                for c in report.checks:
                    if c.status != "PASS":
                        lines.append(f"  - [{c.status}] **{c.name}**: {c.message}")
                        if c.recommendation:
                            lines.append(f"    👉 Recommendation: {c.recommendation}")
                sys_mem = report.system_info.get("memory", {})
                if sys_mem:
                    lines.append(f"- Memory Info: {sys_mem}")
            except Exception as exc:
                lines.append(f"- Ollama check error: {exc}")

        return "\n".join(lines)
