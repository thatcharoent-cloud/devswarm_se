"""
tests.devswarm.test_cloud_budget — Unit tests for cloud money budgets and reservations.

Verifies:
  - Price table lookups and conservative fallbacks for unknown models
  - Environment variable overrides for model pricing
  - Pre-call cost estimation and reservation
  - Halting with CloudBudgetExceededError when budgets are exceeded
  - ProviderRegistry role resolution and --offline fallback to local Ollama
"""

import os
import pytest

from devswarm.platform.budget import CloudBudgetExceededError, CloudBudgetTracker
from devswarm.providers.base import ModelPricing
from devswarm.providers.registry import ProviderRegistry


def test_pricing_lookup_and_fallback():
    tracker = CloudBudgetTracker(
        pricing_table={
            "known-model": ModelPricing(input_per_million=2.0, output_per_million=8.0),
        }
    )

    # Known model
    p_known = tracker.get_pricing("known-model")
    assert p_known.input_per_million == 2.0
    assert p_known.output_per_million == 8.0

    # Unknown model triggers conservative fallback
    p_unknown = tracker.get_pricing("unknown-cloud-model")
    assert p_unknown.input_per_million == 3.0
    assert p_unknown.output_per_million == 15.0


def test_pricing_env_override(monkeypatch):
    monkeypatch.setenv("PRICE_INPUT_MY_MODEL", "1.50")
    monkeypatch.setenv("PRICE_OUTPUT_MY_MODEL", "4.50")

    tracker = CloudBudgetTracker()
    p = tracker.get_pricing("my-model")
    assert p.input_per_million == 1.50
    assert p.output_per_million == 4.50


def test_cloud_budget_reservation_and_commit():
    tracker = CloudBudgetTracker(
        max_cloud_usd_run=1.00,
        max_cloud_usd_role={"pm": 0.50},
        pricing_table={"gpt-4o": ModelPricing(input_per_million=2.5, output_per_million=10.0)},
    )

    # Estimate 0.05 USD: should succeed
    res_id = tracker.reserve(role="pm", provider="openai", model="gpt-4o", estimated_cost=0.05)
    assert res_id.startswith("res_")
    assert tracker.reserved_usd == 0.05

    # Commit with actual 0.04 USD
    tracker.commit(res_id, actual_cost=0.04)
    assert tracker.reserved_usd == 0.0
    assert tracker.spent_usd_total == 0.04
    assert tracker.spent_usd_by_role["pm"] == 0.04
    assert tracker.spent_usd_by_provider["openai"] == 0.04


def test_cloud_budget_exceeded_run_cap():
    tracker = CloudBudgetTracker(max_cloud_usd_run=0.10)

    # First reservation takes 0.08 USD
    res_1 = tracker.reserve(role="pm", provider="anthropic", model="claude-3-7-sonnet", estimated_cost=0.08)
    assert res_1

    # Second reservation requests 0.05 USD -> total 0.13 > 0.10 -> raises CloudBudgetExceededError
    with pytest.raises(CloudBudgetExceededError) as exc_info:
        tracker.reserve(role="architect", provider="anthropic", model="claude-3-7-sonnet", estimated_cost=0.05)
    assert "MAX_CLOUD_USD" in str(exc_info.value)


def test_cloud_budget_exceeded_role_cap():
    tracker = CloudBudgetTracker(
        max_cloud_usd_run=5.00,
        max_cloud_usd_role={"pm": 0.20},
    )

    with pytest.raises(CloudBudgetExceededError) as exc_info:
        tracker.reserve(role="pm", provider="anthropic", model="claude-3-7-sonnet", estimated_cost=0.25)
    assert "Role 'pm' budget exceeded" in str(exc_info.value)


def test_provider_registry_offline_fallback():
    reg = ProviderRegistry(offline=True, mock_mode=True)
    # PM role is configured as cloud provider 'anthropic' in config/roles.yaml
    provider, model, thinking, max_tokens, classes = reg.get_role_binding("pm")

    # In offline mode, PM must fall back to local Ollama
    assert provider.name == "ollama"
    assert "qwen" in model.lower()

