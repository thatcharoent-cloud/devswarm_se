"""
devswarm.providers.cost_estimator — Pre-run cloud cost estimation.

Calculates token estimates and projected dollar costs for PM, Architect, QA,
and Reviewer roles before any API calls are executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devswarm.platform.budget import CloudBudgetTracker
from devswarm.providers.registry import ProviderRegistry


@dataclass
class RoleCostEstimate:
    role: str
    provider: str
    model: str
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float


class CostEstimator:
    """Estimates cloud cost for a given goal across configured specialist roles."""

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        tracker: CloudBudgetTracker | None = None,
    ) -> None:
        self.registry = registry or ProviderRegistry()
        self.tracker = tracker or CloudBudgetTracker.from_config()

        # Typical token heuristics for hybrid swarm stages
        self.token_heuristics: dict[str, tuple[int, int]] = {
            "pm": (3000, 1500),         # Goal, repo map, clarification -> spec.md
            "architect": (4500, 2000),  # Spec, repo map, ADRs -> architecture.md + stubs
            "qa": (4000, 2500),         # Spec, contracts -> test-first suite
            "reviewer": (3500, 800),    # Spec, diffs, gate outputs -> PR summary
            "backend": (3000, 1000),    # Local coder: $0.00
            "tester": (3000, 1000),     # Local tester: $0.00
        }

    def estimate_run(self, goal: str) -> list[RoleCostEstimate]:
        """Estimate costs per role for the specified goal."""
        estimates: list[RoleCostEstimate] = []

        # Adjust base tokens slightly if goal is long
        goal_extra_tokens = len(goal) // 4

        hybrid_roles = ["pm", "architect", "qa", "backend", "tester", "reviewer"]
        for role_name in hybrid_roles:
            r_cfg = self.registry.roles_config.get(role_name, {})
            provider_name = r_cfg.get("provider", "ollama")
            model_name = r_cfg.get("model", "")
            if model_name == "<set model id>" or not model_name:
                model_name = "qwen2.5-coder:7b" if provider_name == "ollama" else "mock-cloud-model"

            base_in, base_out = self.token_heuristics.get(role_name, (2000, 1000))
            est_in = base_in + goal_extra_tokens
            est_out = base_out

            if provider_name.lower() == "ollama":
                cost = 0.0
            else:
                pricing = self.tracker.get_pricing(model_name)
                cost = pricing.calculate_cost(est_in, est_out)

            estimates.append(
                RoleCostEstimate(
                    role=role_name,
                    provider=provider_name,
                    model=model_name,
                    est_input_tokens=est_in,
                    est_output_tokens=est_out,
                    est_cost_usd=cost,
                )
            )

        return estimates

    def format_markdown(self, goal: str, estimates: list[RoleCostEstimate]) -> str:
        """Render estimate as formatted markdown table."""
        total_usd = sum(e.est_cost_usd for e in estimates)
        max_budget = self.tracker.max_cloud_usd_run

        lines = [
            f"# 💰 Pre-Flight Cost Estimate for Swarm Run",
            f"**Goal:** `{goal}`",
            "",
            "| Role | Provider | Model | Est. Input | Est. Output | Est. Cost (USD) |",
            "|---|---|---|:---:|:---:|:---:|",
        ]

        for e in estimates:
            cost_str = "$0.00 (Local)" if e.est_cost_usd == 0.0 else f"${e.est_cost_usd:.4f}"
            lines.append(
                f"| `{e.role}` | `{e.provider}` | `{e.model}` | "
                f"{e.est_input_tokens:,} | {e.est_output_tokens:,} | {cost_str} |"
            )

        lines.append("")
        lines.append(f"**Total Estimated Cloud Cost:** **${total_usd:.4f} USD**")
        lines.append(f"**Max Configured Budget (MAX_CLOUD_USD):** **${max_budget:.2f} USD**")

        if total_usd > max_budget:
            lines.append("")
            lines.append(f"⚠️ **Warning:** Estimated cost (${total_usd:.4f}) exceeds configured budget limit (${max_budget:.2f}).")
        else:
            lines.append("")
            lines.append("✅ **Within Budget:** Projected costs are safely within budget boundaries.")

        return "\n".join(lines)
