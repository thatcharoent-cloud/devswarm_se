"""
devswarm.platform.budget — Time and token budgets with conservative reservation.

Replaces dollar-based budgets for local inference:
  - Prices are $0.00, but cost fields are preserved in records/logs for compatibility.
  - Budgets are constrained by:
      • MAX_RUN_MINUTES: overall wall-clock time limit for the run.
      • MAX_TASK_MINUTES: wall-clock time limit per specialist task.
      • MAX_TOKENS: total token budget (in + out) across all tasks.
      • MAX_LLM_CALLS: hard cap on total inference invocations.
  - Budget reservation pattern:
      Before a task starts, it must reserve estimated tokens and time. If remaining
      budget cannot accommodate the reservation, the task is marked as budget-exceeded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import os
from typing import Any
import yaml

from devswarm.providers.base import ModelPricing

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a swarm or task budget limit is breached."""


class TaskBudgetExceededError(BudgetExceededError):
    """Raised when an individual task exceeds its allocated minutes or tokens."""


class CloudBudgetExceededError(BudgetExceededError):
    """Raised when cloud financial dollar budget is breached, halting for human intervention."""

    def __init__(self, message: str, role: str = "", provider: str = "", cost: float = 0.0) -> None:
        super().__init__(message)
        self.role = role
        self.provider = provider
        self.cost = cost


@dataclass
class CallRecord:
    """Detailed telemetry captured per LLM call."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    tokens_per_sec: float
    time_to_first_token_s: float
    load_duration_s: float
    context_size_used: int
    task_id: str = ""
    role: str = ""
    cost_usd: float = 0.0  # Kept for log format compatibility, always 0.0 for local models

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "time_to_first_token_s": round(self.time_to_first_token_s, 3),
            "load_duration_s": round(self.load_duration_s, 3),
            "context_size_used": self.context_size_used,
            "task_id": self.task_id,
            "role": self.role,
            "cost_usd": self.cost_usd,
        }


@dataclass
class SwarmBudget:
    """Manages run-level and task-level token and time limits with reservations."""

    max_run_minutes: float = 45.0
    max_task_minutes: float = 10.0
    max_tokens: int = 500_000
    max_llm_calls: int = 150

    # Conservative estimates used for reserving budget before task execution
    est_task_tokens: int = 8_000
    est_task_minutes: float = 5.0
    est_task_calls: int = 6

    start_time: float = field(default_factory=time.monotonic)
    records: list[CallRecord] = field(default_factory=list)
    active_reservations: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget_exceeded: bool = False

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_tokens_used(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_llm_calls(self) -> int:
        return len(self.records)

    @property
    def reserved_tokens(self) -> int:
        return sum(r.get("tokens", 0) for r in self.active_reservations.values())

    @property
    def reserved_calls(self) -> int:
        return sum(r.get("calls", 0) for r in self.active_reservations.values())

    def can_reserve(
        self,
        tokens: int | None = None,
        minutes: float | None = None,
        calls: int | None = None,
    ) -> bool:
        """Check whether budget allows reserving resources for a new task."""
        tokens_req = tokens if tokens is not None else self.est_task_tokens
        minutes_req = minutes if minutes is not None else self.est_task_minutes
        calls_req = calls if calls is not None else self.est_task_calls

        # Check token cap
        projected_tokens = self.total_tokens_used + self.reserved_tokens + tokens_req
        if projected_tokens > self.max_tokens:
            logger.warning(
                "Cannot reserve task: projected tokens %d exceeds max %d",
                projected_tokens, self.max_tokens,
            )
            return False

        # Check call cap
        projected_calls = self.total_llm_calls + self.reserved_calls + calls_req
        if projected_calls > self.max_llm_calls:
            logger.warning(
                "Cannot reserve task: projected calls %d exceeds max %d",
                projected_calls, self.max_llm_calls,
            )
            return False

        # Check wall time limit
        projected_minutes = self.elapsed_minutes + minutes_req
        if projected_minutes > self.max_run_minutes:
            logger.warning(
                "Cannot reserve task: projected minutes %.1f exceeds max %.1f",
                projected_minutes, self.max_run_minutes,
            )
            return False

        return True

    def reserve(
        self,
        task_id: str,
        tokens: int | None = None,
        minutes: float | None = None,
        calls: int | None = None,
    ) -> bool:
        """Reserve budget for task_id. Returns True if reserved, False if budget exceeded."""
        if not self.can_reserve(tokens=tokens, minutes=minutes, calls=calls):
            self.budget_exceeded = True
            return False

        self.active_reservations[task_id] = {
            "tokens": tokens if tokens is not None else self.est_task_tokens,
            "minutes": minutes if minutes is not None else self.est_task_minutes,
            "calls": calls if calls is not None else self.est_task_calls,
            "reserved_at": time.monotonic(),
        }
        return True

    def release(self, task_id: str) -> None:
        """Release reservation for task_id once the task completes or fails."""
        self.active_reservations.pop(task_id, None)

    def record_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokens_per_sec: float,
        time_to_first_token_s: float,
        load_duration_s: float,
        context_size_used: int,
        task_id: str = "",
        role: str = "",
    ) -> CallRecord:
        """Record an executed LLM call and check overall budget breaches."""
        rec = CallRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
            time_to_first_token_s=time_to_first_token_s,
            load_duration_s=load_duration_s,
            context_size_used=context_size_used,
            task_id=task_id,
            role=role,
            cost_usd=0.0,
        )
        self.records.append(rec)

        if self.total_tokens_used >= self.max_tokens:
            self.budget_exceeded = True
            logger.warning("Max tokens reached: %d / %d", self.total_tokens_used, self.max_tokens)

        if self.total_llm_calls >= self.max_llm_calls:
            self.budget_exceeded = True
            logger.warning("Max LLM calls reached: %d / %d", self.total_llm_calls, self.max_llm_calls)

        return rec

    def check_run_deadline(self) -> None:
        """Raise BudgetExceededError if overall run wall time has elapsed."""
        if self.elapsed_minutes > self.max_run_minutes:
            self.budget_exceeded = True
            raise BudgetExceededError(
                f"MAX_RUN_MINUTES ({self.max_run_minutes:.1f}m) exceeded: elapsed {self.elapsed_minutes:.1f}m"
            )

    def check_task_deadline(self, task_start_time: float, task_id: str = "") -> None:
        """Raise TaskBudgetExceededError if an individual task ran over MAX_TASK_MINUTES."""
        task_elapsed_minutes = (time.monotonic() - task_start_time) / 60.0
        if task_elapsed_minutes > self.max_task_minutes:
            raise TaskBudgetExceededError(
                f"Task '{task_id}' exceeded MAX_TASK_MINUTES ({self.max_task_minutes:.1f}m): took {task_elapsed_minutes:.1f}m"
            )

    def summary(self) -> dict[str, Any]:
        """Structured summary suitable for logs and manifests."""
        avg_tps = (
            sum(r.tokens_per_sec for r in self.records) / len(self.records)
            if self.records else 0.0
        )
        return {
            "elapsed_minutes": round(self.elapsed_minutes, 2),
            "max_run_minutes": self.max_run_minutes,
            "max_task_minutes": self.max_task_minutes,
            "total_tokens_used": self.total_tokens_used,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "max_tokens": self.max_tokens,
            "total_llm_calls": self.total_llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "avg_tokens_per_sec": round(avg_tps, 2),
            "budget_exceeded": self.budget_exceeded,
            "total_cost_usd": 0.0,
            "call_count": len(self.records),
        }


@dataclass
class CloudBudgetTracker:
    """Tracks and enforces cloud financial budgets (USD) per run, provider, and role."""

    max_cloud_usd_run: float = 5.00
    max_cloud_usd_provider: dict[str, float] = field(default_factory=dict)
    max_cloud_usd_role: dict[str, float] = field(default_factory=dict)
    pricing_table: dict[str, ModelPricing] = field(default_factory=dict)

    # State tracking
    spent_usd_total: float = 0.0
    spent_usd_by_provider: dict[str, float] = field(default_factory=dict)
    spent_usd_by_role: dict[str, float] = field(default_factory=dict)
    spent_usd_by_model: dict[str, float] = field(default_factory=dict)

    active_reservations: dict[str, dict[str, Any]] = field(default_factory=dict)
    _reservation_counter: int = 0

    def __post_init__(self) -> None:
        # Load environment override for max run budget
        env_run = os.getenv("MAX_CLOUD_USD")
        if env_run:
            try:
                self.max_cloud_usd_run = float(env_run)
            except ValueError:
                pass

    @classmethod
    def from_config(cls, providers_config_path: str = "config/providers.yaml", roles_config_path: str = "config/roles.yaml") -> CloudBudgetTracker:
        """Instantiate tracker populated from YAML configurations."""
        prices: dict[str, ModelPricing] = {}
        prov_file = os.path.expanduser(providers_config_path)
        if os.path.exists(prov_file):
            try:
                with open(prov_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for m_id, m_cfg in data.get("models", {}).items():
                    p_cfg = m_cfg.get("pricing", {})
                    prices[m_id] = ModelPricing(
                        input_per_million=float(p_cfg.get("input_per_million", 0.0)),
                        output_per_million=float(p_cfg.get("output_per_million", 0.0)),
                    )
            except Exception as exc:
                logger.warning("Failed loading pricing from %s: %s", providers_config_path, exc)

        role_budgets: dict[str, float] = {}
        role_file = os.path.expanduser(roles_config_path)
        if os.path.exists(role_file):
            try:
                with open(role_file, "r", encoding="utf-8") as f:
                    rdata = yaml.safe_load(f) or {}
                for r_name, r_cfg in rdata.get("roles", {}).items():
                    if "budget_usd" in r_cfg:
                        role_budgets[r_name] = float(r_cfg["budget_usd"])
            except Exception as exc:
                logger.warning("Failed loading role budgets from %s: %s", roles_config_path, exc)

        return cls(pricing_table=prices, max_cloud_usd_role=role_budgets)

    def get_pricing(self, model: str) -> ModelPricing:
        """Lookup pricing for model; check env overrides or use conservative fallback."""
        # 1. Environment overrides: PRICE_INPUT_<MODEL> and PRICE_OUTPUT_<MODEL>
        clean_model_key = model.upper().replace("-", "_").replace(".", "_").replace(":", "_")
        env_in = os.getenv(f"PRICE_INPUT_{clean_model_key}")
        env_out = os.getenv(f"PRICE_OUTPUT_{clean_model_key}")
        if env_in and env_out:
            try:
                return ModelPricing(input_per_million=float(env_in), output_per_million=float(env_out))
            except ValueError:
                pass

        if model in self.pricing_table:
            return self.pricing_table[model]

        # Conservative fallback for unknown cloud model
        logger.warning("Unknown pricing for model '%s'. Using conservative fallback ($3.00 in / $15.00 out per 1M).", model)
        return ModelPricing(input_per_million=3.00, output_per_million=15.00)

    def estimate_cost(self, model: str, input_tokens: int, max_tokens: int = 2048) -> float:
        """Estimate maximum possible cost for a call before execution."""
        pricing = self.get_pricing(model)
        return pricing.calculate_cost(input_tokens, max_tokens)

    @property
    def reserved_usd(self) -> float:
        return sum(r.get("cost", 0.0) for r in self.active_reservations.values())

    def reserve(self, role: str, provider: str, model: str, estimated_cost: float) -> str:
        """Reserve estimated cost against run, provider, and role budgets."""
        # Free local provider
        if provider.lower() == "ollama" or estimated_cost == 0.0:
            return "local_free"

        # Check total run budget
        projected_total = self.spent_usd_total + self.reserved_usd + estimated_cost
        if projected_total > self.max_cloud_usd_run:
            raise CloudBudgetExceededError(
                f"Cloud budget exceeded: projected total ${projected_total:.4f} > MAX_CLOUD_USD (${self.max_cloud_usd_run:.2f})",
                role=role,
                provider=provider,
                cost=projected_total,
            )

        # Check provider budget if defined
        if provider in self.max_cloud_usd_provider:
            prov_spent = self.spent_usd_by_provider.get(provider, 0.0)
            prov_reserved = sum(r["cost"] for r in self.active_reservations.values() if r["provider"] == provider)
            if prov_spent + prov_reserved + estimated_cost > self.max_cloud_usd_provider[provider]:
                raise CloudBudgetExceededError(
                    f"Provider '{provider}' budget exceeded: projected ${prov_spent + prov_reserved + estimated_cost:.4f} > cap ${self.max_cloud_usd_provider[provider]:.2f}",
                    role=role,
                    provider=provider,
                )

        # Check role budget if defined
        if role in self.max_cloud_usd_role:
            role_spent = self.spent_usd_by_role.get(role, 0.0)
            role_reserved = sum(r["cost"] for r in self.active_reservations.values() if r["role"] == role)
            if role_spent + role_reserved + estimated_cost > self.max_cloud_usd_role[role]:
                raise CloudBudgetExceededError(
                    f"Role '{role}' budget exceeded: projected ${role_spent + role_reserved + estimated_cost:.4f} > cap ${self.max_cloud_usd_role[role]:.2f}",
                    role=role,
                    provider=provider,
                )

        self._reservation_counter += 1
        res_id = f"res_{self._reservation_counter}"
        self.active_reservations[res_id] = {
            "role": role,
            "provider": provider,
            "model": model,
            "cost": estimated_cost,
        }
        return res_id

    def commit(self, res_id: str, actual_cost: float) -> None:
        """Commit actual cost from executed call and release reservation."""
        res = self.active_reservations.pop(res_id, None)
        if not res:
            return

        role = res["role"]
        provider = res["provider"]
        model = res["model"]

        self.spent_usd_total += actual_cost
        self.spent_usd_by_provider[provider] = self.spent_usd_by_provider.get(provider, 0.0) + actual_cost
        self.spent_usd_by_role[role] = self.spent_usd_by_role.get(role, 0.0) + actual_cost
        self.spent_usd_by_model[model] = self.spent_usd_by_model.get(model, 0.0) + actual_cost

    def release(self, res_id: str) -> None:
        """Release reservation if call aborted without execution."""
        self.active_reservations.pop(res_id, None)

    def summary(self) -> dict[str, Any]:
        """Summary of financial cloud spend."""
        return {
            "spent_usd_total": round(self.spent_usd_total, 4),
            "reserved_usd": round(self.reserved_usd, 4),
            "max_cloud_usd_run": self.max_cloud_usd_run,
            "by_provider": {k: round(v, 4) for k, v in self.spent_usd_by_provider.items()},
            "by_role": {k: round(v, 4) for k, v in self.spent_usd_by_role.items()},
            "by_model": {k: round(v, 4) for k, v in self.spent_usd_by_model.items()},
        }

