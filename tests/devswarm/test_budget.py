"""
tests/devswarm/test_budget.py — Unit tests for local swarm time and token budgets.
"""

import time
import pytest
from devswarm.platform.budget import (
    SwarmBudget,
    BudgetExceededError,
    TaskBudgetExceededError,
)


def test_budget_reservation_success_and_release():
    budget = SwarmBudget(
        max_run_minutes=30.0,
        max_tokens=100_000,
        max_llm_calls=50,
        est_task_tokens=5_000,
        est_task_minutes=3.0,
        est_task_calls=4,
    )

    # Reserve task-1
    assert budget.reserve("task-1") is True
    assert budget.reserved_tokens == 5_000
    assert budget.reserved_calls == 4

    # Reserve task-2
    assert budget.reserve("task-2") is True
    assert budget.reserved_tokens == 10_000

    # Release task-1
    budget.release("task-1")
    assert budget.reserved_tokens == 5_000


def test_budget_reservation_exceeded():
    budget = SwarmBudget(
        max_run_minutes=10.0,
        max_tokens=15_000,
        max_llm_calls=5,
        est_task_tokens=10_000,
        est_task_minutes=5.0,
        est_task_calls=3,
    )

    # First task fits
    assert budget.reserve("task-1") is True

    # Second task would exceed max_tokens (10k + 10k > 15k) and calls (3 + 3 > 5)
    assert budget.reserve("task-2") is False
    assert budget.budget_exceeded is True


def test_record_call_and_telemetry():
    budget = SwarmBudget(max_tokens=50_000, max_llm_calls=100)

    rec = budget.record_call(
        model="qwen2.5-coder:7b",
        input_tokens=120,
        output_tokens=60,
        tokens_per_sec=32.5,
        time_to_first_token_s=0.15,
        load_duration_s=0.05,
        context_size_used=180,
        task_id="task-10",
        role="backend",
    )

    assert rec.total_tokens == 180
    assert rec.cost_usd == 0.0
    assert budget.total_tokens_used == 180
    assert budget.total_llm_calls == 1

    summary = budget.summary()
    assert summary["total_tokens_used"] == 180
    assert summary["total_cost_usd"] == 0.0
    assert summary["avg_tokens_per_sec"] == 32.5


def test_run_deadline_check():
    budget = SwarmBudget(max_run_minutes=0.001)  # 60ms
    time.sleep(0.08)

    with pytest.raises(BudgetExceededError) as exc_info:
        budget.check_run_deadline()

    assert "MAX_RUN_MINUTES" in str(exc_info.value)
    assert budget.budget_exceeded is True


def test_task_deadline_check():
    budget = SwarmBudget(max_task_minutes=0.001)  # 60ms
    task_start = time.monotonic() - 0.1

    with pytest.raises(TaskBudgetExceededError) as exc_info:
        budget.check_task_deadline(task_start, task_id="task-slow")

    assert "task-slow" in str(exc_info.value)

