"""
tests/devswarm/test_reviewer.py — Unit tests for automated reviewer and model swapping.
"""

from pathlib import Path
import json
import pytest

from devswarm.reviewer import SwarmReviewer, ReviewReport
from devswarm.llm.mock_llm import MockLLM
from devswarm.platform.budget import SwarmBudget
from devswarm.platform.events import EventLog


def test_review_report_schema_validation() -> None:
    valid_data = {
        "verdict": "APPROVE",
        "summary": "Looks good",
        "strengths": ["Clean code"],
        "concerns": [],
        "security_assessment": "Safe",
        "confidence_score": 0.95,
    }
    report = ReviewReport.model_validate(valid_data)
    assert report.verdict == "APPROVE"
    assert report.confidence_score == 0.95


@pytest.mark.asyncio
async def test_reviewer_generates_report(tmp_path: Path) -> None:
    mock_llm = MockLLM()
    # Default responder in MockLLM already produces valid ReviewReport when verdict is in props
    budget = SwarmBudget()
    event_log = EventLog(run_dir=tmp_path, run_id="test_run")

    reviewer = SwarmReviewer(
        client=mock_llm,
        repo_path=tmp_path,
        budget=budget,
        event_log=event_log,
        reviewer_model="qwen3.5:9b",
        default_model="qwen3.5:9b",
    )

    report = await reviewer.review_run(
        goal="Add safe divide",
        base_commit="HEAD~1",
        integration_branch="swarm/test_run",
        task_summaries=[{"id": "task-1", "role": "backend", "title": "Divide implementation", "summary": "Done"}],
        current_active_model="qwen3.5:9b",
    )

    assert report.verdict in ("APPROVE", "REQUEST_CHANGES", "NEEDS_HUMAN")
    assert report.confidence_score > 0.0
    assert budget.summary()["total_llm_calls"] >= 1


@pytest.mark.asyncio
async def test_reviewer_model_swap(tmp_path: Path) -> None:
    mock_llm = MockLLM()
    budget = SwarmBudget()
    event_log = EventLog(run_dir=tmp_path, run_id="test_run")

    # Configure a distinct REVIEWER_MODEL
    reviewer = SwarmReviewer(
        client=mock_llm,
        repo_path=tmp_path,
        budget=budget,
        event_log=event_log,
        reviewer_model="qwen2.5-coder:14b",
        default_model="qwen3.5:9b",
    )

    report = await reviewer.review_run(
        goal="Model swap test",
        base_commit="HEAD~1",
        integration_branch="swarm/test_run",
        task_summaries=[],
        current_active_model="qwen3.5:9b",  # Active is different from reviewer_model
    )

    assert report.verdict == "APPROVE"
    # Verify unload_model was invoked for the previous model
    unloads = [c for c in mock_llm.calls if c.keep_alive == 0 and c.model == "qwen3.5:9b"]
    assert len(unloads) == 1

