"""
tests/devswarm/test_pr.py — Unit tests for PR.md artifact generation.
"""

from pathlib import Path
import pytest

from devswarm.gates import GateResult, GateSuiteResult
from devswarm.pr import PRGenerator
from devswarm.reviewer import ReviewReport


def test_pr_generator_produces_valid_markdown(tmp_path: Path) -> None:
    run_id = "run_20261110T070000Z"
    goal = "Add multiply function to calc"

    gate_result = GateSuiteResult(
        passed=True,
        results={
            "compile": GateResult(name="compile", stage="compile", passed=True, duration_s=0.2),
            "test": GateResult(name="test", stage="test", passed=True, duration_s=1.5),
            "security": GateResult(name="security", stage="security", passed=True, duration_s=0.1),
        },
        duration_s=1.8,
        summary="Quality Gates PASSED in 1.80s",
        has_high_risk_changes=False,
    )

    review_report = ReviewReport(
        verdict="APPROVE",
        summary="Changes are minimal and clean.",
        strengths=["Deterministic test coverage", "Proper type annotations"],
        concerns=[],
        security_assessment="Safe and clean.",
        confidence_score=0.98,
    )

    tasks = [
        {"id": "task-1", "role": "backend", "title": "Implement multiply", "state": "done", "write_scope": ["calc.py"]},
        {"id": "task-2", "role": "tester", "title": "Add test_multiply", "state": "done", "write_scope": ["test_calc.py"]},
    ]

    budget_summary = {
        "elapsed_minutes": 2.5,
        "total_llm_calls": 6,
        "total_input_tokens": 1250,
        "total_output_tokens": 400,
    }

    pr_file = PRGenerator.generate(
        run_id=run_id,
        goal=goal,
        repo_path=tmp_path,
        base_commit="88272967",
        integration_branch=f"swarm/{run_id}",
        tasks=tasks,
        gate_result=gate_result,
        review_report=review_report,
        budget_summary=budget_summary,
        output_dir=tmp_path,
    )

    assert pr_file.exists()
    content = pr_file.read_text(encoding="utf-8")

    # Header and goal
    assert "# Pull Request: Add multiply function to calc" in content
    assert "READY FOR MERGE" in content
    assert run_id in content

    # Tables
    assert "| `task-1` | `backend` | `done` | `calc.py` | Implement multiply |" in content
    assert "| `compile` | ✅ PASSED | Yes | 0.20s |" in content

    # Reviewer
    assert "APPROVE" in content
    assert "Deterministic test coverage" in content

    # $0.00 Cost accounting
    assert "$0.000000" in content

    # Approval instructions
    assert f"python main.py approve --run-id {run_id}" in content
    assert f"python main.py reject --run-id {run_id}" in content


def test_pr_generator_high_risk_alert(tmp_path: Path) -> None:
    gate_result = GateSuiteResult(
        passed=True,
        results={},
        duration_s=0.5,
        summary="Quality Gates PASSED",
        has_high_risk_changes=True,
        high_risk_files=["src/auth/login.py", "requirements.txt"],
    )

    review_report = ReviewReport(
        verdict="NEEDS_HUMAN",
        summary="High risk files touched.",
        strengths=[],
        concerns=["Modifies security critical authentication logic."],
        security_assessment="Requires manual review.",
        confidence_score=0.85,
    )

    pr_file = PRGenerator.generate(
        run_id="run_high_risk",
        goal="Update auth",
        repo_path=tmp_path,
        base_commit="HEAD~1",
        integration_branch="swarm/run_high_risk",
        tasks=[],
        gate_result=gate_result,
        review_report=review_report,
        budget_summary={},
        output_dir=tmp_path,
    )

    content = pr_file.read_text(encoding="utf-8")
    assert "HIGH-RISK PATHS MODIFIED" in content
    assert "`src/auth/login.py`" in content
    assert "`requirements.txt`" in content
    assert "Human approval is mandatory before merge." in content

