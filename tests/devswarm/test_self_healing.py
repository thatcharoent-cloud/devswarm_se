"""
tests/devswarm/test_self_healing.py — Tests for Phase D: Self-healing, escalation, and two-pool scheduling.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest

from devswarm.gates import GateResult, GateSuiteResult
from devswarm.platform.budget import CloudBudgetTracker
from devswarm.platform.outbound_filter import OutboundFilter
from devswarm.platform.sandbox import MockSandbox
from devswarm.platform.taskqueue import SwarmTask, TaskQueue, TaskState
from devswarm.providers.base import ChatMessage, ProviderResponse, TokenUsage
from devswarm.providers.mock_provider import MockProvider
from devswarm.scheduler import TwoPoolScheduler
from devswarm.self_healing import (
    CloudEscalationHandler,
    TaskAttempt,
    TaskAttemptManager,
    extract_failure_diagnostics,
)
from devswarm.traceability import TraceabilityMatrix, TraceabilityRecord


def test_extract_failure_diagnostics_pytest_failure() -> None:
    """Test pure-Python failure diagnostic extraction from pytest output."""
    stdout_text = (
        "============================= test session starts ==============================\n"
        "collecting ... collected 2 items\n\n"
        "tests/test_math.py .F                                                    [100%]\n\n"
        "=================================== FAILURES ===================================\n"
        "__________________________________ test_divide _________________________________\n\n"
        "    def test_divide():\n"
        ">       assert divide(10, 2) == 5\n"
        "E       AssertionError: assert 4 == 5\n"
        "E        +  where 4 = divide(10, 2)\n\n"
        "tests/test_math.py:15: AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_math.py::test_divide - AssertionError: assert 4 == 5\n"
        "======================== 1 failed, 1 passed in 0.12s ===========================\n"
    )

    gate_result = GateSuiteResult(
        passed=False,
        results={
            "compile": GateResult(name="compile", stage="compile", passed=True),
            "lint": GateResult(name="lint", stage="lint", passed=True),
            "test": GateResult(
                name="test",
                stage="test",
                passed=False,
                stdout=stdout_text,
                stderr="",
                exit_code=1,
            ),
        },
        summary="Quality Gates FAILED (test)",
    )

    diagnostics = extract_failure_diagnostics(gate_result)
    assert "<untrusted_gate_feedback>" in diagnostics
    assert "FAILED tests/test_math.py::test_divide" in diagnostics
    assert "AssertionError: assert 4 == 5" in diagnostics
    assert "tests/test_math.py:15" in diagnostics


def test_extract_failure_diagnostics_compile_failure() -> None:
    """Test syntax and compilation error extraction without LLM calls."""
    gate_result = GateSuiteResult(
        passed=False,
        results={
            "compile": GateResult(
                name="compile",
                stage="compile",
                passed=False,
                stdout="",
                stderr="SyntaxError: invalid syntax on line 42 in calc.py",
                exit_code=1,
            ),
        },
        summary="Quality Gates FAILED (compile)",
    )

    diagnostics = extract_failure_diagnostics(gate_result)
    assert "SyntaxError: invalid syntax on line 42 in calc.py" in diagnostics


def test_task_attempt_manager_append_only() -> None:
    """Test that TaskAttemptManager preserves full history across repeated attempts."""
    mgr = TaskAttemptManager()
    assert mgr.attempt_count("task-1") == 0

    att1 = mgr.record_attempt(
        task_id="task-1",
        files_modified=["calc.py"],
        gate_passed=False,
        gate_summary="Test gate failed",
        diagnostics="AssertionError: 2 != 3",
    )
    assert att1.attempt_number == 1
    assert mgr.attempt_count("task-1") == 1

    att2 = mgr.record_attempt(
        task_id="task-1",
        files_modified=["calc.py", "helper.py"],
        gate_passed=True,
        gate_summary="Gates PASSED",
        diagnostics="",
    )
    assert att2.attempt_number == 2
    assert mgr.attempt_count("task-1") == 2

    history = mgr.get_attempts("task-1")
    assert len(history) == 2
    assert history[0].gate_passed is False
    assert history[1].gate_passed is True
    assert history[0].diagnostics == "AssertionError: 2 != 3"


@pytest.mark.asyncio
async def test_cloud_escalation_handler_disabled() -> None:
    """Test that escalation returns None when not enabled."""
    handler = CloudEscalationHandler(enabled=False)
    advice = await handler.request_debug_advice(
        task_id="task-1",
        task_title="Fix divide",
        write_scope=["calc.py"],
        diagnostics="AssertionError",
    )
    assert advice is None


@pytest.mark.asyncio
async def test_cloud_escalation_handler_enabled(tmp_path: Path) -> None:
    """Test that escalation calls cloud debugger, reserves budget, and returns advice."""
    mock_prov = MockProvider(name="anthropic")
    mock_prov.add_response(
        ProviderResponse(
            content="Check line 12: change return a * b to return a / b.",
            model="claude-3-5-haiku",
            usage=TokenUsage(input_tokens=150, output_tokens=30),
            cost_usd=0.0012,
        )
    )

    budget = CloudBudgetTracker()
    outbound = OutboundFilter(swarm_dir=tmp_path)
    # Acknowledge cloud call
    ack_file = tmp_path / "cloud_ack.json"
    ack_file.write_text(json.dumps({"acknowledged": True}), encoding="utf-8")

    handler = CloudEscalationHandler(
        enabled=True,
        provider=mock_prov,
        model="claude-3-5-haiku",
        outbound_filter=outbound,
        budget_tracker=budget,
    )

    advice = await handler.request_debug_advice(
        task_id="task-1",
        task_title="Fix divide by zero",
        write_scope=["calc.py"],
        diagnostics="ZeroDivisionError in line 12",
        recent_code="def divide(a, b): return a / b",
        run_id="run_test",
    )

    assert advice is not None
    assert "<cloud_debugger_guidance>" in advice
    assert "Check line 12" in advice
    assert budget.spent_usd_total > 0


@pytest.mark.asyncio
async def test_two_pool_scheduler_concurrency(tmp_path: Path) -> None:
    """Test two-pool scheduler enforces concurrency 1 for local and allows concurrency N for cloud."""
    db_path = tmp_path / "tasks.db"
    q = TaskQueue(db_path=db_path)

    t1 = SwarmTask(id="task-local-1", title="Local Coder 1", role="backend", description="Code A", write_scope=["a.py"])
    t2 = SwarmTask(id="task-local-2", title="Local Coder 2", role="backend", description="Code B", write_scope=["b.py"])
    t3 = SwarmTask(id="task-cloud-1", title="Cloud Spec 1", role="pm", description="Spec", write_scope=["spec.md"])
    t4 = SwarmTask(id="task-cloud-2", title="Cloud Arch 2", role="architect", description="Arch", write_scope=["arch.md"])
    q.add_tasks([t1, t2, t3, t4])

    scheduler = TwoPoolScheduler(
        task_queue=q,
        local_concurrency=1,
        cloud_concurrency=2,
    )

    assert scheduler.determine_pool("backend", "ollama") == "local"
    assert scheduler.determine_pool("pm", "anthropic") == "cloud"

    role_providers = {
        "backend": "ollama",
        "pm": "anthropic",
        "architect": "anthropic",
    }
    local_cands, cloud_cands = scheduler.get_ready_tasks_by_pool(role_providers)
    assert len(local_cands) == 2
    assert len(cloud_cands) == 2

    # Verify execution under pool semaphore
    async def mock_executor(task: SwarmTask):
        await asyncio.sleep(0.01)
        return True

    res = await scheduler.execute_task_with_pool(t1, "ollama", mock_executor)
    assert res.success is True
    assert res.pool == "local"


def test_pr_generator_with_traceability_and_cloud_telemetry(tmp_path: Path) -> None:
    """Test that PRGenerator renders traceability, cost accounting, and cloud data policy."""
    from devswarm.pr import PRGenerator
    from devswarm.reviewer import ReviewReport

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    gate_res = GateSuiteResult(passed=True, summary="All gates passed")
    review_rep = ReviewReport(
        verdict="APPROVE",
        confidence_score=0.95,
        summary="Well architected implementation.",
        security_assessment="No vulnerabilities detected.",
        strengths=["Modular functions", "Full test coverage"],
        concerns=[],
    )

    # Build traceability matrix
    record = TraceabilityRecord(
        ac_id="AC-1",
        description="Multiply two integers",
        tasks=["task-1"],
        tests=["tests/test_calc.py::test_multiply"],
        status="PASS",
    )
    matrix = TraceabilityMatrix(records={"AC-1": record})
    trace_md = matrix.render_matrix_markdown()

    role_telemetry = [
        {
            "role": "pm",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "input_tokens": 1200,
            "output_tokens": 400,
            "cost_usd": 0.0096,
        },
        {
            "role": "backend",
            "provider": "ollama",
            "model": "qwen3.5:9b",
            "input_tokens": 2500,
            "output_tokens": 800,
            "cost_usd": 0.0,
        },
    ]

    cloud_accounting = {
        "spent_usd_total": 0.0096,
        "content_classes": ["goal", "spec", "architecture"],
        "bytes_sent": 8420,
    }

    sources_cited = [
        {"title": "Python Math Lib", "url": "https://docs.python.org/3/library/math.html", "status": "VERIFIED"}
    ]

    pr_file = PRGenerator.generate(
        run_id="run_phase_d",
        goal="Add multiply with tests",
        repo_path=repo_dir,
        base_commit="HEAD~1",
        integration_branch="swarm/test",
        tasks=[{"id": "task-1", "role": "backend", "title": "Implement multiply", "state": "done"}],
        gate_result=gate_res,
        review_report=review_rep,
        budget_summary={"elapsed_minutes": 2.5, "total_llm_calls": 3, "total_input_tokens": 3700, "total_output_tokens": 1200},
        output_dir=out_dir,
        traceability_markdown=trace_md,
        role_telemetry=role_telemetry,
        cloud_accounting=cloud_accounting,
        sources_cited=sources_cited,
    )

    assert pr_file.exists()
    content = pr_file.read_text(encoding="utf-8")
    assert "Specification & Acceptance Traceability" in content
    assert "AC-1" in content
    assert "Specialist Roles & Cost Accounting" in content
    assert "claude-3-5-sonnet" in content
    assert "$0.00 (Local)" in content
    assert "Cloud Data Policy & Outbound Telemetry" in content
    assert "8,420 bytes" in content
    assert "Research Sources & Citations" in content
    assert "https://docs.python.org/3/library/math.html" in content


@pytest.mark.asyncio
async def test_orchestrator_self_healing_attempt_tracking(tmp_path: Path) -> None:
    """Test orchestrator tracks all self-healing attempts with diagnostics in TaskAttemptManager."""
    from devswarm.dev_orchestrator import DevOrchestrator
    from devswarm.llm.mock_llm import MockLLM
    import subprocess

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestUser"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=str(repo_dir), check=True)
    (repo_dir / "calc.py").write_text("def multiply(a, b): return a * b\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True)

    mock_llm = MockLLM()
    # Plan with 1 task
    plan = {
        "goal": "Test healing",
        "assumptions": [],
        "tasks": [
            {
                "id": "task-heal",
                "title": "Healing test",
                "role": "backend",
                "description": "Fix calculation",
                "acceptance_criteria": ["Works"],
                "write_scope": ["calc.py"],
                "minutes_budget": 5.0,
            }
        ],
    }
    mock_llm.add_response(content=json.dumps(plan))

    # Mock agent execution: attempt 1 produces code
    mock_llm.add_response(
        content="",
        tool_calls=[{
            "function": {
                "name": "write_file",
                "arguments": {"path": "calc.py", "content": "def multiply(a, b): return a + b\n"},
            }
        }],
    )
    mock_llm.add_response(
        content="",
        tool_calls=[{
            "function": {
                "name": "finish_task",
                "arguments": {"summary": "Done", "files_modified": ["calc.py"]},
            }
        }],
    )
    # Fix attempt 1: agent fixes code after seeing diagnostic feedback
    mock_llm.add_response(
        content="",
        tool_calls=[{
            "function": {
                "name": "write_file",
                "arguments": {"path": "calc.py", "content": "def multiply(a, b): return a * b\n"},
            }
        }],
    )
    mock_llm.add_response(
        content="",
        tool_calls=[{
            "function": {
                "name": "finish_task",
                "arguments": {"summary": "Repaired", "files_modified": ["calc.py"]},
            }
        }],
    )

    mock_sb = MockSandbox()
    # Initial gates: compile passes, lint passes, test FAILS
    mock_sb.add_result(stdout="ok", exit_code=0)
    mock_sb.add_result(stdout="clean", exit_code=0)
    mock_sb.add_result(stderr="FAILED tests/test_calc.py::test_multiply - AssertionError: 4 != 5", exit_code=1)

    # Healing attempt gates: compile passes, lint passes, test PASSES
    mock_sb.add_result(stdout="ok", exit_code=0)
    mock_sb.add_result(stdout="clean", exit_code=0)
    mock_sb.add_result(stdout="PASSED", exit_code=0)

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Test healing",
        client=mock_llm,
        sandbox=mock_sb,
        swarm_dir=swarm_dir,
        max_fix_attempts=2,
    )

    exit_code = await orch.execute_run()
    assert exit_code == 0

    # Verify attempt history in TaskAttemptManager
    attempts = orch.attempt_manager.get_attempts("task-heal")
    assert len(attempts) == 2
    assert attempts[0].gate_passed is False
    assert "FAILED tests/test_calc.py::test_multiply" in attempts[0].diagnostics
    assert attempts[1].gate_passed is True


@pytest.mark.asyncio
async def test_orchestrator_qa_post_pass(tmp_path: Path) -> None:
    """Test QA post-implementation pass generates edge-case tests and verifies them."""
    from devswarm.dev_orchestrator import DevOrchestrator
    from devswarm.agents.qa_schema import QATestSuite, QATestFile
    from devswarm.providers.mock_provider import MockProvider

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"

    mock_prov = MockProvider(name="anthropic")
    qa_suite = QATestSuite(
        test_files=[
            QATestFile(
                path="tests/test_edge_cases.py",
                content="def test_zero_multiply():\n    assert 0 * 5 == 0\n",
                ac_coverage=["AC-1"],
            )
        ],
        ac_mapping={"AC-1": ["tests/test_edge_cases.py::test_zero_multiply"]},
        summary="Added edge case tests",
    )
    mock_prov.add_response(
        ProviderResponse(
            content=qa_suite.model_dump_json(),
            model="claude-3-5-haiku",
            usage=TokenUsage(input_tokens=200, output_tokens=100),
        )
    )

    mock_sb = MockSandbox()
    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Post pass test",
        sandbox=mock_sb,
        swarm_dir=swarm_dir,
    )

    res_suite = await orch.run_qa_post_pass(qa_provider=mock_prov, model="claude-3-5-haiku")
    assert res_suite is not None
    assert (repo_dir / "tests/test_edge_cases.py").exists()
    assert "test_zero_multiply" in (repo_dir / "tests/test_edge_cases.py").read_text(encoding="utf-8")

