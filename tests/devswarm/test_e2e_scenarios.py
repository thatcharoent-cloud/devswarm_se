"""
tests/devswarm/test_e2e_scenarios.py — End-to-end integration and resilience scenarios.
"""

from pathlib import Path
import json
import shutil
import subprocess
import pytest

from devswarm.dev_orchestrator import DevOrchestrator
from devswarm.approval import ApprovalManager
from devswarm.llm.mock_llm import MockLLM
from devswarm.platform.guards import KillSwitch, KillSwitchTriggered
from devswarm.platform.sandbox import MockSandbox
from devswarm.platform.taskqueue import TaskState


def setup_toy_repo_copy(tmp_path: Path) -> Path:
    """Clone/copy the toy repository into tmp_path to test real git operations safely."""
    src_repo = Path("tests/fixtures/toy_repo").resolve()
    dest_repo = tmp_path / "toy_repo"
    shutil.copytree(src_repo, dest_repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Ensure clean git state on branch main (initialize git if not present)
    if not (dest_repo / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(dest_repo), check=True, capture_output=True)
        subprocess.run(["git", "branch", "-M", "main"], cwd=str(dest_repo), check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=str(dest_repo), check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@test.com", "commit", "-m", "Initial commit"],
            cwd=str(dest_repo),
            check=True,
            capture_output=True,
        )

    subprocess.run(["git", "checkout", "main"], cwd=str(dest_repo), check=True, capture_output=True)
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(dest_repo), check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=str(dest_repo), check=True, capture_output=True)
    return dest_repo


@pytest.mark.asyncio
async def test_e2e_happy_path_with_approval(tmp_path: Path) -> None:
    repo_dir = setup_toy_repo_copy(tmp_path)
    swarm_dir = tmp_path / ".swarm"

    mock_llm = MockLLM()
    # 1. Plan response
    plan = {
        "goal": "Add power function and unit tests",
        "assumptions": ["Use standard exponentiation **"],
        "tasks": [
            {
                "id": "task-01-power-backend",
                "title": "Implement power in calc.py",
                "role": "backend",
                "description": "Add def power(a, b) to calc.py",
                "acceptance_criteria": ["power(2, 3) == 8"],
                "depends_on": [],
                "write_scope": ["calc.py"],
                "minutes_budget": 5.0,
            },
            {
                "id": "task-02-power-tester",
                "title": "Add test_power in test_calc.py",
                "role": "tester",
                "description": "Add test_power covering 2^3 == 8 and edge cases",
                "acceptance_criteria": ["test_power passes"],
                "depends_on": ["task-01-power-backend"],
                "write_scope": ["test_calc.py"],
                "minutes_budget": 5.0,
            },
        ],
    }

    # Task 1 actions: edit calc.py, finish task
    t1_edit = {
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "edit_file",
                "arguments": {
                    "path": "calc.py",
                    "old_str": "def divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError(\"division by zero\")\n    return a / b",
                    "new_str": "def divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError(\"division by zero\")\n    return a / b\n\ndef power(a: float, b: float) -> float:\n    return a ** b",
                },
            }
        }],
    }
    t1_finish = {
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "finish_task",
                "arguments": {"summary": "Added power function", "files_modified": ["calc.py"]},
            }
        }],
    }

    # Task 2 actions: edit test_calc.py, finish task
    t2_edit = {
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "edit_file",
                "arguments": {
                    "path": "test_calc.py",
                    "old_str": "from calc import add, subtract, multiply, divide",
                    "new_str": "from calc import add, subtract, multiply, divide, power\n\ndef test_power():\n    assert power(2, 3) == 8\n    assert power(5, 0) == 1",
                },
            }
        }],
    }
    t2_finish = {
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "finish_task",
                "arguments": {"summary": "Added test_power", "files_modified": ["test_calc.py"]},
            }
        }],
    }

    # Reviewer report
    review_resp = {
        "verdict": "APPROVE",
        "summary": "Implementation and tests are clean and deterministic.",
        "strengths": ["Power function correctly typed", "Unit tests assert boundary 5^0 == 1"],
        "concerns": [],
        "security_assessment": "Zero secrets, no eval or dangerous builtins.",
        "confidence_score": 0.99,
    }

    mock_llm.set_scripted_responses([
        json.dumps(plan),
        t1_edit,
        t1_finish,
        t2_edit,
        t2_finish,
        json.dumps(review_resp),
    ])

    mock_sb = MockSandbox()

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Add power function and unit tests",
        client=mock_llm,
        sandbox=mock_sb,
        swarm_dir=swarm_dir,
    )

    exit_code = await orch.execute_run()
    assert exit_code == 0

    # Verify both tasks marked DONE
    status = orch.get_status()
    assert status["total_tasks"] == 2
    assert status["done_tasks"] == 2
    assert status["failed_tasks"] == 0

    # Verify PR.md artifact was created
    pr_path = swarm_dir / "runs" / orch.run_id / "PR.md"
    assert pr_path.exists()
    assert "Add power function and unit tests" in pr_path.read_text(encoding="utf-8")

    # Now simulate Human Product Owner approval
    approval_mgr = ApprovalManager(swarm_dir=swarm_dir)
    approved = approval_mgr.approve_run(run_id=orch.run_id, default_branch="main")
    assert approved is True

    # Verify main branch in repo_dir now contains power function and test_power
    calc_text = (repo_dir / "calc.py").read_text(encoding="utf-8")
    assert "def power(a: float, b: float) -> float:" in calc_text

    test_calc_text = (repo_dir / "test_calc.py").read_text(encoding="utf-8")
    assert "def test_power():" in test_calc_text


@pytest.mark.asyncio
async def test_e2e_kill_switch_and_resume(tmp_path: Path) -> None:
    repo_dir = setup_toy_repo_copy(tmp_path)
    swarm_dir = tmp_path / ".swarm"

    mock_llm = MockLLM()
    plan = {
        "goal": "Resilience test",
        "assumptions": [],
        "tasks": [
            {
                "id": "task-resilience-1",
                "title": "Task 1",
                "role": "backend",
                "description": "Task 1",
                "acceptance_criteria": ["None"],
                "write_scope": ["calc.py"],
                "minutes_budget": 5.0,
            }
        ],
    }
    mock_llm.add_response(content=json.dumps(plan))

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Resilience test",
        client=mock_llm,
        sandbox=MockSandbox(),
        swarm_dir=swarm_dir,
    )

    # Trigger kill switch before run
    orch.kill_switch.trigger("Operator test halt")

    with pytest.raises(KillSwitchTriggered):
        await orch.execute_run()

    # Reset kill switch
    orch.kill_switch.remove()
    assert not orch.kill_switch.is_triggered()


@pytest.mark.asyncio
async def test_e2e_hybrid_full_pipeline(tmp_path: Path) -> None:
    """Full End-to-End Hybrid Swarm Lifecycle: PM -> Gate 1 -> Arch -> Gate 2 -> QA -> Dev -> PR -> Merge."""
    from devswarm.pipeline import PipelineController, GatePausedException
    from devswarm.agents.pm_schema import PMSpecification, AcceptanceCriterion
    from devswarm.agents.architect_schema import ArchitectureDesign, InterfaceContract
    from devswarm.agents.qa_schema import QATestSuite, QATestFile
    from devswarm.platform.events import EventLog
    from devswarm.providers.mock_provider import MockProvider
    from devswarm.providers.base import ProviderResponse, TokenUsage
    from devswarm.traceability import TraceabilityMatrix
    from devswarm.pr import PRGenerator
    from devswarm.reviewer import ReviewReport
    from devswarm.gates import GateSuiteResult

    repo_dir = setup_toy_repo_copy(tmp_path)
    swarm_dir = tmp_path / ".swarm"
    run_dir = swarm_dir / "runs" / "run_e2e_hybrid"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(run_dir=run_dir, run_id="run_e2e_hybrid")
    sandbox = MockSandbox()

    # 1. PM Agent generates spec
    pm_spec = PMSpecification(
        problem_statement="Add modulo operator to calc.py",
        user_stories=["As a user, I want modulo calculation"],
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="modulo(10, 3) == 1", observable_result="Returns 1", verification_method="pytest")
        ],
        non_goals=["No complex numbers"],
        assumptions=["Standard Python % operator"],
    )
    mock_prov = MockProvider(name="anthropic")
    mock_prov.add_response(content=pm_spec.model_dump_json(), model="claude-3-5-sonnet")

    controller = PipelineController(
        run_id="run_e2e_hybrid",
        repo_path=repo_dir,
        artifacts_dir=artifacts_dir,
        event_log=event_log,
        sandbox=sandbox,
    )

    spec = await controller.run_pm_stage(mock_prov, "claude-3-5-sonnet", "Add modulo", "", [])
    assert (artifacts_dir / "spec.md").exists()

    # 2. Gate 1 check: pauses before approval
    with pytest.raises(GatePausedException):
        controller.check_gate_1_spec_approval()

    # Human approves Gate 1
    controller.approve_spec("Human PO approved spec")
    assert controller.check_gate_1_spec_approval() is True

    # 3. Architect Agent generates architecture and interface contract
    arch_design = ArchitectureDesign(
        stack_decision="Python 3.11 stdlib math",
        stack_justification="Minimal overhead, zero third-party dependencies",
        alternatives_considered=["numpy: rejected due to heavyweight dependency"],
        folder_structure=["calc.py", "tests/"],
        components=["calc: arithmetic functions"],
        contracts=[
            InterfaceContract(
                path="calc.py",
                content=(
                    "def modulo(a: int, b: int) -> int:\n"
                    "    '''Calculate remainder.'''\n"
                    "    raise NotImplementedError('Unimplemented stub')\n"
                ),
            )
        ],
    )
    mock_prov.add_response(content=arch_design.model_dump_json(), model="claude-3-5-sonnet")
    design = await controller.run_architect_stage(mock_prov, "claude-3-5-sonnet", "Add modulo", "")
    assert (artifacts_dir / "architecture.md").exists()
    assert (repo_dir / "calc.py").exists()

    # 4. Gate 2 check: pauses before approval
    with pytest.raises(GatePausedException):
        controller.check_gate_2_design_approval()

    # Human approves Gate 2
    controller.approve_design("Lead Architect approved design")
    assert controller.check_gate_2_design_approval() is True

    # 5. QA Agent writes test-first suite
    qa_suite = QATestSuite(
        test_files=[
            QATestFile(
                path="tests/test_modulo.py",
                content="from calc import modulo\ndef test_modulo():\n    assert modulo(10, 3) == 1\n",
                ac_coverage=["AC-1"],
            )
        ],
        ac_mapping={"AC-1": ["tests/test_modulo.py::test_modulo"]},
        summary="Test-first modulo tests",
    )
    mock_prov.add_response(content=qa_suite.model_dump_json(), model="claude-3-5-haiku")
    # Red-check simulation: step 1 collects cleanly (exit code 0), step 2 fails on NotImplementedError (exit code 1)
    sandbox.add_result(stdout="collected 1 item", exit_code=0)
    sandbox.add_result(stdout="tests/test_modulo.py F\nE   NotImplementedError: Unimplemented stub", exit_code=1)
    suite = await controller.run_qa_test_first_stage(mock_prov, "claude-3-5-haiku")
    assert (repo_dir / "tests/test_modulo.py").exists()

    # 6. Task brief synthesis (< 1500 tokens)
    brief = controller.generate_task_brief(
        task_id="task-modulo-1",
        task_title="Implement modulo",
        task_description="Implement modulo function in calc.py to satisfy tests",
        write_scope=["calc.py"],
    )
    assert "AC-1" in brief
    assert "Interface Contracts & Signatures" in brief
    assert len(brief.split()) < 1500

    # 7. Local coder implements function
    calc_path = repo_dir / "calc.py"
    calc_path.write_text("def modulo(a: int, b: int) -> int:\n    return a % b\n", encoding="utf-8")

    # 8. Deterministic quality gates pass
    sandbox.add_result(stdout="ok", exit_code=0)      # compile
    sandbox.add_result(stdout="clean", exit_code=0)   # lint
    sandbox.add_result(stdout="PASSED", exit_code=0)  # test

    # 9. Traceability Matrix update
    matrix = TraceabilityMatrix.create_from_spec(
        spec=spec,
        planned_tasks=[{"id": "task-modulo-1", "title": "Implement modulo"}],
        qa_mapping=suite.ac_mapping,
    )
    matrix.update_test_results(test_output="PASSED tests/test_modulo.py::test_modulo", exit_code=0)
    ready, unres = matrix.is_review_ready()
    assert ready is True

    # 10. Generate PR with full traceability and telemetry
    gate_res = GateSuiteResult(passed=True, summary="All quality gates passed")
    review_rep = ReviewReport(
        verdict="APPROVE",
        confidence_score=0.98,
        summary="Clean modulo implementation adhering to architecture contract.",
        security_assessment="Safe arithmetic, zero vulnerabilities.",
        strengths=["Clear types", "Full test coverage"],
        concerns=[],
    )
    pr_file = PRGenerator.generate(
        run_id="run_e2e_hybrid",
        goal="Add modulo operator",
        repo_path=repo_dir,
        base_commit="HEAD~1",
        integration_branch="main",
        tasks=[{"id": "task-modulo-1", "role": "backend", "title": "Implement modulo", "state": "done"}],
        gate_result=gate_res,
        review_report=review_rep,
        budget_summary={"elapsed_minutes": 1.5, "total_llm_calls": 3, "total_input_tokens": 1200, "total_output_tokens": 400},
        output_dir=run_dir,
        traceability_markdown=matrix.render_matrix_markdown(),
        role_telemetry=[
            {"role": "pm", "provider": "anthropic", "model": "claude-3-5-sonnet", "cost_usd": 0.009},
            {"role": "architect", "provider": "anthropic", "model": "claude-3-5-sonnet", "cost_usd": 0.012},
            {"role": "qa", "provider": "anthropic", "model": "claude-3-5-haiku", "cost_usd": 0.003},
            {"role": "backend", "provider": "ollama", "model": "qwen3.5:9b", "cost_usd": 0.0},
        ],
        cloud_accounting={"spent_usd_total": 0.024, "content_classes": ["goal", "spec", "architecture"], "bytes_sent": 5120},
    )

    assert pr_file.exists()
    pr_text = pr_file.read_text(encoding="utf-8")
    assert "Acceptance Criteria Traceability Matrix" in pr_text
    assert "AC-1" in pr_text
    assert "$0.00 (Local)" in pr_text
    assert "Cloud Data Policy & Outbound Telemetry" in pr_text


