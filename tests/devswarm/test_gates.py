"""
tests/devswarm/test_gates.py — Unit and integration tests for Deterministic Quality Gates.
"""

from pathlib import Path
import pytest

from devswarm.gates import (
    GateRunner,
    GateConfig,
    GateResult,
    GateSuiteResult,
    BuiltinSecurityScanner,
)
from devswarm.platform.sandbox import MockSandbox, CommandResult


def test_builtin_security_scanner_secrets(tmp_path: Path) -> None:
    scanner = BuiltinSecurityScanner()
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("def add(a, b): return a + b\n", encoding="utf-8")
    assert len(scanner.scan_file_for_secrets(clean_file)) == 0

    secret_file = tmp_path / "leak.py"
    secret_file.write_text("API_KEY = 'sk-ant-api03-abcdef1234567890abcdef'\n", encoding="utf-8")
    findings = scanner.scan_file_for_secrets(secret_file)
    assert len(findings) > 0
    assert "Hardcoded secret detected" in findings[0]


def test_builtin_security_scanner_ast(tmp_path: Path) -> None:
    scanner = BuiltinSecurityScanner()
    clean_file = tmp_path / "calc.py"
    clean_file.write_text("def mul(a, b): return a * b\n", encoding="utf-8")
    assert len(scanner.scan_file_for_dangerous_ast(clean_file)) == 0

    eval_file = tmp_path / "eval_danger.py"
    eval_file.write_text("def run(code): return eval(code)\n", encoding="utf-8")
    findings = scanner.scan_file_for_dangerous_ast(eval_file)
    assert len(findings) == 1
    assert "Disallowed function 'eval()'" in findings[0]

    exec_file = tmp_path / "exec_danger.py"
    exec_file.write_text("exec('print(1)')\n", encoding="utf-8")
    exec_findings = scanner.scan_file_for_dangerous_ast(exec_file)
    assert len(exec_findings) == 1
    assert "Disallowed function 'exec()'" in exec_findings[0]


def test_builtin_security_scanner_high_risk_paths() -> None:
    scanner = BuiltinSecurityScanner()
    modified = ["calc.py", "src/auth/service.py", "test_calc.py", "requirements.txt"]
    high_risk = scanner.check_high_risk_paths(modified)
    assert "requirements.txt" in high_risk
    assert "src/auth/service.py" in high_risk
    assert "calc.py" not in high_risk
    assert "test_calc.py" not in high_risk


@pytest.mark.asyncio
async def test_gate_runner_all_passed(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

    sandbox = MockSandbox()
    runner = GateRunner(sandbox=sandbox, gates_config_path="config/gates.yaml")

    res = await runner.run_suite(worktree, modified_files=["calc.py"])
    assert res.passed is True
    assert "compile" in res.results
    assert res.results["compile"].passed is True
    assert "security" in res.results
    assert res.results["security"].passed is True
    assert res.has_high_risk_changes is False


@pytest.mark.asyncio
async def test_gate_runner_command_failure_generates_feedback(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

    sandbox = MockSandbox()
    # Script compile pass, lint pass, test failure
    sandbox.add_result(stdout="ok", exit_code=0)      # compile
    sandbox.add_result(stdout="clean", exit_code=0)   # lint
    sandbox.add_result(stderr="FAILED: test_add - AssertionError: 5 != 6", exit_code=1)  # test

    runner = GateRunner(sandbox=sandbox, gates_config_path="config/gates.yaml")
    res = await runner.run_suite(worktree, modified_files=["calc.py"])

    assert res.passed is False
    assert "test" in res.results
    assert res.results["test"].passed is False
    assert "DETERMINISTIC QUALITY GATES FAILED" in res.feedback_prompt
    assert "FAILED: test_add" in res.feedback_prompt


@pytest.mark.asyncio
async def test_gate_runner_security_violation_fails_suite(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    # Code with eval
    (worktree / "danger.py").write_text("x = eval('1 + 1')\n", encoding="utf-8")

    sandbox = MockSandbox()
    runner = GateRunner(sandbox=sandbox, gates_config_path="config/gates.yaml")

    res = await runner.run_suite(worktree, modified_files=["danger.py"])
    assert res.passed is False
    assert res.results["security"].passed is False
    assert "eval()" in res.results["security"].stderr
    assert "QUALITY GATES FAILED" in res.feedback_prompt


@pytest.mark.asyncio
async def test_gate_runner_detects_high_risk_changes(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "auth_handler.py").write_text("def login(): pass\n", encoding="utf-8")

    sandbox = MockSandbox()
    runner = GateRunner(sandbox=sandbox, gates_config_path="config/gates.yaml")

    res = await runner.run_suite(worktree, modified_files=["auth_handler.py"])
    assert res.passed is True
    assert res.has_high_risk_changes is True
    assert "auth_handler.py" in res.high_risk_files


import json
import subprocess
from devswarm.dev_orchestrator import DevOrchestrator
from devswarm.llm.mock_llm import MockLLM
from devswarm.platform.taskqueue import TaskState


def init_test_git_repo(repo_dir: Path) -> str:
    def run(args):
        subprocess.run(["git"] + args, cwd=str(repo_dir), check=True, capture_output=True)

    run(["init", "-b", "main"])
    run(["config", "user.name", "Gate Tester"])
    run(["config", "user.email", "tester@example.com"])
    (repo_dir / "calc.py").write_text("def add(a, b): return a + b\n")
    run(["add", "calc.py"])
    run(["commit", "-m", "Initial commit"])
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    return res.stdout.strip()


@pytest.mark.asyncio
async def test_orchestrator_gates_failure_blocks_merge(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    init_test_git_repo(repo_dir)

    mock_llm = MockLLM()
    # Script a single task plan
    plan_dict = {
        "goal": "Failing gate test",
        "assumptions": [],
        "tasks": [
            {
                "id": "task-fail-gates",
                "title": "Task that breaks gates",
                "role": "backend",
                "description": "Produce invalid code",
                "acceptance_criteria": ["None"],
                "write_scope": ["calc.py"],
                "minutes_budget": 5.0,
            }
        ],
    }
    mock_llm.add_response(content=json.dumps(plan_dict))

    # Mock sandbox where test gate fails every time
    mock_sb = MockSandbox()
    # For compile, lint, test gates
    mock_sb.add_result(stdout="ok", exit_code=0)      # compile
    mock_sb.add_result(stdout="clean", exit_code=0)   # lint
    mock_sb.add_result(stderr="FAILED: test suite crashed", exit_code=1)  # test (initial)
    # For repair attempt
    mock_sb.add_result(stdout="ok", exit_code=0)      # compile
    mock_sb.add_result(stdout="clean", exit_code=0)   # lint
    mock_sb.add_result(stderr="FAILED: test suite still crashed", exit_code=1)  # test (repair)

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Failing gate test",
        client=mock_llm,
        sandbox=mock_sb,
        swarm_dir=swarm_dir,
    )

    exit_code = await orch.execute_run()
    # Run should exit with code 1 because a task failed quality gates
    assert exit_code == 1

    status = orch.get_status()
    assert status["total_tasks"] == 1
    assert status["failed_tasks"] == 1
    assert status["done_tasks"] == 0
    task_info = status["tasks"][0]
    assert task_info["state"] == "failed"
    assert "Quality gates failed" in task_info["error"]


@pytest.mark.asyncio
async def test_orchestrator_gates_high_risk_routes_to_needs_human(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    init_test_git_repo(repo_dir)

    mock_llm = MockLLM()
    # Plan touching a high risk file: auth_service.py
    plan_dict = {
        "goal": "High risk test",
        "assumptions": [],
        "tasks": [
            {
                "id": "task-high-risk",
                "title": "Modify auth service",
                "role": "backend",
                "description": "Update auth logic",
                "acceptance_criteria": ["None"],
                "write_scope": ["auth_service.py"],
                "minutes_budget": 5.0,
            }
        ],
    }
    # Scripted sequence: 1. plan response, 2. write_file tool call, 3. finish_task tool call
    mock_llm.set_scripted_responses([
        json.dumps(plan_dict),
        {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": {
                        "path": "auth_service.py",
                        "content": "def authenticate(): return True\n",
                    },
                }
            }],
        },
        {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "finish_task",
                    "arguments": {
                        "summary": "Implemented auth logic",
                        "files_modified": ["auth_service.py"],
                    },
                }
            }],
        },
    ])

    mock_sb = MockSandbox()
    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="High risk test",
        client=mock_llm,
        sandbox=mock_sb,
        swarm_dir=swarm_dir,
    )

    exit_code = await orch.execute_run()
    # Run pauses with code 1 when tasks need human approval
    assert exit_code == 1

    status = orch.get_status()
    task_info = status["tasks"][0]
    assert task_info["state"] == "needs_human"
    assert "auth_service.py" in task_info["result_summary"]
