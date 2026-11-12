"""
tests/devswarm/test_dev_orchestrator.py — Unit tests for Chief Engineering Orchestrator.
"""

import json
import subprocess
import pytest
from pathlib import Path

from devswarm.dev_orchestrator import DevOrchestrator, InvalidPlanError
from devswarm.llm.mock_llm import MockLLM
from devswarm.platform.guards import KillSwitchTriggered
from devswarm.platform.sandbox import MockSandbox
from devswarm.platform.taskqueue import TaskState


def init_git_repo(repo_dir: Path) -> str:
    """Helper to initialize a clean git repo."""
    def run(args):
        subprocess.run(["git"] + args, cwd=str(repo_dir), check=True, capture_output=True)

    run(["init", "-b", "main"])
    run(["config", "user.name", "Test User"])
    run(["config", "user.email", "test@example.com"])
    (repo_dir / "calc.py").write_text("def add(a, b): return a + b\n")
    run(["add", "calc.py"])
    run(["commit", "-m", "Initial commit"])
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    return res.stdout.strip()


@pytest.mark.asyncio
async def test_orchestrator_planning_and_execution(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    init_git_repo(repo_dir)

    mock_llm = MockLLM()
    # Script valid plan
    valid_plan = {
        "goal": "Add multiply and tests",
        "assumptions": ["Use standard arithmetic"],
        "tasks": [
            {
                "id": "task-1",
                "title": "Implement multiply",
                "role": "backend",
                "description": "Add multiply function to calc.py",
                "acceptance_criteria": ["multiply(2, 3) == 6"],
                "depends_on": [],
                "write_scope": ["calc.py"],
                "minutes_budget": 5.0,
            },
            {
                "id": "task-2",
                "title": "Add test for multiply",
                "role": "tester",
                "description": "Add test in test_calc.py",
                "acceptance_criteria": ["Test passes"],
                "depends_on": ["task-1"],
                "write_scope": ["test_calc.py"],
                "minutes_budget": 4.0,
            },
        ],
    }
    mock_llm.add_response(content=json.dumps(valid_plan))

    mock_sb = MockSandbox()

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Add multiply and tests",
        client=mock_llm,
        sandbox=mock_sb,
        swarm_dir=swarm_dir,
    )

    exit_code = await orch.execute_run()
    assert exit_code == 0

    # Verify tasks are marked DONE
    status = orch.get_status()
    assert status["total_tasks"] == 2
    assert status["done_tasks"] == 2

    # Verify integration branch was created and merged
    res = subprocess.run(
        ["git", "branch", "--list", orch.worktrees.integration_branch],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    assert orch.worktrees.integration_branch in res.stdout


@pytest.mark.asyncio
async def test_orchestrator_repair_retry_on_invalid_plan(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    init_git_repo(repo_dir)

    mock_llm = MockLLM()

    # Attempt 1: Plan with disabled role 'frontend'
    bad_plan = {
        "goal": "Add feature",
        "assumptions": [],
        "tasks": [
            {
                "id": "task-1",
                "title": "Build UI button",
                "role": "frontend",  # Disabled role
                "description": "bad role",
                "acceptance_criteria": ["ok"],
                "write_scope": ["button.tsx"],
            }
        ],
    }
    # Attempt 2: Repaired plan with enabled role 'backend'
    repaired_plan = {
        "goal": "Add feature",
        "assumptions": ["Use backend API"],
        "tasks": [
            {
                "id": "task-1",
                "title": "Build backend endpoint",
                "role": "backend",  # Enabled
                "description": "repaired",
                "acceptance_criteria": ["ok"],
                "write_scope": ["api.py"],
            }
        ],
    }

    mock_llm.add_response(content=json.dumps(bad_plan))
    mock_llm.add_response(content=json.dumps(repaired_plan))

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Add feature",
        client=mock_llm,
        sandbox=MockSandbox(),
        swarm_dir=swarm_dir,
    )

    plan = await orch.generate_plan(assumptions=[])
    assert len(plan.tasks) == 1
    assert plan.tasks[0].role == "backend"


@pytest.mark.asyncio
async def test_orchestrator_stop_kill_switch(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    init_git_repo(repo_dir)

    mock_llm = MockLLM()
    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Halt immediately",
        client=mock_llm,
        sandbox=MockSandbox(),
        swarm_dir=swarm_dir,
    )

    # Trigger kill switch
    orch.kill_switch.trigger("Operator abort test")

    with pytest.raises(KillSwitchTriggered):
        await orch.execute_run()


@pytest.mark.asyncio
async def test_orchestrator_status_output(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm"
    init_git_repo(repo_dir)

    orch = DevOrchestrator(
        repo_path=repo_dir,
        goal="Status test",
        client=MockLLM(),
        sandbox=MockSandbox(),
        swarm_dir=swarm_dir,
    )

    status = orch.get_status()
    assert status["goal"] == "Status test"
    assert status["total_tasks"] == 0
    # Printing status table should run without exception
    orch.print_status_table()

