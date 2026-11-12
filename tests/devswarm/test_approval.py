"""
tests/devswarm/test_approval.py — Unit tests for human approval and merge management.
"""

from pathlib import Path
import json
import subprocess
import pytest

from devswarm.approval import ApprovalManager, ApprovalError


def init_test_git_repo(repo_dir: Path) -> None:
    def run(args):
        subprocess.run(["git"] + args, cwd=str(repo_dir), check=True, capture_output=True)

    run(["init", "-b", "main"])
    run(["config", "user.name", "Approval Tester"])
    run(["config", "user.email", "approval@example.com"])
    (repo_dir / "calc.py").write_text("def add(a, b): return a + b\n")
    run(["add", "calc.py"])
    run(["commit", "-m", "Initial commit on main"])


def create_integration_branch(repo_dir: Path, branch_name: str) -> None:
    def run(args):
        subprocess.run(["git"] + args, cwd=str(repo_dir), check=True, capture_output=True)

    run(["checkout", "-b", branch_name])
    (repo_dir / "calc.py").write_text("def add(a, b): return a + b\ndef mul(a, b): return a * b\n")
    run(["add", "calc.py"])
    run(["commit", "-m", "Implement mul on integration branch"])
    run(["checkout", "main"])


def test_approval_manager_nonexistent_run(tmp_path: Path) -> None:
    mgr = ApprovalManager(swarm_dir=tmp_path / ".swarm")
    with pytest.raises(ApprovalError):
        mgr.approve_run(run_id="run_non_existent")


def test_approve_run_merges_into_main(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    init_test_git_repo(repo_dir)

    run_id = "run_20261110T080000Z"
    integration_branch = f"swarm/{run_id}"
    create_integration_branch(repo_dir, integration_branch)

    swarm_dir = tmp_path / ".swarm"
    run_dir = swarm_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    manifest_data = {
        "run_id": run_id,
        "goal": "Add multiplication logic",
        "target_repo": str(repo_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    mgr = ApprovalManager(swarm_dir=swarm_dir)
    success = mgr.approve_run(run_id=run_id, default_branch="main")
    assert success is True

    # Verify main branch now contains the changes from integration branch
    calc_content = (repo_dir / "calc.py").read_text(encoding="utf-8")
    assert "def mul(a, b):" in calc_content

    # Verify APPROVED marker
    assert (run_dir / "APPROVED").exists()

    # Verify audit event
    events_file = run_dir / "events.jsonl"
    assert events_file.exists()
    events_text = events_file.read_text(encoding="utf-8")
    assert "run_approved_and_merged" in events_text


def test_reject_run_preserves_branch(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    init_test_git_repo(repo_dir)

    run_id = "run_rejected_01"
    integration_branch = f"swarm/{run_id}"
    create_integration_branch(repo_dir, integration_branch)

    swarm_dir = tmp_path / ".swarm"
    run_dir = swarm_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    manifest_data = {
        "run_id": run_id,
        "goal": "Feature with flaws",
        "target_repo": str(repo_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    mgr = ApprovalManager(swarm_dir=swarm_dir)
    success = mgr.reject_run(run_id=run_id, reason="Code does not meet performance requirements.")
    assert success is True

    # Verify REJECTED marker
    rej_file = run_dir / "REJECTED"
    assert rej_file.exists()
    rej_data = json.loads(rej_file.read_text(encoding="utf-8"))
    assert "performance requirements" in rej_data["reason"]

    # Verify branch was NOT merged into main
    calc_content = (repo_dir / "calc.py").read_text(encoding="utf-8")
    assert "def mul(a, b):" not in calc_content

