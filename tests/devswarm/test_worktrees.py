"""
tests/devswarm/test_worktrees.py — Unit tests for git worktree isolation and conflict detection.
"""

import subprocess
import pytest
from pathlib import Path

from devswarm.platform.worktrees import WorktreeManager, GitWorktreeError, GitConflictError


def init_test_git_repo(repo_dir: Path) -> str:
    """Initialize a git repo with an initial commit for testing."""
    def run(args):
        subprocess.run(["git"] + args, cwd=str(repo_dir), check=True, capture_output=True)

    run(["init", "-b", "main"])
    run(["config", "user.name", "Test User"])
    run(["config", "user.email", "test@example.com"])

    file_a = repo_dir / "app.py"
    file_a.write_text("def hello():\n    return 'hello'\n", encoding="utf-8")

    run(["add", "app.py"])
    run(["commit", "-m", "Initial commit"])

    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    return res.stdout.strip()


def test_worktree_lifecycle_and_merge(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm_meta"
    init_commit = init_test_git_repo(repo_dir)

    manager = WorktreeManager(repo_path=repo_dir, run_id="run-001", swarm_dir=swarm_dir)

    # 1. Ensure clean repo
    assert manager.ensure_clean_repo() == init_commit

    # 2. Setup integration branch
    int_branch = manager.setup_integration_branch()
    assert int_branch == "swarm/run-001"

    # 3. Create task worktree
    session = manager.create_task_worktree("task-1")
    assert session.worktree_path.exists()
    assert (session.worktree_path / "app.py").exists()

    # 4. Agent simulates editing app.py inside the worktree
    wt_app = session.worktree_path / "app.py"
    wt_app.write_text("def hello():\n    return 'hello world'\n", encoding="utf-8")

    # 5. Platform commits the agent's work
    commit_hash = manager.commit_changes("task-1", "Update hello return string")
    assert commit_hash is not None

    # 6. Merge into integration branch
    merge_hash = manager.merge_task_branch("task-1")
    assert merge_hash is not None

    # Verify integration branch has the updated text
    res = subprocess.run(
        ["git", "show", f"swarm/run-001:app.py"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "hello world" in res.stdout

    # 7. Cleanup worktrees
    manager.cleanup_all()


def test_worktree_ensure_clean_repo_allow_dirty_on_resume(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm_meta"
    init_test_git_repo(repo_dir)

    manager = WorktreeManager(repo_path=repo_dir, run_id="run-003", swarm_dir=swarm_dir)
    manager.setup_integration_branch()

    # Simulate an agent or crashed stage leaving an untracked file
    untracked = repo_dir / "tests" / "test_dummy.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("def test_dummy(): pass\n")

    # Regular ensure_clean_repo raises GitWorktreeError
    with pytest.raises(GitWorktreeError):
        manager.ensure_clean_repo(allow_dirty_on_resume=False)

    # When allow_dirty_on_resume is True, it commits the changes cleanly
    manager.ensure_clean_repo(allow_dirty_on_resume=True)

    # Verify working tree is clean afterwards
    res = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    assert res.stdout.strip() == ""

    assert not (swarm_dir / "worktrees" / "run-001").exists()


def test_worktree_merge_conflict_detection(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm_meta"
    init_test_git_repo(repo_dir)

    manager = WorktreeManager(repo_path=repo_dir, run_id="run-002", swarm_dir=swarm_dir)
    manager.setup_integration_branch()

    # Create Task 1: edits line 2 of app.py
    s1 = manager.create_task_worktree("task-1")
    (s1.worktree_path / "app.py").write_text("def hello():\n    return 'option A'\n")
    manager.commit_changes("task-1", "Option A")

    # Create Task 2 off the same integration branch: edits the exact same line differently
    s2 = manager.create_task_worktree("task-2")
    (s2.worktree_path / "app.py").write_text("def hello():\n    return 'option B'\n")
    manager.commit_changes("task-2", "Option B")

    # Merge Task 1: succeeds
    manager.merge_task_branch("task-1")

    # Merge Task 2: conflicting edit on same line -> MUST raise GitConflictError
    with pytest.raises(GitConflictError) as exc_info:
        manager.merge_task_branch("task-2")

    assert "Merge conflict detected" in str(exc_info.value)

    # Verify merge was aborted cleanly and repo is not left in a merge state
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == ""

    manager.cleanup_all()


def test_worktree_ensure_clean_repo_allow_dirty_on_resume(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    swarm_dir = tmp_path / "swarm_meta"
    init_test_git_repo(repo_dir)

    manager = WorktreeManager(repo_path=repo_dir, run_id="run-003", swarm_dir=swarm_dir)
    manager.setup_integration_branch()

    # Simulate an agent or crashed stage leaving an untracked file
    untracked = repo_dir / "tests" / "test_dummy.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("def test_dummy(): pass\n")

    # Regular ensure_clean_repo raises GitWorktreeError
    with pytest.raises(GitWorktreeError):
        manager.ensure_clean_repo(allow_dirty_on_resume=False)

    # When allow_dirty_on_resume is True, it commits the changes cleanly
    manager.ensure_clean_repo(allow_dirty_on_resume=True)

    # Verify working tree is clean afterwards
    res = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
    assert res.stdout.strip() == ""


