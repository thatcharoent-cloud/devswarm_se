"""
devswarm.platform.worktrees — Git worktree and branch isolation manager.

Security principles:
  1. Agents NEVER run git commands directly. All branching, committing, and merging
     are executed exclusively by this platform module.
  2. The swarm NEVER merges into or pushes to the repository's default branch.
  3. Every task gets an isolated branch and worktree: .swarm/worktrees/<run_id>/<task_id>.
  4. Changes are merged into the integration branch `swarm/<run_id>` in dependency order.
  5. Merge conflicts are strictly detected and flagged as `needs_human` (never auto-resolved).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SWARM_COMMITTER_NAME = "Swarm Platform"
SWARM_COMMITTER_EMAIL = "swarm@local"


class GitWorktreeError(Exception):
    """Base exception for Git worktree operations."""


class GitConflictError(GitWorktreeError):
    """Raised when a task branch cannot be merged cleanly without conflict."""


@dataclass
class WorktreeSession:
    """Active worktree location and branch metadata for a task."""
    run_id: str
    task_id: str
    worktree_path: Path
    branch_name: str
    integration_branch: str
    base_commit: str


class WorktreeManager:
    """Manages git branches and worktrees for tasks in a run."""

    def __init__(self, repo_path: Path | str, run_id: str, swarm_dir: Path | str = ".swarm") -> None:
        self.repo_path = Path(repo_path).resolve()
        self.run_id = run_id
        self.swarm_dir = Path(swarm_dir).resolve()
        self.worktrees_root = self.swarm_dir / "worktrees" / run_id
        self.integration_branch = f"swarm/{run_id}"
        self._base_commit: str = ""

    @property
    def base_commit(self) -> str:
        if not self._base_commit:
            _, head_commit, _ = self._run_git(["rev-parse", "HEAD"], check=False)
            self._base_commit = head_commit
        return self._base_commit

    @base_commit.setter
    def base_commit(self, val: str) -> None:
        self._base_commit = val

    def _run_git(self, args: list[str], cwd: Path | None = None, check: bool = True) -> tuple[int, str, str]:
        """Execute a git command safely in repo or worktree."""
        exec_cwd = str(cwd or self.repo_path)
        env = dict(os.environ)
        env["GIT_AUTHOR_NAME"] = SWARM_COMMITTER_NAME
        env["GIT_AUTHOR_EMAIL"] = SWARM_COMMITTER_EMAIL
        env["GIT_COMMITTER_NAME"] = SWARM_COMMITTER_NAME
        env["GIT_COMMITTER_EMAIL"] = SWARM_COMMITTER_EMAIL

        res = subprocess.run(
            ["git"] + args,
            cwd=exec_cwd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if check and res.returncode != 0:
            raise GitWorktreeError(
                f"git {' '.join(args)} failed in {exec_cwd} (exit {res.returncode}): {res.stderr.strip()}"
            )
        return res.returncode, res.stdout.strip(), res.stderr.strip()

    def ensure_clean_repo(self, allow_dirty_on_resume: bool = False) -> str:
        """Verify the repo is git-initialized and has no uncommitted changes."""
        # 1. Check directory existence and git repo status
        if not self.repo_path.exists():
            self.repo_path.mkdir(parents=True, exist_ok=True)

        rc, out, _ = self._run_git(["rev-parse", "--is-inside-work-tree"], check=False)
        if rc != 0 or out != "true":
            logger.info("Initializing new git repository at %s", self.repo_path)
            self._run_git(["init", "-b", "main"])
            self._run_git(["commit", "--allow-empty", "-m", "Initial commit"])

        # 2. Check clean working directory
        # Discard untracked or modified transient OS metadata files (e.g. .DS_Store)
        self._run_git(["checkout", "--", ".DS_Store"], check=False)

        _, status_out, _ = self._run_git(["status", "--porcelain"])
        relevant_lines = [
            line for line in status_out.strip().splitlines()
            if line.strip() and not any(k in line for k in (".DS_Store", "__pycache__", ".pytest_cache"))
        ]
        if relevant_lines:
            status_text = "\n".join(relevant_lines)
            if allow_dirty_on_resume:
                logger.info(
                    "Resuming run with uncommitted working tree changes (%s); preserving them on integration branch.",
                    status_text.replace("\n", ", "),
                )
                rc_b, _, _ = self._run_git(["rev-parse", "--verify", self.integration_branch], check=False)
                if rc_b == 0:
                    self._run_git(["checkout", self.integration_branch], check=False)
                else:
                    _, head_c, _ = self._run_git(["rev-parse", "HEAD"], check=False)
                    self._run_git(["checkout", "-b", self.integration_branch, head_c or "HEAD"], check=False)
                self.commit_integration_changes("Auto-commit uncommitted changes on resume")
            else:
                raise GitWorktreeError(
                    f"Repository has uncommitted changes. Please commit or stash before starting swarm:\n{status_text}"
                )

        # 3. Get current commit (or create initial empty commit if empty repository)
        rc_head, head_commit, _ = self._run_git(["rev-parse", "HEAD"], check=False)
        if rc_head != 0:
            logger.info("Git repository at %s has no commits; creating initial commit", self.repo_path)
            self._run_git(["commit", "--allow-empty", "-m", "Initial commit"])
            _, head_commit, _ = self._run_git(["rev-parse", "HEAD"])
        return head_commit

    def setup_integration_branch(self, base_commit: str | None = None, allow_dirty_on_resume: bool = False) -> str:
        """Create integration branch swarm/<run_id> from current HEAD and switch to it."""
        # Check if integration branch already exists
        rc, _, _ = self._run_git(["rev-parse", "--verify", self.integration_branch], check=False)
        if rc == 0:
            rc_curr, curr_b, _ = self._run_git(["branch", "--show-current"], check=False)
            if rc_curr == 0 and curr_b.strip() == self.integration_branch:
                return self.integration_branch
            self._run_git(["checkout", self.integration_branch], check=False)
            return self.integration_branch

        base = base_commit or self.ensure_clean_repo(allow_dirty_on_resume=allow_dirty_on_resume)
        self._run_git(["checkout", "-b", self.integration_branch, base])
        logger.info("Created and switched to integration branch '%s' at %s", self.integration_branch, base[:8])
        return self.integration_branch

    def commit_integration_changes(self, message: str) -> str | None:
        """Stage and commit changes directly on the integration branch."""
        self._run_git(["add", "-A"])
        _, status_out, _ = self._run_git(["status", "--porcelain"])
        if not status_out.strip():
            return None
        commit_msg = f"[devswarm:pre-impl] {message}"
        self._run_git(["commit", "-m", commit_msg])
        _, commit_hash, _ = self._run_git(["rev-parse", "HEAD"])
        logger.info("Committed pre-implementation changes to integration branch: %s", commit_hash[:8])
        return commit_hash

    def create_task_worktree(self, task_id: str) -> WorktreeSession:
        """Create an isolated branch and worktree for task_id based off integration_branch."""
        self.setup_integration_branch()

        task_branch = f"swarm-task/{self.run_id}/{task_id}"
        worktree_dir = self.worktrees_root / task_id

        # Clean existing worktree directory if it was left from a previous crashed run
        if worktree_dir.exists():
            self.remove_worktree(task_id)

        worktree_dir.parent.mkdir(parents=True, exist_ok=True)

        # Check if task branch exists; if not, branch from integration_branch
        rc, _, _ = self._run_git(["rev-parse", "--verify", task_branch], check=False)
        if rc != 0:
            self._run_git(["branch", task_branch, self.integration_branch])

        # Create git worktree
        self._run_git(["worktree", "add", str(worktree_dir), task_branch])
        logger.info("Created worktree for '%s' at %s", task_id, worktree_dir)

        _, base_commit, _ = self._run_git(["rev-parse", "HEAD"], cwd=worktree_dir)

        return WorktreeSession(
            run_id=self.run_id,
            task_id=task_id,
            worktree_path=worktree_dir,
            branch_name=task_branch,
            integration_branch=self.integration_branch,
            base_commit=base_commit,
        )

    def commit_changes(self, task_id: str, message: str) -> str | None:
        """Stage all modified files in the task worktree and create a commit."""
        worktree_dir = self.worktrees_root / task_id
        if not worktree_dir.exists():
            raise GitWorktreeError(f"Worktree for task {task_id} does not exist at {worktree_dir}")

        # Check status
        _, status_out, _ = self._run_git(["status", "--porcelain"], cwd=worktree_dir)
        if not status_out.strip():
            logger.info("No modifications to commit in worktree %s", task_id)
            return None

        # Stage and commit
        self._run_git(["add", "-A"], cwd=worktree_dir)
        commit_msg = f"[devswarm:{task_id}] {message}"
        self._run_git(["commit", "-m", commit_msg], cwd=worktree_dir)

        _, commit_hash, _ = self._run_git(["rev-parse", "HEAD"], cwd=worktree_dir)
        logger.info("Committed changes for %s: %s", task_id, commit_hash[:8])
        return commit_hash

    def merge_task_branch(self, task_id: str) -> str:
        """Merge task branch into integration branch. Raises GitConflictError if merge fails."""
        task_branch = f"swarm-task/{self.run_id}/{task_id}"

        # 1. Switch repo or a temporary worktree to integration_branch to test/execute merge
        # Run merge in the repository root or dedicated integration worktree
        rc, _, _ = self._run_git(["checkout", self.integration_branch], check=False)
        if rc != 0:
            raise GitWorktreeError(f"Could not switch to integration branch '{self.integration_branch}'")

        # 2. Attempt merge with conflict detection
        merge_msg = f"Merge task {task_id} into {self.integration_branch}"
        m_rc, m_out, m_err = self._run_git(
            ["merge", "--no-ff", "-m", merge_msg, task_branch],
            check=False,
        )

        if m_rc != 0:
            # Abort merge immediately on conflict
            self._run_git(["merge", "--abort"], check=False)
            logger.error("Merge conflict detected while merging '%s' into '%s'", task_branch, self.integration_branch)
            raise GitConflictError(
                f"Merge conflict detected merging {task_branch} into {self.integration_branch}: {m_out} {m_err}"
            )

        _, merge_commit, _ = self._run_git(["rev-parse", "HEAD"])
        logger.info("Merged %s into %s (commit %s)", task_branch, self.integration_branch, merge_commit[:8])
        return merge_commit

    def remove_worktree(self, task_id: str) -> None:
        """Remove a task worktree safely."""
        worktree_dir = self.worktrees_root / task_id
        if worktree_dir.exists():
            self._run_git(["worktree", "remove", "--force", str(worktree_dir)], check=False)
            if worktree_dir.exists():
                shutil.rmtree(worktree_dir, ignore_errors=True)
            self._run_git(["worktree", "prune"], check=False)

    def cleanup_all(self) -> None:
        """Remove all task worktrees for this run."""
        if self.worktrees_root.exists():
            for child in self.worktrees_root.iterdir():
                if child.is_dir():
                    self.remove_worktree(child.name)
            shutil.rmtree(self.worktrees_root, ignore_errors=True)
        self._run_git(["worktree", "prune"], check=False)
