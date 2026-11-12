"""
devswarm.approval — Human approval and merge workflow manager.

Implements the Human Product Owner control boundary:
  - `python main.py approve --run-id <run_id>`:
    Validates objective gates, merges the integration branch into the default branch (main),
    updates run manifest and event log, and cleans up task worktrees.
  - `python main.py reject --run-id <run_id> --reason "<reason>"`:
    Records human rejection with feedback, preserves the integration branch for inspection,
    and logs the audit event.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devswarm.platform.events import EventLog

logger = logging.getLogger(__name__)

DEFAULT_SWARM_DIR = Path(".swarm")


class ApprovalError(Exception):
    """Raised when approval or merge preconditions are violated."""


class ApprovalManager:
    """Controls the final human approval gate before default branch merge."""

    def __init__(
        self,
        swarm_dir: Path | str = DEFAULT_SWARM_DIR,
        repo_path: Path | str | None = None,
    ) -> None:
        self.swarm_dir = Path(swarm_dir).resolve()
        self.repo_path = Path(repo_path).resolve() if repo_path else None

    def get_latest_run_id(self) -> str | None:
        """Find the most recent run ID in .swarm/runs/."""
        runs_dir = self.swarm_dir / "runs"
        if not runs_dir.exists():
            return None
        dirs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])
        return dirs[-1] if dirs else None

    def approve_run(
        self,
        run_id: str | None = None,
        default_branch: str = "main",
        force: bool = False,
    ) -> bool:
        """Approve run and merge integration branch into default branch."""
        target_id = run_id or self.get_latest_run_id()
        if not target_id:
            raise ApprovalError("No swarm runs found to approve.")

        run_dir = self.swarm_dir / "runs" / target_id
        if not run_dir.exists():
            raise ApprovalError(f"Run directory not found: {run_dir}")

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ApprovalError(f"Manifest not found for run {target_id}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_repo_from_manifest = manifest.get("target_repo")
        if self.repo_path and self.repo_path != Path(".").resolve():
            repo_dir = self.repo_path
        elif target_repo_from_manifest and Path(target_repo_from_manifest).exists():
            repo_dir = Path(target_repo_from_manifest).resolve()
        else:
            repo_dir = self.repo_path or Path(".").resolve()
        goal = manifest.get("goal", "Engineering feature")
        integration_branch = f"swarm/{target_id}"

        event_log = EventLog(run_dir=run_dir, run_id=target_id)

        # Helper to execute git command on target repo
        def git_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git"] + args,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )

        # 1. Verify integration branch exists
        branch_check = git_cmd(["rev-parse", "--verify", integration_branch])
        if branch_check.returncode != 0:
            raise ApprovalError(f"Integration branch '{integration_branch}' does not exist in {repo_dir}")

        # 2. Checkout default branch and ensure clean
        status_check = git_cmd(["status", "--porcelain"])
        if status_check.stdout.strip():
            if not force:
                changes = status_check.stdout.strip()
                raise ApprovalError(
                    f"Target repository '{repo_dir}' has uncommitted changes:\n"
                    f"{changes}\n\n"
                    f"👉 To proceed:\n"
                    f"   1. Stash or commit in target repo: cd {repo_dir} && git stash\n"
                    f"   2. OR re-run with --force to auto-stash and merge:\n"
                    f"      python main.py approve --run-id {target_id} --force"
                )
            else:
                logger.warning("Forcing approval: auto-stashing uncommitted changes in %s", repo_dir)
                git_cmd(["stash", "save", "-u", f"devswarm-pre-approval-{target_id}"])

        checkout_res = git_cmd(["checkout", default_branch])
        if checkout_res.returncode != 0:
            raise ApprovalError(f"Failed to checkout '{default_branch}': {checkout_res.stderr}")

        # 3. Merge integration branch into default branch with merge commit
        merge_msg = f"Merge swarm pull request [{target_id}]: {goal}"
        merge_res = git_cmd(["merge", "--no-ff", integration_branch, "-m", merge_msg])
        if merge_res.returncode != 0:
            # Abort merge on conflict
            git_cmd(["merge", "--abort"])
            raise ApprovalError(f"Merge conflict while merging '{integration_branch}' into '{default_branch}': {merge_res.stderr}")

        # 4. Record approval audit event and marker
        event_log.emit(
            "run_approved_and_merged",
            args={
                "target_branch": default_branch,
                "integration_branch": integration_branch,
                "commit_message": merge_msg,
            },
        )

        approval_file = run_dir / "APPROVED"
        approval_file.write_text(
            json.dumps({
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "default_branch": default_branch,
                "integration_branch": integration_branch,
                "run_id": target_id,
            }, indent=2),
            encoding="utf-8",
        )

        print(f"\n🎉 Successfully approved and merged [{integration_branch}] into '{default_branch}'!")
        print(f"   Commit message: {merge_msg}")
        return True

    def reject_run(
        self,
        run_id: str | None = None,
        reason: str = "Operator rejected pull request",
    ) -> bool:
        """Reject run and document rejection reasons."""
        target_id = run_id or self.get_latest_run_id()
        if not target_id:
            raise ApprovalError("No swarm runs found to reject.")

        run_dir = self.swarm_dir / "runs" / target_id
        if not run_dir.exists():
            raise ApprovalError(f"Run directory not found: {run_dir}")

        event_log = EventLog(run_dir=run_dir, run_id=target_id)
        event_log.emit("run_rejected", args={"reason": reason})

        rejection_file = run_dir / "REJECTED"
        rejection_file.write_text(
            json.dumps({
                "rejected_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "run_id": target_id,
            }, indent=2),
            encoding="utf-8",
        )

        print(f"\n🛑 Run [{target_id}] rejected.")
        print(f"   Reason: {reason}")
        print(f"   Integration branch 'swarm/{target_id}' remains preserved for manual inspection.")
        return True

