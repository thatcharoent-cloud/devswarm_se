"""
devswarm.tools.command_tool — Sandboxed command execution tool for agents.

Ensures that all shell commands issued by agents pass through the configured Sandbox
(Docker or MockSandbox) with network isolation and role allowlist enforcement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from devswarm.platform.sandbox import Sandbox, CommandResult

logger = logging.getLogger(__name__)


class CommandTool:
    """Provides the `run_command` tool to agents."""

    def __init__(
        self,
        sandbox: Sandbox,
        worktree_dir: Path | str,
        role_allowlist: list[str] | set[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.sandbox = sandbox
        self.worktree_dir = Path(worktree_dir).resolve()
        self.role_allowlist = role_allowlist
        self.timeout = timeout

    def execute(self, command: str) -> str:
        """Execute a shell command inside the sandbox and return delimited output."""
        logger.info("Executing sandboxed command: %s", command)
        res = self.sandbox.run_command(
            command=command,
            worktree_dir=self.worktree_dir,
            timeout=self.timeout,
            role_allowlist=self.role_allowlist,
        )
        return res.formatted_output()

