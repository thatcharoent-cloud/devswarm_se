"""
devswarm.platform.sandbox — Sandboxed command execution engine.

Security principles:
  1. Commands run ONLY inside an isolated sandbox (Docker by default).
  2. Sandboxes have `--network none` (zero internet / zero LAN / cannot reach Ollama).
  3. Non-root user (`--user 1000:1000`).
  4. Only the task's worktree is mounted (`-v <worktree>:/workspace -w /workspace`).
  5. Strict resource limits: 2 CPUs, 2 GB RAM, 256 PIDs, execution timeout.
  6. No host environment variables or credentials leaked into the container.
  7. If Docker is unavailable, commands are REFUSED unless `--unsafe-local-exec` is passed.
  8. Output is treated as UNTRUSTED DATA: truncated, sanitized, and wrapped in delimiters.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from devswarm.platform.events import redact_secrets

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = "devswarm-sandbox:latest"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_CPUS = "2"
DEFAULT_MEMORY = "1g"
DEFAULT_PIDS = 256
MAX_OUTPUT_CHARS = 6000

# Safe common shell utilities accessible alongside role allowlists
SAFE_UTILITIES = {
    "cat", "ls", "grep", "find", "head", "tail", "wc", "echo", "pwd",
}

# Base allowed command binaries
GLOBAL_ALLOWLIST = {
    "pytest", "python", "python3", "mypy", "ruff", "black", "flake8",
    "npm", "node", "npx", "yarn", "eslint", "jest",
    "cargo", "rustc",
    "go",
    "cat", "ls", "grep", "find", "head", "tail", "wc", "echo", "pwd",
}

# Strictly forbidden command binaries (even if someone attempts them)
STRICT_DENYLIST = {
    "git", "curl", "wget", "nc", "netcat", "ssh", "scp", "rsync",
    "sudo", "su", "chmod", "chown", "apt", "apk", "yum", "brew",
    "reboot", "shutdown", "rmdir", "mkfs", "docker",
}


@dataclass
class CommandResult:
    """Output and status of a sandboxed command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    denied: bool = False
    denial_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.denied

    def formatted_output(self) -> str:
        """Return delimited and truncated output suitable for an LLM prompt."""
        if self.denied:
            return f"<COMMAND_OUTPUT status=\"DENIED\">\nCommand denied: {self.denial_reason}\n</COMMAND_OUTPUT>"
        if self.timed_out:
            return f"<COMMAND_OUTPUT status=\"TIMEOUT\">\nCommand timed out after {self.duration_s:.1f}s.\n</COMMAND_OUTPUT>"

        out = self.stdout.strip()
        err = self.stderr.strip()
        combined = []
        if out:
            combined.append(f"[STDOUT]\n{out}")
        if err:
            combined.append(f"[STDERR]\n{err}")

        text = "\n".join(combined) if combined else "[No output]"
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n... [TRUNCATED {len(text) - MAX_OUTPUT_CHARS} chars]"

        sanitized = redact_secrets(text)
        status_str = "SUCCESS" if self.exit_code == 0 else f"FAILED_EXIT_{self.exit_code}"
        return f"<COMMAND_OUTPUT status=\"{status_str}\">\n{sanitized}\n</COMMAND_OUTPUT>"


def check_command_allowed(cmd_str: str, role_allowlist: set[str] | list[str] | None = None) -> tuple[bool, str]:
    """Validate a command against global denylist and role allowlists."""
    if not cmd_str.strip():
        return False, "Empty command"

    try:
        parts = shlex.split(cmd_str)
    except ValueError as exc:
        return False, f"Malformed shell syntax: {exc}"

    if not parts:
        return False, "Empty command"

    base_cmd = Path(parts[0]).name

    # 1. Denylist check
    if base_cmd in STRICT_DENYLIST:
        return False, f"Command '{base_cmd}' is strictly forbidden by swarm security policy"

    # 2. Allowlist check
    if role_allowlist is not None:
        allowed = set(role_allowlist) | SAFE_UTILITIES
    else:
        allowed = set(GLOBAL_ALLOWLIST)

    if base_cmd not in allowed:
        return False, f"Command '{base_cmd}' is not in the allowlist for this role"

    # 3. Check for obvious pipeline/chain attempts with forbidden commands
    for token in parts[1:]:
        t_base = Path(token).name
        if t_base in STRICT_DENYLIST:
            return False, f"Argument/subcommand '{t_base}' is strictly forbidden"

    return True, ""


class Sandbox(ABC):
    """Abstract interface for executing sandboxed commands."""

    @abstractmethod
    def run_command(
        self,
        command: str,
        worktree_dir: Path | str,
        timeout: float = DEFAULT_TIMEOUT_S,
        role_allowlist: set[str] | list[str] | None = None,
    ) -> CommandResult:
        """Run a command within the worktree sandbox."""


BaseSandbox = Sandbox


class DockerSandbox(Sandbox):
    """Executes commands inside an isolated Docker container."""

    def __init__(
        self,
        image: str = DEFAULT_SANDBOX_IMAGE,
        cpus: str = DEFAULT_CPUS,
        memory: str = DEFAULT_MEMORY,
        pids_limit: int = DEFAULT_PIDS,
        unsafe_local_exec: bool = False,
    ) -> None:
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.pids_limit = pids_limit
        env_unsafe = os.getenv("UNSAFE_LOCAL_EXEC", "false").lower() in ("true", "1", "yes")
        self.unsafe_local_exec = unsafe_local_exec or env_unsafe
        self.docker_available = shutil.which("docker") is not None

    def build_docker_args(self, command: str, worktree_dir: Path) -> list[str]:
        """Construct the exact docker CLI invocation command list."""
        wt_resolved = str(worktree_dir.resolve())
        return [
            "docker", "run", "--rm",
            "--network", "none",
            "--user", "1000:1000",
            "--cpus", str(self.cpus),
            "--memory", str(self.memory),
            "--pids-limit", str(self.pids_limit),
            "-v", f"{wt_resolved}:/workspace",
            "-w", "/workspace",
            self.image,
            "sh", "-c", command,
        ]

    def run_command(
        self,
        command: str,
        worktree_dir: Path | str,
        timeout: float = DEFAULT_TIMEOUT_S,
        role_allowlist: set[str] | list[str] | None = None,
    ) -> CommandResult:
        """Validate and execute a command inside Docker (or fallback if unsafe_local_exec)."""
        wt_path = Path(worktree_dir).resolve()
        start_time = time.monotonic()

        # 1. Enforce command allowlist
        allowed, reason = check_command_allowed(command, role_allowlist)
        if not allowed:
            logger.warning("Command rejected by security policy: %s (%s)", command, reason)
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr="",
                duration_s=0.0,
                denied=True,
                denial_reason=reason,
            )

        # 2. Check Docker availability
        if not self.docker_available:
            if not self.unsafe_local_exec:
                err_msg = (
                    "Docker is not available and sandboxed execution is required. "
                    "Refusing to execute command without container isolation. "
                    "Pass --unsafe-local-exec to override (NOT recommended)."
                )
                logger.error(err_msg)
                return CommandResult(
                    command=command,
                    exit_code=127,
                    stdout="",
                    stderr=err_msg,
                    duration_s=0.0,
                    denied=True,
                    denial_reason="Docker unavailable and unsafe-local-exec is False",
                )
            else:
                logger.warning(
                    "⚠️  RUNNING UNSANDBOXED ON HOST (unsafe-local-exec enabled): %s",
                    command,
                )
                return self._run_local(command, wt_path, timeout)

        # 3. Execute inside Docker
        args = self.build_docker_args(command, wt_path)
        try:
            res = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.monotonic() - start_time
            return CommandResult(
                command=command,
                exit_code=res.returncode,
                stdout=res.stdout,
                stderr=res.stderr,
                duration_s=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start_time
            return CommandResult(
                command=command,
                exit_code=124,
                stdout="",
                stderr="Execution timed out inside container",
                duration_s=elapsed,
                timed_out=True,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            return CommandResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"Docker execution error: {exc}",
                duration_s=elapsed,
            )

    def _run_local(self, command: str, cwd: Path, timeout: float) -> CommandResult:
        """Fallback local runner only when explicitly permitted via --unsafe-local-exec."""
        start = time.monotonic()
        env = dict(os.environ)
        # Ensure active Python environment bin dir (e.g. .venv/bin) is in PATH
        py_bin_dir = str(Path(sys.executable).parent)
        env["PATH"] = f"{py_bin_dir}:{env.get('PATH', '')}"
        cwd_str = str(cwd.resolve())
        env["PYTHONPATH"] = f"{cwd_str}:{cwd_str}/src:{env.get('PYTHONPATH', '')}".rstrip(":")

        # If command uses python CLI utilities, map to active Python interpreter
        cmd_to_run = command
        trimmed = cmd_to_run.strip()
        first_token = trimmed.split()[0] if trimmed else ""
        if first_token in ("pytest", "ruff", "mypy", "black", "flake8"):
            cmd_to_run = f'"{sys.executable}" -m ' + trimmed
        elif first_token in ("python", "python3"):
            cmd_to_run = f'"{sys.executable}" ' + trimmed[len(first_token):].lstrip()

        try:
            res = subprocess.run(
                ["sh", "-c", cmd_to_run],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
            return CommandResult(
                command=command,
                exit_code=res.returncode,
                stdout=res.stdout,
                stderr=res.stderr,
                duration_s=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                command=command,
                exit_code=124,
                stdout="",
                stderr="Local command timed out",
                duration_s=time.monotonic() - start,
                timed_out=True,
            )


class MockSandbox(Sandbox):
    """Scriptable Mock Sandbox for testing without Docker or host execution."""

    def __init__(self) -> None:
        self.scripted_outputs: list[CommandResult | Exception] = []
        self.executed_commands: list[dict[str, Any]] = []

    def add_result(
        self,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        timed_out: bool = False,
    ) -> None:
        self.scripted_outputs.append(
            CommandResult(
                command="mock",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_s=0.1,
                timed_out=timed_out,
            )
        )

    def script_command(
        self,
        command_pattern: str = "",
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        timed_out: bool = False,
    ) -> None:
        """Alias for add_result allowing command pattern documentation in tests."""
        self.add_result(stdout=stdout, stderr=stderr, exit_code=exit_code, timed_out=timed_out)

    def run_command(
        self,
        command: str,
        worktree_dir: Path | str,
        timeout: float = DEFAULT_TIMEOUT_S,
        role_allowlist: set[str] | list[str] | None = None,
    ) -> CommandResult:
        self.executed_commands.append({
            "command": command,
            "worktree_dir": str(worktree_dir),
            "timeout": timeout,
        })

        # Security check applies even to MockSandbox
        allowed, reason = check_command_allowed(command, role_allowlist)
        if not allowed:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr="",
                duration_s=0.0,
                denied=True,
                denial_reason=reason,
            )

        if self.scripted_outputs:
            item = self.scripted_outputs.pop(0)
            if isinstance(item, Exception):
                raise item
            item.command = command
            return item

        # Default mock output
        return CommandResult(
            command=command,
            exit_code=0,
            stdout="mock output",
            stderr="",
            duration_s=0.05,
        )

